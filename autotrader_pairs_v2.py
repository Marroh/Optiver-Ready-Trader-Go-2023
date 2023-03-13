"""
ETF_VOLUME_THRESHOLD    PROFIT_THRESHOLD    Net profit  data
5000                    300                 11385       1
"""
import asyncio
import itertools

from typing import List
import numpy as np
import time

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

# MINIMUM_BID = 1
# MAXIMUM_ASK = 2 ** 31 - 1
# TOP_LEVEL_COUNT = 5
# FILL_AND_KILL = 0  # Fill and kill orders trade immediately if possible, otherwise they are cancelled
# GOOD_FOR_DAY = 1  # Good for day orders remain in the market until they trade or are explicitly cancelled
# IMMEDIATE_OR_CANCEL = FILL_AND_KILL
# LIMIT_ORDER = GOOD_FOR_DAY
# FAK = FILL_AND_KILL
# GFD = GOOD_FOR_DAY
# F = FILL_AND_KILL
# G = GOOD_FOR_DAY

# TODO: 1. [Doing] Create a timer, record the time for each trade, make sure order < 50/s
#       2. [Delay] Buy maximun lots if an instrument cost zero.
#       3. [Done] Consider available volume before trading.
#       4. [Done](Help nothing) Dynamic profit threshold.
#       5. [Discard](Unnecessary) Pairs trading all profitable volume.
#       6. [Discard](SNR is too low making ML prediction intractable) Price prediction. 
#       7. [Done] Trade when ETF is rising up.


LOT_SIZE = 10
POSITION_LIMIT = 100
HEDGE_POSITION_LIMIT = POSITION_LIMIT * 1
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
PROFIT_THRESHOLD = 3 * TICK_SIZE_IN_CENTS
ETF_VOLUME_THRESHOLD = 5000

class AutoTrader(BaseAutoTrader):
    """
    A slightly redical AutoTrader, who trades when pairs trading 100% profit, or ETF is in short supply.
    Once (ETF best ask price < Future best bid price) or (Future best ask price < ETF best bid price),
    trade the maxinum lots of instrument then hedge at another.
    Meanwhile capture the signal of ETF in short supply, and trade the maxinum to 10 lots of ETF.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.hedge_bids = set()
        self.hedge_asks = set()
        self.future_bids = set()
        self.future_asks = set()
        self.no_hedge_bids = set()
        self.no_hedge_asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.etf_position = self.future_position = 0
        self.last_future_bid_price = 0
        self.base_t = time.time()
        self.unhedge_t = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.hedge_bids or client_order_id in self.hedge_asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)
        print("error with order {}: {}".format(client_order_id, error_message.decode()))

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.future_asks:
            self.future_position -= volume
        elif client_order_id in self.future_bids:
            self.last_future_bid_price = price
            self.future_position += volume
        # print("received hedge filled for order {} with average price {} and volume {}".format(client_order_id, price,
                                                                                            #  volume))

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        global PROFIT_THRESHOLD
        # # print('Order book\ninstrument:{}\nsequence number:{}\nask prices:{}\nask volumes:{}\nbid prices:{}\nbid volumes:{}\n'
        #       .format(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes))
        # # print('Instrument:{}, time:{}'.format(instrument, time.time()))
        if instrument == Instrument.FUTURE:
            self.future_mid_price = (ask_prices[0] + bid_prices[0]) / 2
            self.future_best_ask = ask_prices[0]
            self.future_best_bid = bid_prices[0]
            self.future_ask_prices = ask_prices
            self.future_bid_prices = bid_prices
            self.future_ask_volume = ask_volumes
            self.future_bid_volume = bid_volumes

        elif instrument == Instrument.ETF:
            self.etf_mid_price = (ask_prices[0] + bid_prices[0]) / 2
            self.etf_best_ask = ask_prices[0]
            self.etf_best_bid = bid_prices[0]
            self.etf_ask_prices = ask_prices
            self.etf_bid_prices = bid_prices
            self.etf_ask_volume = ask_volumes
            self.etf_bid_volume = bid_volumes
    
            buy_profit = self.future_best_bid - self.etf_best_ask
            sell_profit = self.etf_best_bid - self.future_best_ask

            # tend to sell when position is high, buy when position is low (or even negative)
            # price_adjustment = - (self.etf_position // (LOT_SIZE*5)) * TICK_SIZE_IN_CENTS
            price_adjustment = 0
            new_bid_price = bid_prices[0] + price_adjustment if bid_prices[0] != 0 else 0
            new_ask_price = ask_prices[0] + price_adjustment if ask_prices[0] != 0 else 0

            # cancel the old bid/ask order price and reorder, unless
            # 1. the price is zero
            # 2. or the price is the same as the previous one
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            # if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
            #     self.send_cancel_order(self.ask_id)
            #     # print("cancel ask order {}".format(self.ask_id))
            #     self.ask_id = 0

            # Trade when ETF price will go up. related theory: Order book flow.
            etf_volume_spread = self.etf_bid_volume[0] - self.etf_ask_volume[0]
            if self.bid_id == 0 and self.ask_id == 0 and etf_volume_spread >= ETF_VOLUME_THRESHOLD and self.etf_position < POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                # print('1here | bid id {}, ask id {}'.format(self.bid_id, self.ask_id))
                self.bid_price = self.etf_best_bid
                volume = min(self.etf_bid_volume[0], POSITION_LIMIT - self.etf_position, LOT_SIZE)
                self.send_insert_order(self.bid_id, Side.BUY, bid_price, volume, Lifespan.GOOD_FOR_DAY)
                self.no_hedge_bids.add(self.bid_id)

            # Pairs trading.
            can_bid_and_hedge = buy_profit >= PROFIT_THRESHOLD and self.etf_position < HEDGE_POSITION_LIMIT and self.future_position > -HEDGE_POSITION_LIMIT
            if can_bid_and_hedge:  # ETF is cheaper
                bid_id = next(self.order_ids)
                bid_price = self.etf_best_ask
                # print('2here | bid id {}, ask id {}'.format(bid_id, self.ask_id))
                etf_best_ask_volume = self.etf_ask_volume[self.etf_ask_prices.index(self.etf_best_ask)]
                available_volume = min(etf_best_ask_volume, HEDGE_POSITION_LIMIT - self.etf_position, HEDGE_POSITION_LIMIT + self.future_position)
                self.send_insert_order(bid_id, Side.BUY, bid_price, int(available_volume), Lifespan.GOOD_FOR_DAY)
                self.hedge_bids.add(bid_id)
            
            can_ask_and_hedge = sell_profit >= PROFIT_THRESHOLD and self.etf_position > -HEDGE_POSITION_LIMIT and self.future_position < HEDGE_POSITION_LIMIT
            if can_ask_and_hedge:  # Future is cheaper
                ask_id = next(self.order_ids)
                ask_price = self.etf_best_bid
                # self.ask_price = new_ask_price
                # print('3here | bid id {}, ask id {}'.format(self.bid_id, ask_id))
                etf_best_bid_volume = self.etf_bid_volume[self.etf_bid_prices.index(self.etf_best_bid)]
                available_volume = min(etf_best_bid_volume, HEDGE_POSITION_LIMIT + self.etf_position, HEDGE_POSITION_LIMIT - self.future_position)
                # print(ask_id, Side.SELL, self.ask_price, available_volume, Lifespan.GOOD_FOR_DAY)
                self.send_insert_order(ask_id, Side.SELL, ask_price, int(available_volume), Lifespan.GOOD_FOR_DAY)
                self.hedge_asks.add(ask_id)

            if abs(abs(self.etf_position) - abs(self.future_position)) > 10:
                self.unhedge_t = time.time() - self.base_t
            else:
                self.base_t = time.time()
                
            # # adaptive profit threshold
            # if self.bid_id == 0 and self.ask_id == 0:
            #     untrade_time = time.time() - self.base_t
            #     if untrade_time > 10:
            #         PROFIT_THRESHOLD = PROFIT_THRESHOLD - 100 if PROFIT_THRESHOLD > 100 else 100
            # else:
            #     self.base_t = time.time()
            #     PROFIT_THRESHOLD = 3 * TICK_SIZE_IN_CENTS


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        # print('Order Filled client_order_id:{} fill_volume:{} price:{}'
            #   .format(client_order_id, volume, price))
        if client_order_id in self.hedge_bids:
            self.etf_position += volume
            future_ask_id = next(self.order_ids)
            self.send_hedge_order(future_ask_id, Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.future_asks.add(future_ask_id)
        elif client_order_id in self.hedge_asks:
            self.etf_position -= volume
            future_bid_id = next(self.order_ids)
            self.send_hedge_order(future_bid_id, Side.BID, MAX_ASK_NEAREST_TICK, volume)
            self.future_bids.add(future_bid_id)

        elif client_order_id in self.no_hedge_bids:
            # TODO: you have 1min to choose a good price
            self.etf_position += volume
            self.ask_id = next(self.order_ids)
            if self.unhedge_t > 59:
                ask_price = self.etf_best_bid
            else:
                ask_price = max(self.etf_best_ask, price + TICK_SIZE_IN_CENTS)
            self.send_insert_order(self.ask_id, Side.SELL, ask_price, volume, Lifespan.GOOD_FOR_DAY)
            self.no_hedge_asks.add(self.ask_id)

        elif client_order_id in self.no_hedge_asks:
            self.etf_position -= volume

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        # print('Order status client_order_id:{} fill_volume:{} remaining_volume:{} fees:{}'
            #   .format(client_order_id, fill_volume, remaining_volume, fees))
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.hedge_bids.discard(client_order_id)
            self.hedge_asks.discard(client_order_id)
            self.no_hedge_bids.discard(client_order_id)
            self.no_hedge_asks.discard(client_order_id)
            self.future_asks.discard(client_order_id)
            self.future_bids.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        # # print('Trading ticks\ninstrument:{}\nsequence number:{}\nask prices:{}\nask volumes:{}\nbid prices:{}\nbid volumes:{}\n'
        #       .format(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes))
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
