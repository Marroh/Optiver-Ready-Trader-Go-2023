# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools

from typing import List

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


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
PROFIT_THRESHOLD = 3 * TICK_SIZE_IN_CENTS



# TODO: 1. Create a timer, record the time for each trade, make sure order < 50/s
#       2. Buy maximun lots if an instrument cost zero.
#       3. [Done] Consider available volume before trading.
#       4. Dynamic profit threshold.
#       5. Adaptive order size.
#       6. [Doing]Price prediction.


class AutoTrader(BaseAutoTrader):
    """
    A conservative AutoTrader, who only trade when pairs trading 100% profit.
    Once (ETF best ask price < Future best bid price) or (Future best ask price < ETF best bid price),
    trade the maxinum lots of instrument then hedge at another.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.no_hedge_bids = set()
        self.no_hedge_asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.etf_position = self.future_position = 0
        self.train_data = []

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
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
        if client_order_id == self.future_bid_id:
            self.last_future_bid_price = price
        print("received hedge filled for order {} with average price {} and volume {}".format(client_order_id, price,
                                                                                             volume))

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
        # print('Order book\ninstrument:{}\nsequence number:{}\nask prices:{}\nask volumes:{}\nbid prices:{}\nbid volumes:{}\n'
        #       .format(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes))
        # print('Instrument:{}, time:{}'.format(instrument, time.time()))
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
            # price_adjustment = - (self.etf_position // LOT_SIZE) * TICK_SIZE_IN_CENTS
            price_adjustment = 0
            new_bid_price = bid_prices[0] + price_adjustment if bid_prices[0] != 0 else 0
            new_ask_price = ask_prices[0] + price_adjustment if ask_prices[0] != 0 else 0

            # cancel the old bid/ask order price and reorder, unless
            # 1. the price is zero
            # 2. or the price is the same as the previous one
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                print("cancel bid order {}".format(self.bid_id))
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                print("cancel ask order {}".format(self.ask_id))
                self.ask_id = 0

            # balance position
            sell_is_profitable = self.etf_best_bid > self.bid_price
            if self.ask_id == 0 and sell_is_profitable and self.etf_position >= POSITION_LIMIT * 0.9:
                self.ask_id = next(self.order_ids)
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE * 5, Lifespan.GOOD_FOR_DAY)
                self.no_hedge_asks.add(self.ask_id)
            
            buy_is_profitable = self.future_best_bid > self.last_future_bid_price
            if self.bid_id == 0 and buy_is_profitable and self.etf_position <= -POSITION_LIMIT * 0.9:
                self.bid_id = next(self.order_ids)
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE * 5, Lifespan.GOOD_FOR_DAY)
                self.no_hedge_bids.add(self.bid_id)
            print('here | bid id {}, ask id {}'.format(self.bid_id, self.ask_id))

            if self.bid_id == 0 and buy_profit >= PROFIT_THRESHOLD and self.etf_position < POSITION_LIMIT * 0.9:  # ETF is cheaper
                self.bid_id = next(self.order_ids)
                self.bid_price = self.etf_best_ask
                # self.bid_price = new_bid_price
                etf_best_ask_volume = self.etf_ask_volume[self.etf_ask_prices.index(self.etf_best_ask)]
                available_volume = min(etf_best_ask_volume, POSITION_LIMIT - self.etf_position)
                self.send_insert_order(self.bid_id, Side.BUY, self.bid_price, available_volume, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)

            if self.ask_id == 0 and sell_profit >= PROFIT_THRESHOLD and self.etf_position > -POSITION_LIMIT * 0.9:  # Future is cheaper
                self.ask_id = next(self.order_ids)
                self.ask_price = self.etf_best_bid
                # self.ask_price = new_ask_price
                etf_best_bid_volume = self.etf_bid_volume[self.etf_bid_prices.index(self.etf_best_bid)]
                available_volume = min(etf_best_bid_volume, POSITION_LIMIT + self.etf_position)
                self.send_insert_order(self.ask_id, Side.SELL, self.ask_price, available_volume, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        print('Order Filled client_order_id:{} fill_volume:{} price:{}'
              .format(client_order_id, volume, price))
        if client_order_id in self.bids:
            self.etf_position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.etf_position -= volume
            self.future_bid_id = next(self.order_ids)
            self.send_hedge_order(self.future_bid_id, Side.BID, MAX_ASK_NEAREST_TICK, volume)
        elif client_order_id in self.no_hedge_bids:
            # TODO: you have 1min to choose a good price
            pass
        elif client_order_id in self.no_hedge_asks:
            pass

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        print('Order status client_order_id:{} fill_volume:{} remaining_volume:{} fees:{}'
              .format(client_order_id, fill_volume, remaining_volume, fees))
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)
            self.no_hedge_bids.discard(client_order_id)
            self.no_hedge_asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        # print('Trading ticks\ninstrument:{}\nsequence number:{}\nask prices:{}\nask volumes:{}\nbid prices:{}\nbid volumes:{}\n'
        #       .format(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes))
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
