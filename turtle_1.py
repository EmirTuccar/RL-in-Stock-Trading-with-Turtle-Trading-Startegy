import gym
from gym import spaces
import numpy as np
import pandas as pd
from enum import Enum

class Actions(Enum):
    Hold = 0
    Buy1 = 1
    Buy2 = 2
    Sell1 = 3
    Sell2 = 4

class TurtleTradingEnv(gym.Env):
    def __init__(self, prices, signal_features=None, window_size=20, frame_bound=(20, None)):
        super(TurtleTradingEnv, self).__init__()

        self.original_prices = prices
        self.original_signals = signal_features
        self.window_size = window_size
        self.frame_bound = frame_bound

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1] if self.frame_bound[1] else len(prices)

        self.prices = prices[start:end]
        self.signal_features = signal_features[start:end] if signal_features is not None else None

        self.current_step = self.window_size
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.position = 0
        self.short_position = 0
        self.short_entry_price = None
        self.total_reward = 0
        self.total_profit = self.initial_cash
        self.entry_prices = []
        self.risk_per_trade = 0.01
        self.last_action = None
        self.last_buy_price = None
        self.pyramid_price = None
        self.trade_log = []

        obs_size = 3 + (self.signal_features.shape[1] if self.signal_features is not None else 3)
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0
        self.short_position = 0
        self.short_entry_price = None
        self.total_reward = 0
        self.total_profit = self.initial_cash
        self.entry_prices = []
        self.last_action = None
        self.last_buy_price = None
        self.pyramid_price = None
        self.trade_log = []
        return self._get_obs(), {}

    def _log_trade(self, action_type, units, price):
        self.trade_log.append({
            'step': self.current_step,
            'action': action_type,
            'units': units,
            'price': price,
            'cash': self.cash,
            'position': self.position,
            'short_position': self.short_position,
            'portfolio': self._get_portfolio_value(price)
        })

    def _long_entry(self, units, price):
        if self.position == 0 and self.short_position == 0:
            self._buy(units, price)
            self.pyramid_price = price
            self._log_trade('long_entry', units, price)

    def _long_exit(self, units, price):
        if self.position >= units:
            self._sell(units, price)
            self._log_trade('long_exit', units, price)
            if self.last_buy_price and price < self.last_buy_price:
                print("💸 Sold at a loss")

    def _short(self, units, price):
        self.short_position += units
        self.short_entry_price = price
        self._log_trade('short_entry', units, price)
        print(f"🔻 SHORT {units} @ {price:.2f} | ShortPOS: {self.short_position}, Portfolio: {self._get_portfolio_value(price):.2f}")

    def _cover(self, units, price):
        if self.short_position >= units:
            gain = (self.short_entry_price - price) * units
            self.cash += gain
            self.short_position -= units
            self._log_trade('short_exit', units, price)
            print(f"✅ COVER {units} @ {price:.2f} | ShortPOS: {self.short_position}, Cash: {self.cash:.2f}, Portfolio: {self._get_portfolio_value(price):.2f}")
        else:
            print(f"❌ COVER FAILED: Tried {units}, Only Short {self.short_position}")

    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)
    
    def step(self, action):
        price = self.prices[self.current_step]

        high_20 = np.max(self.prices[self.current_step - 20:self.current_step])
        low_20 = np.min(self.prices[self.current_step - 20:self.current_step])
        high_10 = np.max(self.prices[self.current_step - 10:self.current_step])
        low_10 = np.min(self.prices[self.current_step - 10:self.current_step])
        atr = self._calculate_atr(self.current_step, period=14)

        max_position = self._calculate_max_position(price, atr)
        requested_units = 1 if action == Actions.Buy1.value else 2 if action == Actions.Buy2.value else 0
        allowed_units = min(requested_units, max_position - self.position)

        prev_action = self.last_action
        self.last_action = action

        if action in [Actions.Buy1.value, Actions.Buy2.value]:
            if price > high_20:
                if allowed_units > 0:
                    self._long_entry(allowed_units, price)
                else:
                    print(f"⛔ BUY SKIPPED: Price broke high_20 but allowed_units={allowed_units}, position={self.position}, max={max_position}")
            else:
                print("❌ Turtle BUY rule not triggered")

        if self.position < max_position and self.pyramid_price:
            if price >= self.pyramid_price + 0.5 * atr:
                self._buy(1, price)
                self.pyramid_price = price

        if action in [Actions.Sell1.value, Actions.Sell2.value]:
            if price < low_10:
                self._long_exit(1 if action == Actions.Sell1.value else 2, price)
            else:
                print("❌ Turtle SELL rule not triggered")

        if self.position == 0 and self.short_position == 0 and price < low_20:
            self._short(1, price)

        if self.short_position > 0 and price > high_10:
            self._cover(self.short_position, price)

        #if self.short_position > 0 and self.short_entry_price and price > self.short_entry_price + 4 * atr:
            #print("🔺 SHORT STOP LOSS triggered")
            #self._cover(self.short_position, price)

        #if self.position > 0 and self.last_buy_price and price < self.last_buy_price - 4 * atr:
            #print("🔻 STOP LOSS triggered by Turtle rule")
            #self._sell(self.position, price)

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        self._update_profit(price)
        reward = (self.total_profit - self.initial_cash) / self.initial_cash

        if self.position > 0:
            current_profit = sum((price - entry) for entry in self.entry_prices)
            if current_profit > 0:
                reward += 0.1
            if current_profit < 0:
                reward -= 0.05

        if price > high_20 and action in [Actions.Buy1.value, Actions.Buy2.value]:
            reward += 0.2

        if price < low_10 and action in [Actions.Sell1.value, Actions.Sell2.value]:
            reward += 0.2

        # Penalize if agent sells at a loss
        if action in [Actions.Sell1.value, Actions.Sell2.value] and self.last_buy_price and price < self.last_buy_price:
            reward -= 0.3

        if prev_action is not None and action != Actions.Hold.value and prev_action != Actions.Hold.value:
            reward -= 0.1

        if self.position == 0 and action == Actions.Hold.value:
            reward -= 0.01

        if action in [Actions.Buy1.value, Actions.Buy2.value] and price <= high_20:
            distance = high_20 - price
            penalty = (distance / high_20) * 2
            reward -= penalty

        if action in [Actions.Buy1.value, Actions.Buy2.value] and self.last_buy_price:
            diff = abs(price - self.last_buy_price)
            if diff < atr:
                penalty = max(0, (1 - (diff / atr)) * 2)
                reward -= penalty

        self.total_reward += reward

        obs = self._get_obs()
        print(f"Price: {price:.2f}, High_20: {high_20:.2f}, Low_10: {low_10:.2f}, ATR: {atr:.2f}, LastBuy: {self.last_buy_price}")

        return obs, reward, False, done, {}

    def _buy(self, units, price):
        cost = units * price
        if self.cash >= cost:
            self.cash -= cost
            self.position += units
            self.last_buy_price = price
            print(f"✅ BUY {units} @ {price:.2f} | POS: {self.position}, CASH: {self.cash:.2f}, Portfolio: {self._get_portfolio_value(price):.2f}")
        else:
            print(f"❌ BUY FAILED: Need ${cost:.2f}, Have ${self.cash:.2f}")

    def _sell(self, units, price):
        if self.position >= units:
            self.cash += units * price
            self.position -= units
            print(f"✅ SELL {units} @ {price:.2f} | POS: {self.position}, CASH: {self.cash:.2f}, Portfolio: {self._get_portfolio_value(price):.2f}")
        else:
            print(f"❌ SELL FAILED: Tried {units}, Only Have {self.position}")

    def _calculate_max_position(self, price, atr):
        if atr == 0:
            return 0
        risk_dollars = self.risk_per_trade * self.total_profit
        unit_risk = atr
        return int(risk_dollars / unit_risk)

    def _update_profit(self, price):
        short_value = self.short_position * (self.short_entry_price - price) if self.short_position > 0 else 0
        self.total_profit = self.cash + self.position * price + short_value

    def _get_obs(self):
        price = self.prices[self.current_step]
        if self.signal_features is not None:
            signals = self.signal_features[self.current_step]
        else:
            signals = self._get_signal_features()
        return np.array([price, self.position, self.cash, *signals], dtype=np.float32)

    def _get_signal_features(self):
        high_20 = np.max(self.prices[self.current_step - 20:self.current_step])
        low_20 = np.min(self.prices[self.current_step - 20:self.current_step])
        high_10 = np.max(self.prices[self.current_step - 10:self.current_step])
        low_10 = np.min(self.prices[self.current_step - 10:self.current_step])
        atr = self._calculate_atr(self.current_step, period=14)
        return [high_20, low_20, high_10, low_10, atr]

    def _calculate_atr(self, idx, period=14):
        if idx < period + 1:
            return 0.0
        tr_list = []
        for i in range(idx - period + 1, idx + 1):
            high = self.original_signals[i][1]
            low = self.original_signals[i][2]
            prev_close = self.prices[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        return np.mean(tr_list)

    def _get_portfolio_value(self, price):
        short_value = self.short_position * (self.short_entry_price - price) if self.short_position > 0 else 0
        return self.cash + self.position * price + short_value


    def _buy(self, units, price):
        cost = units * price
        if self.cash >= cost:
            self.cash -= cost
            self.position += units
            self.last_buy_price = price
            print(f"✅ BUY {units} @ {price:.2f} | POS: {self.position}, CASH: {self.cash:.2f}, Portfolio: {self._get_portfolio_value(price):.2f}")
        else:
            print(f"❌ BUY FAILED: Need ${cost:.2f}, Have ${self.cash:.2f}")

    def _sell(self, units, price):
        if self.position >= units:
            self.cash += units * price
            self.position -= units
            print(f"✅ SELL {units} @ {price:.2f} | POS: {self.position}, CASH: {self.cash:.2f}, Portfolio: {self._get_portfolio_value(price):.2f}")
        else:
            print(f"❌ SELL FAILED: Tried {units}, Only Have {self.position}")

    def _calculate_max_position(self, price, atr):
        if atr == 0:
            return 0
        risk_dollars = self.risk_per_trade * self.total_profit
        unit_risk = atr
        return int(risk_dollars / unit_risk)

    def _update_profit(self, price):
        short_value = self.short_position * (self.short_entry_price - price) if self.short_position > 0 else 0
        self.total_profit = self.cash + self.position * price + short_value

    def _get_obs(self):
        price = self.prices[self.current_step]
        if self.signal_features is not None:
            signals = self.signal_features[self.current_step]
        else:
            signals = self._get_signal_features()
        return np.array([price, self.position, self.cash, *signals], dtype=np.float32)

    def _get_signal_features(self):
        high_20 = np.max(self.prices[self.current_step - 20:self.current_step])
        low_20 = np.min(self.prices[self.current_step - 20:self.current_step])
        high_10 = np.max(self.prices[self.current_step - 10:self.current_step])
        low_10 = np.min(self.prices[self.current_step - 10:self.current_step])
        atr = self._calculate_atr(self.current_step, period=14)
        return [high_20, low_20, high_10, low_10, atr]

    def _calculate_atr(self, idx, period=14):
        if idx < period + 1:
            return 0.0
        tr_list = []
        for i in range(idx - period + 1, idx + 1):
            high = self.original_signals[i][1]
            low = self.original_signals[i][2]
            prev_close = self.prices[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        return np.mean(tr_list)

    def _get_portfolio_value(self, price):
        short_value = self.short_position * (self.short_entry_price - price) if self.short_position > 0 else 0
        return self.cash + self.position * price + short_value
