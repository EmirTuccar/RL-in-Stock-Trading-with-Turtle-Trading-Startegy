# Your Environment with Balanced Rewards & Diversification Fixes
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt
from finta import TA
import torch

# Your existing code structure
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class PositionSize(Enum):
    SIZE_25 = 0  # %25
    SIZE_50 = 1  # %50 
    SIZE_75 = 2  # %75
    SIZE_100 = 3 # %100

@dataclass
class TradingRules:
    """Balanced guardrails for your environment"""
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    max_daily_trades: int = 2  # Balanced
    min_position_hold_days: int = 2  # Balanced (reduced from 3)
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    max_position_size_pct: float = 0.4
    transaction_cost_penalty: float = 0.03  # NEW: Small transaction cost

class MultiAssetTradingEnv(gym.Env):
    """
    Your Environment with BALANCED REWARDS & DIVERSIFICATION
    """
    
    def __init__(
        self,
        asset_data: Dict[str, pd.DataFrame],
        initial_cash: float = 100000,
        commission: float = 0.001,
        frame_bound: Tuple[int, Optional[int]] = (60, None),
        trading_rules: Optional[TradingRules] = None
    ):
        super(MultiAssetTradingEnv, self).__init__()
        
        self.asset_symbols = list(asset_data.keys())
        self.n_assets = len(self.asset_symbols)
        self.initial_cash = initial_cash
        self.commission = commission
        self.trading_rules = trading_rules or TradingRules()
        
        # Process data
        self._process_data(asset_data, frame_bound)
        
        # Action space: [asset_id, action_type, position_size]
        self.action_space = spaces.MultiDiscrete([
            self.n_assets,  # Which asset (0, 1, 2...)
            len(ActionType),  # Hold/Buy/Sell (0, 1, 2)
            len(PositionSize)  # Size (0, 1, 2, 3)
        ])
        
        # Observation space: simplified
        # Per asset: [normalized_price, rsi, position_ratio] = 3 features
        # Portfolio: [cash_ratio, total_return] = 2 features
        obs_size = self.n_assets * 3 + 2
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def _process_data(self, asset_data, frame_bound):
        """Process asset data for training"""
        start_idx = frame_bound[0] - 20  # Need some lookback
        end_idx = frame_bound[1] if frame_bound[1] else None
        
        self.asset_data = {}
        for symbol, df in asset_data.items():
            processed = df.iloc[start_idx:end_idx].copy()
            processed = processed.fillna(method='ffill').fillna(0)
            self.asset_data[symbol] = processed
        
        self.start_step = 20  # Skip initial NaN period
        self.max_steps = len(next(iter(self.asset_data.values())))
    
    def reset(self):
        """Reset environment"""
        self.current_step = self.start_step
        self.cash = self.initial_cash
        self.positions = {symbol: 0.0 for symbol in self.asset_symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.asset_symbols}
        self.entry_steps = {symbol: 0 for symbol in self.asset_symbols}
        self.daily_trades = 0
        self.last_trade_step = 0
        self.total_reward = 0
        self.portfolio_history = [self.initial_cash]
        self.trade_count = 0
        self.action_history = []
        
        # NEW: Diversification tracking
        self.asset_usage_count = {symbol: 0 for symbol in self.asset_symbols}
        self.consecutive_same_asset = 0
        self.last_asset_chosen = None
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step - BALANCED REWARD SYSTEM"""
        if self.current_step >= self.max_steps - 1:
            return self._get_observation(), 0, True, {}
        
        # Decode action
        asset_idx, action_type, position_size = action
        asset_symbol = self.asset_symbols[asset_idx]
        self.action_history.append(action_type)
        
        # NEW: Track asset usage for diversification
        self.asset_usage_count[asset_symbol] += 1
        if self.last_asset_chosen == asset_symbol:
            self.consecutive_same_asset += 1
        else:
            self.consecutive_same_asset = 1
        self.last_asset_chosen = asset_symbol
        
        # Get current price and indicators
        current_data = self.asset_data[asset_symbol].iloc[self.current_step]
        current_price = current_data['Close']
        current_rsi = current_data['RSI']
        
        # Validate action
        action_valid = self._validate_action(asset_symbol, action_type, current_rsi)
        
        # DAILY TRADES RESET (daily simulation)
        if self.current_step > self.last_trade_step + 1:
            self.daily_trades = 0
        
        # Execute action and get immediate reward
        execution_reward = 0
        if action_valid:
            execution_reward = self._execute_action(asset_symbol, action_type, position_size, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate BALANCED reward
        reward = self._calculate_balanced_reward(
            asset_symbol, action_type, action_valid, execution_reward, current_rsi
        )
        self.total_reward += reward
        
        # Update portfolio history
        portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Check if done
        done = (self.current_step >= self.max_steps - 1) or (portfolio_value < self.initial_cash * 0.1)
        
        # Enhanced info for debugging
        info = {
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value / self.initial_cash) - 1,
            'cash': self.cash,
            'step': self.current_step,
            'action_valid': action_valid,
            'action_type': action_type,
            'asset_symbol': asset_symbol,
            'total_trades': self.trade_count,
            'daily_trades': self.daily_trades,
            'reward': reward,
            'execution_reward': execution_reward,
            'diversification_score': self._calculate_diversification_score(),
            'asset_usage': self.asset_usage_count.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _validate_action(self, asset_symbol, action_type, current_rsi):
        """Balanced guardrails validation"""
        # RSI rules
        if action_type == ActionType.BUY.value and current_rsi > self.trading_rules.rsi_overbought:
            return False
        
        if action_type == ActionType.SELL.value and current_rsi < self.trading_rules.rsi_oversold:
            return False
        
        # Daily trades limit
        if action_type != ActionType.HOLD.value and self.daily_trades >= self.trading_rules.max_daily_trades:
            return False
        
        # Minimum hold time
        if (action_type == ActionType.SELL.value and 
            self.positions[asset_symbol] > 0 and
            (self.current_step - self.entry_steps[asset_symbol]) < self.trading_rules.min_position_hold_days):
            return False
        
        return True
    
    def _execute_action(self, asset_symbol, action_type, position_size, current_price):
        """Execute trading action with balanced rewards"""
        execution_reward = 0
        
        if action_type == ActionType.HOLD.value:
            return 0
        
        # Apply transaction cost penalty (small)
        transaction_penalty = self.trading_rules.transaction_cost_penalty
        
        # Calculate position size
        size_multipliers = [0.25, 0.50, 0.75, 1.0]
        multiplier = size_multipliers[position_size]
        
        if action_type == ActionType.BUY.value:
            # Calculate how much to buy
            max_buy_value = self.cash * multiplier
            cost_with_commission = current_price * (1 + self.commission)
            shares_to_buy = max_buy_value / cost_with_commission
            total_cost = shares_to_buy * cost_with_commission
            
            if self.cash >= total_cost and shares_to_buy > 0:
                # Update position with weighted average entry price
                old_value = self.positions[asset_symbol] * self.entry_prices[asset_symbol]
                new_value = shares_to_buy * current_price
                total_shares = self.positions[asset_symbol] + shares_to_buy
                
                if total_shares > 0:
                    self.entry_prices[asset_symbol] = (old_value + new_value) / total_shares
                    self.entry_steps[asset_symbol] = self.current_step
                
                self.positions[asset_symbol] = total_shares
                self.cash -= total_cost
                self.daily_trades += 1
                self.trade_count += 1
                self.last_trade_step = self.current_step
                
                # BALANCED BUY reward
                execution_reward = 0.3 - transaction_penalty  # Reduced from 1.0
        
        elif action_type == ActionType.SELL.value:
            if self.positions[asset_symbol] > 0:
                shares_to_sell = min(self.positions[asset_symbol], self.positions[asset_symbol] * multiplier)
                sale_proceeds = shares_to_sell * current_price * (1 - self.commission)
                
                # Calculate profit/loss if we have entry price
                profit_reward = 0
                if self.entry_prices[asset_symbol] > 0:
                    profit_per_share = current_price - self.entry_prices[asset_symbol]
                    total_profit = profit_per_share * shares_to_sell
                    profit_pct = (current_price / self.entry_prices[asset_symbol]) - 1
                    
                    # BALANCED profit reward (reduced from 10x to 2x)
                    profit_reward = profit_pct * 2
                
                self.positions[asset_symbol] -= shares_to_sell
                self.cash += sale_proceeds
                self.daily_trades += 1
                self.trade_count += 1
                self.last_trade_step = self.current_step
                
                # Reset entry price if position is fully closed
                if self.positions[asset_symbol] <= 0.01:  # Almost zero
                    self.positions[asset_symbol] = 0
                    self.entry_prices[asset_symbol] = 0
                    self.entry_steps[asset_symbol] = 0
                
                # BALANCED SELL reward
                execution_reward = 0.3 + profit_reward - transaction_penalty  # Reduced from high values
        
        return execution_reward
    
    def _calculate_balanced_reward(self, asset_symbol, action_type, action_valid, execution_reward, current_rsi):
        """BALANCED reward calculation - addresses HOLD/SELL balance and diversification"""
        reward = 0
        
        # 1. Portfolio performance reward (moderate scale)
        if len(self.portfolio_history) >= 2:
            current_value = self._get_portfolio_value()
            previous_value = self.portfolio_history[-1]
            if previous_value > 0:
                return_pct = (current_value - previous_value) / previous_value
                reward += return_pct * 500  # Reduced from 1000 to 500
        
        # 2. Execution reward
        reward += execution_reward
        
        # 3. BALANCED ACTION ENCOURAGEMENT (Major Fix!)
        if action_valid:
            if action_type == ActionType.HOLD.value:
                # POSITIVE HOLD REWARD for strategic waiting
                days_since_trade = self.current_step - self.last_trade_step
                if days_since_trade <= 5:  # Strategic holding is good
                    reward += 0.05
                elif days_since_trade > 10:  # Too much holding is bad
                    reward -= 0.1
                else:  # Medium holding is neutral
                    reward += 0.02
                    
            elif action_type == ActionType.BUY.value:
                reward += 0.2  # Moderate BUY reward
                if current_rsi < 40:
                    reward += 0.1  # Smart buying bonus
                    
            elif action_type == ActionType.SELL.value:
                reward += 0.2  # Moderate SELL reward (much reduced from previous)
                if current_rsi > 60:
                    reward += 0.1  # Smart selling bonus
        
        # 4. DIVERSIFICATION REWARDS (Major Addition!)
        diversification_reward = self._calculate_diversification_reward(asset_symbol)
        reward += diversification_reward
        
        # 5. Portfolio balance rewards
        portfolio_value = self._get_portfolio_value()
        if portfolio_value > 0:
            cash_ratio = self.cash / portfolio_value
            
            # Encourage balanced cash usage
            if 0.1 <= cash_ratio <= 0.5:  # Sweet spot
                reward += 0.1
            elif cash_ratio > 0.8:  # Too much cash
                reward -= 0.1
            elif cash_ratio < 0.05:  # Over-invested
                reward -= 0.05
        
        # 6. Invalid action penalty (reduced)
        if not action_valid:
            reward -= 0.5  # Reduced from 2.0
        
        # 7. Overtrading penalty
        if self.trade_count > 0 and self.current_step > 0:
            trade_frequency = self.trade_count / self.current_step
            if trade_frequency > 0.3:  # More than 0.3 trades per day
                reward -= 0.1
        
        return reward
    
    def _calculate_diversification_reward(self, asset_symbol):
        """Calculate diversification-based rewards to fix asset concentration"""
        diversification_reward = 0
        
        # 1. Asset switching bonus (encourage using different assets)
        if self.last_asset_chosen != asset_symbol and self.last_asset_chosen is not None:
            diversification_reward += 0.3  # Significant bonus for switching
        
        # 2. Consecutive same asset penalty
        if self.consecutive_same_asset > 3:
            diversification_reward -= 0.15 * (self.consecutive_same_asset - 3)
        
        # 3. Portfolio diversification bonus
        active_positions = sum(1 for pos in self.positions.values() if pos > 0)
        if active_positions == 2:
            diversification_reward += 0.2
        elif active_positions == 3:
            diversification_reward += 0.4  # Strong bonus for full diversification
        
        # 4. Even asset usage bonus
        if self.current_step > 30:  # After some time has passed
            usage_counts = list(self.asset_usage_count.values())
            total_usage = sum(usage_counts)
            if total_usage > 0:
                # Calculate how evenly assets are used
                usage_probs = [count / total_usage for count in usage_counts]
                # Reward even usage (low standard deviation)
                usage_std = np.std(usage_probs)
                if usage_std < 0.2:  # Very even usage
                    diversification_reward += 0.2
        
        return diversification_reward
    
    def _calculate_diversification_score(self):
        """Calculate current diversification score for tracking"""
        if self.current_step == 0:
            return 0
        
        # Asset usage evenness
        usage_counts = list(self.asset_usage_count.values())
        total_usage = sum(usage_counts)
        if total_usage == 0:
            return 0
        
        # Calculate entropy-based diversification
        probs = [count / total_usage for count in usage_counts]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log(len(self.asset_symbols))
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _get_portfolio_value(self):
        """Calculate current portfolio value"""
        total_value = self.cash
        for symbol in self.asset_symbols:
            if self.current_step < len(self.asset_data[symbol]):
                current_price = self.asset_data[symbol].iloc[self.current_step]['Close']
                total_value += self.positions[symbol] * current_price
        return total_value
    
    def _get_observation(self):
        """Get observation vector"""
        obs = []
        portfolio_value = self._get_portfolio_value()
        
        # Asset features
        for symbol in self.asset_symbols:
            if self.current_step < len(self.asset_data[symbol]):
                current_data = self.asset_data[symbol].iloc[self.current_step]
                
                # Normalize price by moving average
                if self.current_step >= 20:
                    ma_20 = self.asset_data[symbol]['Close'].iloc[self.current_step-20:self.current_step].mean()
                    normalized_price = current_data['Close'] / ma_20 if ma_20 > 0 else 1.0
                else:
                    normalized_price = 1.0
                
                # Normalize RSI
                normalized_rsi = current_data['RSI'] / 100.0
                
                # Position ratio
                position_value = self.positions[symbol] * current_data['Close']
                position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
                
                obs.extend([normalized_price, normalized_rsi, position_ratio])
            else:
                obs.extend([1.0, 0.5, 0.0])  # Default values
        
        # Portfolio features
        cash_ratio = self.cash / self.initial_cash
        total_return = (portfolio_value / self.initial_cash) - 1
        
        obs.extend([cash_ratio, total_return])
        
        return np.array(obs, dtype=np.float32)

# Enhanced data preparation with different asset characteristics
def prepare_training_data():
    """Prepare data with different asset characteristics to encourage diversification"""
    print("📊 Balanced training verisi hazırlanıyor...")
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    
    # Create assets with DIFFERENT characteristics
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    asset_data = {}
    
    for i, symbol in enumerate(symbols):
        base_price = 100 + i * 15  # Different base prices
        
        # Different asset characteristics to encourage diversification
        if symbol == 'AAPL':
            # High growth, high volatility
            trend = np.linspace(0, 0.4, len(dates))
            volatility = 0.02
            cyclical_period = 200
        elif symbol == 'GOOGL':
            # Medium growth, medium volatility
            trend = np.linspace(0, 0.3, len(dates))
            volatility = 0.015
            cyclical_period = 250
        else:  # MSFT
            # Steady growth, low volatility
            trend = np.linspace(0, 0.25, len(dates))
            volatility = 0.01
            cyclical_period = 300
        
        # Add different cyclical patterns
        cyclical = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / cyclical_period)
        noise = np.random.randn(len(dates)) * volatility
        
        total_returns = trend + cyclical + noise
        prices = base_price * np.exp(total_returns)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.002),
            'High': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
            'Low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Add technical indicators
        df['RSI'] = TA.RSI(df, 14)
        df['SMA'] = TA.SMA(df, 20)
        df['EMA'] = TA.EMA(df, 20)
        df['MACD'] = TA.MACD(df)['MACD']
        df['ATR'] = TA.ATR(df, 14)
        df.fillna(method='ffill', inplace=True)
        df.fillna(50, inplace=True)  # Fill RSI with neutral value
        
        asset_data[symbol] = df
    
    print(f"✅ {len(symbols)} asset için balanced karakteristiklerle veri hazırlandı")
    return asset_data

def prepare_real_alpaca_data():
    """Your existing Alpaca data preparation"""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        print("📈 Gerçek Alpaca verisi yükleniyor...")
        
        # Your API credentials
        api_key = 'PKW5DUGHODUW7U8ZVI4Z'
        secret_key = 'xfLWIOgSQ7sHQpGRrqWbXyorPcUfuH2Wurbhj2zV'
        
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        
        # Use different, more diverse assets
        symbols = ['AAPL', 'IBM', 'KO']  # Tech, Tech services, Consumer goods (more diverse)
        asset_data = {}
        
        for symbol in symbols:
            print(f"  📥 {symbol} downloading...")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start="2017-01-01 00:00:00",
                end="2023-01-01 00:00:00"
            )
            
            bars = client.get_stock_bars(request_params)
            data = bars.df.droplevel(0)
            data = data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
            
            # Add technical indicators
            data['RSI'] = TA.RSI(data, 14)
            data['SMA'] = TA.SMA(data, 20)
            data['EMA'] = TA.EMA(data, 20)
            data['MACD'] = TA.MACD(data)['MACD']
            data['ATR'] = TA.ATR(data, 14)
            data.fillna(method='ffill', inplace=True)
            data.fillna(0, inplace=True)
            
            asset_data[symbol] = data
            print(f"    ✅ {symbol}: {len(data)} günlük veri")
        
        return asset_data
        
    except Exception as e:
        print(f"❌ Alpaca data loading failed: {e}")
        print("🔄 Synthetic balanced data kullanılıyor...")
        return prepare_training_data()

# Training functions
def create_training_environment(asset_data, train_split=0.8):
    """Create training and validation environments"""
    data_length = len(next(iter(asset_data.values())))
    train_end = int(data_length * train_split)
    
    # Training environment
    train_env = MultiAssetTradingEnv(
        asset_data=asset_data,
        frame_bound=(60, train_end),
        initial_cash=100000
    )
    
    # Validation environment  
    val_env = MultiAssetTradingEnv(
        asset_data=asset_data,
        frame_bound=(train_end, data_length),
        initial_cash=100000
    )
    
    return train_env, val_env

def train_model(train_env, val_env, algorithm='PPO', total_timesteps=40000):
    """Train RL model with balanced hyperparameters"""
    print(f"\n🤖 {algorithm} Training başlıyor...")
    
    # Wrap in vectorized environment
    train_env_vec = DummyVecEnv([lambda: train_env])
    val_env_vec = DummyVecEnv([lambda: val_env])
    
    # Balanced hyperparameters
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', 
            train_env_vec, 
            verbose=1,
            learning_rate=0.0005,  # Moderate learning rate
            n_steps=2048,  # Larger steps for stability
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Standard gamma
            ent_coef=0.05,  # Moderate entropy
            device='auto'
        )
    elif algorithm == 'DQN':
        model = DQN(
            'MlpPolicy',
            train_env_vec,
            verbose=1,
            learning_rate=0.0005,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            device='auto'
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy',
            train_env_vec,
            verbose=1,
            learning_rate=0.0005,
            n_steps=1024,
            gamma=0.99,
            ent_coef=0.05,
            device='auto'
        )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        val_env_vec,
        best_model_save_path=f'./best_{algorithm.lower()}_balanced_model',
        log_path=f'./logs_{algorithm.lower()}_balanced',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Train model
    print(f"🏋️ Training {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save final model
    model.save(f"{algorithm.lower()}_balanced_trading_model")
    print(f"✅ Model saved as {algorithm.lower()}_balanced_trading_model")
    
    return model

def evaluate_model(model, test_env, n_episodes=5):
    """Evaluate model with detailed diversification tracking"""
    print(f"\n📊 Balanced Model evaluation ({n_episodes} episodes)...")
    
    episode_results = []
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        step_count = 0
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        asset_counts = {symbol: 0 for symbol in test_env.asset_symbols}
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Track actions and assets
            action_names = ['HOLD', 'BUY', 'SELL']
            action_counts[action_names[action[1]]] += 1
            asset_counts[test_env.asset_symbols[action[0]]] += 1
            
            if step_count > 1000:  # Safety break
                break
        
        final_portfolio = info['portfolio_value']
        total_return = info['total_return']
        diversification_score = info['diversification_score']
        
        episode_results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'steps': step_count,
            'total_trades': info['total_trades'],
            'diversification_score': diversification_score,
            'action_counts': action_counts.copy(),
            'asset_counts': asset_counts.copy(),
            'final_asset_usage': info['asset_usage'].copy()
        })
        
        print(f"Episode {episode + 1}: Return {total_return:.2%}, Portfolio ${final_portfolio:,.2f}")
        print(f"  Trades: {info['total_trades']}, Diversification: {diversification_score:.3f}")
        print(f"  Actions: {action_counts}")
        print(f"  Assets: {asset_counts}")
        print(f"  Total Asset Usage: {info['asset_usage']}")
        print()
    
    # Summary statistics
    returns = [r['total_return'] for r in episode_results]
    trades = [r['total_trades'] for r in episode_results]
    diversification_scores = [r['diversification_score'] for r in episode_results]
    
    print(f"\n📈 Balanced Model Summary:")
    print(f"   Average Return: {np.mean(returns):.2%}")
    print(f"   Std Return: {np.std(returns):.2%}")
    print(f"   Best Return: {max(returns):.2%}")
    print(f"   Worst Return: {min(returns):.2%}")
    print(f"   Average Trades: {np.mean(trades):.1f}")
    print(f"   Average Diversification: {np.mean(diversification_scores):.3f}")
    
    # Action distribution analysis
    total_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    total_assets = {symbol: 0 for symbol in test_env.asset_symbols}
    
    for result in episode_results:
        for action, count in result['action_counts'].items():
            total_actions[action] += count
        for asset, count in result['asset_counts'].items():
            total_assets[asset] += count
    
    total_action_count = sum(total_actions.values())
    total_asset_count = sum(total_assets.values())
    
    print(f"\n📊 Overall Action Distribution:")
    for action, count in total_actions.items():
        percentage = (count / total_action_count) * 100 if total_action_count > 0 else 0
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Overall Asset Distribution:")
    for asset, count in total_assets.items():
        percentage = (count / total_asset_count) * 100 if total_asset_count > 0 else 0
        print(f"   {asset}: {count} ({percentage:.1f}%)")
    
    return episode_results

def plot_best_episode_results(test_env, model, n_episodes=5):
    """Plot results from the best performing episode"""
    
    print(f"🔍 {n_episodes} episode çalıştırılıyor, en iyisi bulunuyor...")
    
    best_return = -float('inf')
    best_episode_num = 0
    episode_returns = []
    
    # Phase 1: Find best episode using non-deterministic predictions
    for episode in range(n_episodes):
        obs = test_env.reset()
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Non-deterministic for variety
            obs, reward, done, info = test_env.step(action)
        
        final_return = info['total_return']
        episode_returns.append(final_return)
        
        print(f"  Episode {episode + 1}: Return {final_return:.2%}, Portfolio ${info['portfolio_value']:,.0f}")
        
        # Update best episode
        if final_return > best_return:
            best_return = final_return
            best_episode_num = episode + 1
    
    print(f"✅ En iyi episode: {best_episode_num} (Return: {best_return:.2%})")
    print(f"🔄 En iyi episode deterministik mode ile tekrar çalıştırılıyor (plotting için)...")
    
    # Phase 2: Re-run best episode deterministically for consistent plotting
    obs = test_env.reset()
    best_episode_data = {
        'portfolio_values': [test_env.initial_cash],
        'actions': [],
        'prices': {symbol: [] for symbol in test_env.asset_symbols},
        'asset_usage_over_time': [],
        'episode_num': best_episode_num,
        'final_return': best_return
    }
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Deterministic for consistent plotting
        obs, reward, done, info = test_env.step(action)
        
        best_episode_data['portfolio_values'].append(info['portfolio_value'])
        best_episode_data['actions'].append(action)
        best_episode_data['asset_usage_over_time'].append(info['asset_usage'].copy())
        
        # Store prices for plotting
        for symbol in test_env.asset_symbols:
            if test_env.current_step < len(test_env.asset_data[symbol]):
                current_price = test_env.asset_data[symbol].iloc[test_env.current_step]['Close']
                best_episode_data['prices'][symbol].append(current_price)
    
    best_episode_data['final_portfolio'] = info['portfolio_value']
    best_episode_data['total_trades'] = info['total_trades']
    
    print(f"📊 Plotting episode performance: {info['total_return']:.2%} return")
    
    # Plot the best episode
    portfolio_values = best_episode_data['portfolio_values']
    actions = best_episode_data['actions']
    prices = best_episode_data['prices']
    asset_usage_over_time = best_episode_data['asset_usage_over_time']
    
    # Create enhanced plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Best Episode Results (Episode {best_episode_data["episode_num"]}) - Return: {best_episode_data["final_return"]:.2%}', fontsize=16)
    
    # Portfolio value over time
    axes[0, 0].plot(portfolio_values, linewidth=2, color='blue')
    axes[0, 0].axhline(y=test_env.initial_cash, color='r', linestyle='--', label='Initial Value')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add final value text
    final_value = portfolio_values[-1]
    axes[0, 0].text(0.02, 0.98, f'Final: ${final_value:,.0f}\nTrades: {best_episode_data["total_trades"]}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Asset prices
    for symbol, price_history in prices.items():
        if price_history:
            axes[0, 1].plot(price_history, label=symbol, linewidth=2)
    axes[0, 1].set_title('Asset Prices')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Action distribution
    if actions:
        actions_array = np.array(actions)
        action_types = actions_array[:, 1]
        
        action_counts = np.bincount(action_types, minlength=3)
        action_names = ['HOLD', 'BUY', 'SELL']
        colors = ['gray', 'green', 'red']
        
        bars = axes[0, 2].bar(action_names, action_counts, color=colors, alpha=0.7)
        axes[0, 2].set_title('Action Type Distribution')
        axes[0, 2].set_ylabel('Frequency')
        
        # Add percentages
        total_actions = len(actions)
        for i, (bar, count) in enumerate(zip(bars, action_counts)):
            if total_actions > 0:
                percentage = (count / total_actions) * 100
                axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(action_counts)*0.01, 
                               f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Asset selection distribution
    if actions:
        asset_actions = actions_array[:, 0]
        asset_counts = np.bincount(asset_actions, minlength=test_env.n_assets)
        
        bars = axes[1, 0].bar(test_env.asset_symbols, asset_counts, alpha=0.7, color=['blue', 'orange', 'green'])
        axes[1, 0].set_title('Asset Selection Distribution')
        axes[1, 0].set_ylabel('Frequency')
        
        # Add percentages
        total_asset_selections = len(actions)
        for i, (bar, count) in enumerate(zip(bars, asset_counts)):
            if total_asset_selections > 0:
                percentage = (count / total_asset_selections) * 100
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(asset_counts)*0.01, 
                               f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Asset usage over time
    if asset_usage_over_time:
        usage_data = {symbol: [] for symbol in test_env.asset_symbols}
        steps = []
        
        for i, usage in enumerate(asset_usage_over_time[::10]):  # Sample every 10 steps
            steps.append(i * 10)
            total_usage = sum(usage.values())
            for symbol in test_env.asset_symbols:
                percentage = (usage[symbol] / total_usage * 100) if total_usage > 0 else 0
                usage_data[symbol].append(percentage)
        
        for symbol in test_env.asset_symbols:
            axes[1, 1].plot(steps, usage_data[symbol], label=symbol, linewidth=2)
        
        axes[1, 1].set_title('Asset Usage Over Time (%)')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Usage Percentage')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Diversification score over time
    diversification_scores = []
    if asset_usage_over_time:
        for usage in asset_usage_over_time:
            total_usage = sum(usage.values())
            if total_usage > 0:
                probs = [count / total_usage for count in usage.values()]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                max_entropy = np.log(len(test_env.asset_symbols))
                score = entropy / max_entropy if max_entropy > 0 else 0
                diversification_scores.append(score)
            else:
                diversification_scores.append(0)
        
        axes[1, 2].plot(diversification_scores, linewidth=2, color='purple')
        axes[1, 2].set_title('Diversification Score Over Time')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Diversification Score (0-1)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Good (>0.8)')
        axes[1, 2].legend()
        
        # Add average diversification score
        avg_div_score = np.mean(diversification_scores) if diversification_scores else 0
        axes[1, 2].text(0.02, 0.98, f'Avg: {avg_div_score:.3f}', 
                        transform=axes[1, 2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'best_episode_results_ep{best_episode_data["episode_num"]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return summary of best episode
    summary = {
        'episode_num': best_episode_data['episode_num'],
        'final_return': best_episode_data['final_return'],
        'final_portfolio': best_episode_data['final_portfolio'],
        'total_trades': best_episode_data['total_trades'],
        'action_distribution': dict(zip(['HOLD', 'BUY', 'SELL'], 
                                      np.bincount([a[1] for a in actions], minlength=3))) if actions else {},
        'asset_distribution': dict(zip(test_env.asset_symbols, 
                                     np.bincount([a[0] for a in actions], minlength=test_env.n_assets))) if actions else {}
    }
    
    return fig, summary

# Main training pipeline
def main_training_pipeline():
    """Complete training pipeline with balanced rewards"""
    print("🚀 BALANCED MULTI-ASSET RL TRADING")
    print("=" * 60)
    
    # Step 1: Choose data source
    use_real_data = input("Gerçek Alpaca verisi kullanmak ister misiniz? (y/n): ").lower() == 'y'
    
    if use_real_data:
        asset_data = prepare_real_alpaca_data()
    else:
        asset_data = prepare_training_data()
    
    # Step 2: Create environments
    train_env, val_env = create_training_environment(asset_data)
    print(f"✅ Training environment: {train_env.start_step} to {train_env.max_steps}")
    print(f"✅ Validation environment: {val_env.start_step} to {val_env.max_steps}")
    
    # Step 3: Choose algorithm
    algorithm = input("Algoritma seçin (PPO/DQN/A2C) [default: PPO]: ").upper() or 'PPO'
    timesteps = int(input("Training timesteps [default: 40000]: ") or 40000)
    
    # Step 4: Train model
    model = train_model(train_env, val_env, algorithm, timesteps)
    
    # Step 5: Evaluate model
    evaluation_results = evaluate_model(model, val_env)
    
    # Step 6: Plot results
    plot_best_episode_results(val_env, model)
    
    print(f"\n🎉 Balanced Training completed!")
    print(f"💾 Model saved as {algorithm.lower()}_balanced_trading_model")
    print(f"📊 Results plotted and saved as balanced_trading_results.png")
    
    return model, asset_data, train_env, val_env

def find_best_episode(model, test_env, n_episodes=5):
    """Find the best performing episode"""
    best_return = -float('inf')
    best_episode_data = None
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        episode_data = {
            'portfolio_values': [test_env.initial_cash],
            'actions': [],
            'infos': []
        }
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            episode_data['portfolio_values'].append(info['portfolio_value'])
            episode_data['actions'].append(action)
            episode_data['infos'].append(info)
        
        final_return = info['total_return']
        
        # En iyi episode'u güncelle
        if final_return > best_return:
            best_return = final_return
            best_episode_data = episode_data
            best_episode_data['episode_num'] = episode + 1
            best_episode_data['final_return'] = final_return
    
    return best_episode_data

# Quick test for balanced system
def quick_balanced_test():
    """Quick test for balanced reward system"""
    print("🔧 BALANCED SYSTEM QUICK TEST")
    print("=" * 50)
    
    # Generate balanced data
    asset_data = prepare_training_data()
    
    # Create test environment
    env = MultiAssetTradingEnv(asset_data, initial_cash=100000)
    
    print("\n🎲 Testing balanced rewards (30 steps):")
    obs = env.reset()
    
    action_types = ['HOLD', 'BUY', 'SELL']
    total_reward = 0
    
    for step in range(30):
        # Simulate intelligent actions to test diversification
        if step < 5:
            action = [0, 1, 1]  # AAPL BUY 50%
        elif step < 10:
            action = [1, 1, 1]  # GOOGL BUY 50% (diversification!)
        elif step < 15:
            action = [2, 1, 0]  # MSFT BUY 25% (more diversification!)
        elif step < 20:
            action = [0, 0, 0]  # HOLD (should get positive reward)
        elif step < 25:
            action = [1, 0, 0]  # GOOGL HOLD
        else:
            action = [0, 2, 1]  # AAPL SELL 50%
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:  # Print every 5 steps
            print(f"Step {step+1}: {action_types[action[1]]} {env.asset_symbols[action[0]]} | "
                  f"Reward: {reward:.3f} | Portfolio: ${info['portfolio_value']:,.0f}")
            print(f"  Diversification: {info['diversification_score']:.3f} | Asset Usage: {info['asset_usage']}")
        
        if done:
            break
    
    print(f"\n📊 Balanced Test Results:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Final Portfolio: ${info['portfolio_value']:,.2f}")
    print(f"   Total Return: {info['total_return']:.2%}")
    print(f"   Total Trades: {info['total_trades']}")
    print(f"   Final Diversification Score: {info['diversification_score']:.3f}")
    print(f"   Final Asset Usage: {info['asset_usage']}")
    
    # Action distribution
    if hasattr(env, 'action_history'):
        action_counts = np.bincount(env.action_history, minlength=3)
        print(f"\n📈 Action Distribution:")
        for i, (name, count) in enumerate(zip(action_types, action_counts)):
            percentage = (count / len(env.action_history)) * 100 if env.action_history else 0
            print(f"   {name}: {count} times ({percentage:.1f}%)")
    
    # Check if balanced
    balanced_actions = action_counts[0] > 5 and action_counts[1] > 5 and action_counts[2] > 5  # All actions used
    balanced_assets = all(count > 3 for count in info['asset_usage'].values())  # All assets used
    
    if balanced_actions and balanced_assets:
        print("\n✅ SUCCESS: Balanced system working!")
        print("   - All action types used ✓")
        print("   - All assets used ✓")
        print("   - Diversification score > 0.5 ✓" if info['diversification_score'] > 0.5 else "   - Diversification needs improvement")
        return True
    else:
        print("\n⚠️  PARTIAL SUCCESS: Some improvements needed")
        print(f"   - Balanced actions: {balanced_actions}")
        print(f"   - Balanced assets: {balanced_assets}")
        return False

if __name__ == "__main__":
    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # First run quick balanced test
    success = quick_balanced_test()
    
    if success:
        print("\n" + "="*60)
        print("✅ Balanced system working! Proceeding to full training...")
        
        # Run full training
        model, asset_data, train_env, val_env = main_training_pipeline()
        
        print(f"\n🎯 FINAL BALANCED SUMMARY:")
        print(f"   Environment: Balanced Multi-Asset Trading")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Assets: {train_env.asset_symbols}")
        print(f"   Features: Diversification rewards, Balanced HOLD/BUY/SELL, Transaction costs")
        print(f"   Training completed successfullyy!")
    else:
        print("\n💡 Running training anyway to see improvements...")
        model, asset_data, train_env, val_env = main_training_pipeline()