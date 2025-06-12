# ENTEGRASYON REHBERİ
# 1. Bu dosyayı 'academic_evaluation.py' olarak kaydet
# 2. Mevcut sonn.py dosyanız ile aynı klasörde olsun

# ======================================================================
# ACADEMIC STANDARD EVALUATION FOR YOUR EXISTING TRADING CODE
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings("ignore")
# MEVCUT KODUNUZU IMPORT ET
try:
    from sonn import *
    print("✅ Mevcut trading kodunuz başarıyla import edildi!")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("📁 Bu dosyayı sonn.py ile aynı klasöre koyun")

# ======================================================================
# YENİ ACADEMIC FONKSIYONLAR (Mevcut kodunuza eklenti)
# ======================================================================

def train_model_with_seed(train_env, val_env, algorithm='PPO', total_timesteps=40000, seed=42):
    """
    SEED'Lİ TRAINING - Mevcut train_model() fonksiyonunuzun seed'li versiyonu
    """
    print(f"🎲 {algorithm} Training with seed {seed}")
    
    # Set all seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Wrap environments (sizin kodunuzdaki gibi)
    train_env_vec = DummyVecEnv([lambda: train_env])
    val_env_vec = DummyVecEnv([lambda: val_env])
    
    # Create model with seed (sizin parametrelerinizle)
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', 
            train_env_vec, 
            verbose=0,  # Sessiz (multiple runs için)
            seed=seed,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.95,
            ent_coef=0.1,
            device='auto'
        )
    elif algorithm == 'DQN':
        model = DQN(
            'MlpPolicy',
            train_env_vec,
            verbose=0,
            seed=seed,
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
            verbose=0,
            seed=seed,
            learning_rate=0.0005,
            n_steps=1024,
            gamma=0.99,
            ent_coef=0.05,
            device='auto'
        )
    
    # Train model (sizin kodunuzdaki gibi)
    model.learn(total_timesteps=total_timesteps)
    
    return model

def evaluate_single_run_academic(model, test_env):
    """
    TEK RUN EVALUATION - Sizin evaluate_model() fonksiyonunuzun tek episode versiyonu
    """
    obs = test_env.reset()
    episode_reward = 0
    step_count = 0
    action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    asset_counts = {symbol: 0 for symbol in test_env.asset_symbols}
    
    portfolio_values = [test_env.initial_cash]
    daily_returns = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Track actions (sizin kodunuzdaki gibi)
        action_names = ['HOLD', 'BUY', 'SELL']
        if len(action) > 1:
            action_counts[action_names[action[1]]] += 1
        asset_counts[test_env.asset_symbols[action[0]]] += 1
        
        # Portfolio tracking
        portfolio_values.append(info['portfolio_value'])
        
        if step_count > 1000:
            break
    
    # Calculate metrics (sizin info dict'inizdeki metrikler + akademik metrikler)
    if len(portfolio_values) > 1:
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    final_portfolio = info['portfolio_value']
    total_return = info['total_return']
    
    # Academic metrics
    volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        excess_returns = daily_returns - (0.02/252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    if len(portfolio_values) > 1:
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        annualized_return = (final_portfolio / test_env.initial_cash) ** (252/len(portfolio_values)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        max_drawdown = 0
        annualized_return = 0
        calmar_ratio = 0
    
    return {
        # Sizin mevcut metrikleriniz
        'total_return': total_return,
        'final_portfolio': final_portfolio,
        'total_trades': info['total_trades'],
        'steps': step_count,
        'diversification_score': info.get('diversification_score', 0),
        'action_counts': action_counts.copy(),
        'asset_counts': asset_counts.copy(),
        # Academic eklentiler
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'portfolio_values': portfolio_values.copy(),
        'daily_returns': daily_returns.copy() if len(daily_returns) > 0 else []
    }

def run_academic_evaluation(algorithm='PPO', total_timesteps=40000, n_runs=5, use_real_data=False):
    """
    ANA ACADEMIC EVALUATION FONKSIYONU
    Sizin main_training_pipeline() fonksiyonunuzun academic versiyonu
    """
    print(f"\n🎓 ACADEMIC STANDARD EVALUATION")
    print(f"Algorithm: {algorithm} | Timesteps: {total_timesteps:,} | Runs: {n_runs}")
    print("=" * 70)
    
    # Data preparation (sizin fonksiyonlarınızla)
    if use_real_data:
        asset_data = prepare_real_alpaca_data()
    else:
        asset_data = prepare_training_data()
    
    # Seeds for reproducibility
    seeds = [42, 123, 456, 789, 999, 333, 666, 888, 111, 222][:n_runs]
    all_results = []
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n🏃 Run {run_idx + 1}/{n_runs} (seed: {seed})")
        print("-" * 40)
        
        # Create environments (sizin fonksiyonunuzla)
        train_env, val_env = create_training_environment(asset_data)
        
        # Train model with seed
        model = train_model_with_seed(
            train_env, val_env, 
            algorithm=algorithm, 
            total_timesteps=total_timesteps, 
            seed=seed
        )
        
        # Evaluate
        run_result = evaluate_single_run_academic(model, val_env)
        run_result['seed'] = seed
        run_result['run_id'] = run_idx + 1
        
        all_results.append(run_result)
        
        print(f"  ✅ Run {run_idx + 1}: Return {run_result['total_return']:.2%} | "
              f"Sharpe {run_result['sharpe_ratio']:.2f} | "
              f"MaxDD {run_result['max_drawdown']:.2%}")
    
    # Statistical Analysis
    summary = analyze_results_academic(all_results, algorithm)
    
    # Visualization
    plot_academic_results(all_results, algorithm, total_timesteps)
    
    # Benchmark comparison
    benchmark_results = compare_with_buy_and_hold_benchmark(all_results, asset_data)
    
    return summary, all_results, benchmark_results

def analyze_results_academic(results, algorithm):
    """STATISTICAL ANALYSIS"""
    if not results:
        return {}
    
    print(f"\n📊 ACADEMIC STATISTICAL ANALYSIS ({algorithm})")
    print("=" * 60)
    
    metrics = ['total_return', 'annualized_return', 'volatility', 
               'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'total_trades']
    
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
        
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # 95% confidence interval
            if len(values) > 1:
                t_stat = stats.t.ppf(0.975, len(values)-1)
                margin_error = t_stat * (std_val / np.sqrt(len(values)))
                ci_lower = mean_val - margin_error
                ci_upper = mean_val + margin_error
            else:
                ci_lower = ci_upper = mean_val
            
            summary[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': len(values)
            }
            
            # Pretty print
            if 'return' in metric or 'drawdown' in metric:
                print(f"{metric:20}: {mean_val:7.2%} ± {std_val:6.2%} "
                      f"[{min_val:7.2%}, {max_val:7.2%}] "
                      f"95%CI: [{ci_lower:6.2%}, {ci_upper:6.2%}]")
            elif 'ratio' in metric:
                print(f"{metric:20}: {mean_val:7.2f} ± {std_val:6.2f} "
                      f"[{min_val:7.2f}, {max_val:7.2f}] "
                      f"95%CI: [{ci_lower:6.2f}, {ci_upper:6.2f}]")
            else:
                print(f"{metric:20}: {mean_val:7.1f} ± {std_val:6.1f} "
                      f"[{min_val:7.1f}, {max_val:7.1f}] "
                      f"95%CI: [{ci_lower:6.1f}, {ci_upper:6.1f}]")
    
    # Statistical significance test
    returns = [r['total_return'] for r in results]
    if len(returns) > 1:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        print(f"\n🔬 STATISTICAL SIGNIFICANCE TEST:")
        print(f"   H₀: Mean return = 0% (no profit)")
        print(f"   T-statistic: {t_stat:.3f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")
        
        summary['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Academic paper format
    print(f"\n📝 ACADEMIC PAPER FORMAT:")
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    mean_sharpe = np.mean([r['sharpe_ratio'] for r in results if not np.isnan(r['sharpe_ratio'])])
    
    print(f'   "Our {algorithm} model achieved {mean_return:.2%} ± {std_return:.2%} total return')
    print(f'    (mean ± std over {len(results)} independent runs) with Sharpe ratio {mean_sharpe:.2f}."')
    
    return summary

def plot_academic_results(results, algorithm, timesteps):
    """ACADEMIC VISUALIZATION"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Academic Evaluation: {algorithm} ({len(results)} runs)', fontsize=16)
    
    returns = [r['total_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results if not np.isnan(r['sharpe_ratio'])]
    max_drawdowns = [r['max_drawdown'] for r in results]
    
    # Return distribution
    axes[0, 0].hist(returns, bins=max(3, len(results)//2), alpha=0.7, edgecolor='black')
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    axes[0, 0].axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2%}')
    axes[0, 0].set_title('Total Return Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe distribution
    if sharpe_ratios:
        axes[0, 1].hist(sharpe_ratios, bins=max(3, len(sharpe_ratios)//2), alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Sharpe Ratio Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[0, 2].hist(max_drawdowns, bins=max(3, len(max_drawdowns)//2), alpha=0.7, edgecolor='black', color='red')
    axes[0, 2].axvline(np.mean(max_drawdowns), color='darkred', linestyle='--', linewidth=2)
    axes[0, 2].set_title('Max Drawdown Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Run-by-run
    run_ids = [r['run_id'] for r in results]
    axes[1, 0].plot(run_ids, returns, 'o-', linewidth=2, markersize=8)
    axes[1, 0].axhline(mean_return, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Return by Run')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error bar plot
    axes[1, 1].errorbar([0], [mean_return], yerr=[std_return], 
                       fmt='o', markersize=10, capsize=10, capthick=2, linewidth=3)
    axes[1, 1].set_ylabel('Return (Mean ± Std)')
    axes[1, 1].set_title('Academic Style')
    axes[1, 1].text(0, mean_return + std_return + 0.01, 
                    f'{mean_return:.2%} ± {std_return:.2%}', 
                    ha='center', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks([])
    
    # Box plot
    axes[1, 2].boxplot([returns, sharpe_ratios if sharpe_ratios else [0]], 
                       labels=['Returns', 'Sharpe'])
    axes[1, 2].set_title('Distribution Comparison')
    
    plt.tight_layout()
    plt.savefig(f'academic_results_{algorithm}.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_buy_and_hold_benchmark(results, asset_data):
    """BENCHMARK COMPARISON"""
    print(f"\n🏆 BENCHMARK COMPARISON")
    print("=" * 40)
    
    # Equal weight buy-and-hold
    symbols = list(asset_data.keys())
    bnh_returns = {}
    
    for symbol in symbols:
        data = asset_data[symbol]
        initial_price = data['Close'].iloc[60]
        final_price = data['Close'].iloc[-1]
        bnh_return = (final_price / initial_price) - 1
        bnh_returns[symbol] = bnh_return
    
    equal_weight_return = np.mean(list(bnh_returns.values()))
    
    # Model comparison
    model_returns = [r['total_return'] for r in results]
    model_mean = np.mean(model_returns)
    
    # Statistical test
    t_stat, p_value = stats.ttest_1samp(model_returns, equal_weight_return)
    
    print(f"   Model Return: {model_mean:.2%}")
    print(f"   Buy-Hold Return: {equal_weight_return:.2%}")
    print(f"   Outperformance: {model_mean - equal_weight_return:.2%}")
    print(f"   P-value: {p_value:.6f}")
    
    if p_value < 0.05 and model_mean > equal_weight_return:
        print(f"   ✅ SIGNIFICANTLY BETTER than buy-and-hold!")
    else:
        print(f"   ⚠️ No significant outperformance")
    
    return {
        'benchmark_return': equal_weight_return,
        'outperformance': model_mean - equal_weight_return,
        'significant': p_value < 0.05 and model_mean > equal_weight_return
    }

# ======================================================================
# KOLAY KULLANIM FONKSİYONLARI
# ======================================================================

def quick_academic_test():
    """Hızlı test (5 dakika)"""
    print("🧪 QUICK ACADEMIC TEST")
    summary, results, benchmark = run_academic_evaluation(
        algorithm='PPO',
        total_timesteps=5000,
        n_runs=3,
        use_real_data=False
    )
    return summary, results, benchmark

def full_academic_evaluation():
    """Tam academic evaluation (20-30 dakika)"""
    print("🎓 FULL ACADEMIC EVALUATION")
    
    algorithm = input("Algoritma (PPO/DQN/A2C) [PPO]: ").upper() or 'PPO'
    timesteps = int(input("Timesteps [40000]: ") or 40000)
    n_runs = int(input("Runs [5]: ") or 5)
    use_real = input("Real data? (y/n) [n]: ").lower() == 'y'
    
    summary, results, benchmark = run_academic_evaluation(
        algorithm=algorithm,
        total_timesteps=timesteps,
        n_runs=n_runs,
        use_real_data=use_real
    )
    
    return summary, results, benchmark



# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    print("🎓 ACADEMIC EVALUATION SYSTEM")
    print("=" * 50)
    
    choice = input("""
Seçin:
1. Quick Test (5 dakika)
2. Full Academic Evaluation 
3. Sizin Original Pipeline
Choice [1]: """).strip() or "1"
    
    if choice == "1":
        summary, results, benchmark = quick_academic_test()
        
    elif choice == "2":
        summary, results, benchmark = full_academic_evaluation()
        
    elif choice == "3":
        print("🔧 Original pipeline çalışıyor...")
        # Sizin original fonksiyonunuz
        model, asset_data, train_env, val_env = main_training_pipeline()
        print("✅ Original pipeline completed!")
        
    else:
        print("🎯 Default: Quick test...")
        summary, results, benchmark = quick_academic_test()
    
    print("\n🎉 Evaluation completed!")
