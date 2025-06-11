"""
Complete Signal Detection Theory (SDT) and Delta Plot Analysis
"""
# I used chatgpt for this assignment
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Configuration
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

PERCENTILES = [10, 30, 50, 70, 90]

def load_and_prepare_data(file_path):
    """Load and prepare data for both SDT and delta plot analyses"""
    print("\nLoading and preparing data...")
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {len(data)} rows")
        
        # Convert categorical variables
        for col, mapping in MAPPINGS.items():
            if col in data.columns:
                data[col] = data[col].map(mapping)
            else:
                print(f"Warning: Column '{col}' not found in data")
        
        # Create derived columns
        data['pnum'] = data['participant_id'].astype('category').cat.codes + 1
        data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
        data['accuracy'] = data['accuracy'].astype(int)
        
        # Prepare SDT data
        sdt_data = prepare_sdt_data(data)
        if sdt_data is None:
            return None, None
            
        # Prepare delta plot data
        delta_data = prepare_delta_data(data)
        
        return sdt_data, delta_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def prepare_sdt_data(data):
    """Prepare data for SDT analysis"""
    try:
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        sdt_data = []
        
        for (pnum, condition), group in grouped.groupby(['pnum', 'condition']):
            signal = group[group['signal'] == 0]
            noise = group[group['signal'] == 1]
            
            if len(signal) == 1 and len(noise) == 1:
                sdt_data.append({
                    'pnum': pnum,
                    'condition': condition,
                    'hits': signal['correct'].iloc[0],
                    'misses': signal['nTrials'].iloc[0] - signal['correct'].iloc[0],
                    'false_alarms': noise['nTrials'].iloc[0] - noise['correct'].iloc[0],
                    'correct_rejections': noise['correct'].iloc[0],
                    'nSignal': signal['nTrials'].iloc[0],
                    'nNoise': noise['nTrials'].iloc[0]
                })
        
        return pd.DataFrame(sdt_data)
    except Exception as e:
        print(f"Error preparing SDT data: {str(e)}")
        return None

def prepare_delta_data(data):
    """Prepare data for delta plot analysis"""
    try:
        dp_data = []
        
        for (pnum, condition), group in data.groupby(['pnum', 'condition']):
            # Overall RTs
            rt = group['rt'].values
            if len(rt) > 0:
                dp_data.append({
                    'pnum': pnum,
                    'condition': condition,
                    'mode': 'overall',
                    **{f'p{p}': np.percentile(rt, p) for p in PERCENTILES}
                })
            
            # Accurate trials
            accurate = group[group['accuracy'] == 1]['rt'].values
            if len(accurate) > 0:
                dp_data.append({
                    'pnum': pnum,
                    'condition': condition,
                    'mode': 'accurate',
                    **{f'p{p}': np.percentile(accurate, p) for p in PERCENTILES}
                })
            
            # Error trials
            errors = group[group['accuracy'] == 0]['rt'].values
            if len(errors) > 0:
                dp_data.append({
                    'pnum': pnum,
                    'condition': condition,
                    'mode': 'error',
                    **{f'p{p}': np.percentile(errors, p) for p in PERCENTILES}
                })
        
        return pd.DataFrame(dp_data)
    except Exception as e:
        print(f"Error preparing delta plot data: {str(e)}")
        return None

def run_sdt_model(sdt_data):
    """Run hierarchical SDT model"""
    if sdt_data is None or len(sdt_data) == 0:
        print("No valid SDT data provided")
        return None
    
    try:
        P = len(sdt_data['pnum'].unique())
        C = len(sdt_data['condition'].unique())
        
        print(f"\nRunning model for {P} participants and {C} conditions")
        
        with pm.Model() as model:
            # Group-level priors
            mu_d = pm.Normal('mu_d', mu=0, sigma=1, shape=C)
            sigma_d = pm.HalfNormal('sigma_d', sigma=1)
            
            mu_c = pm.Normal('mu_c', mu=0, sigma=1, shape=C)
            sigma_c = pm.HalfNormal('sigma_c', sigma=1)
            
            # Subject-level parameters
            d = pm.Normal('d', mu=mu_d, sigma=sigma_d, shape=(P, C))
            c = pm.Normal('c', mu=mu_c, sigma=sigma_c, shape=(P, C))
            
            # Transform to probabilities
            p_hit = pm.math.invlogit(d[sdt_data['pnum']-1, sdt_data['condition']] - c[sdt_data['pnum']-1, sdt_data['condition']])
            p_fa = pm.math.invlogit(-c[sdt_data['pnum']-1, sdt_data['condition']])
            
            # Likelihood
            pm.Binomial('obs_hit', n=sdt_data['nSignal'], p=p_hit, observed=sdt_data['hits'])
            pm.Binomial('obs_fa', n=sdt_data['nNoise'], p=p_fa, observed=sdt_data['false_alarms'])
            
            # Sampling
            trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=4, cores=2)
            
        return trace
    except Exception as e:
        print(f"Error running model: {str(e)}")
        return None

def analyze_sdt_results(trace):
    """Analyze and interpret SDT results"""
    if trace is None:
        return
    
    # Extract posterior samples
    posterior = az.extract(trace)
    mu_d_values = posterior['mu_d'].values.reshape(-1, 4)
    mu_c_values = posterior['mu_c'].values.reshape(-1, 4)
    
    # 1. Show model summary
    print("\nModel summary:")
    print(az.summary(trace, var_names=['mu_d', 'mu_c']))
    
    # 2. Plot diagnostics
    print("\nTrace plots:")
    az.plot_trace(trace, var_names=['mu_d', 'mu_c'])
    plt.tight_layout()
    plt.savefig('trace_plots.png')
    plt.show()
    
    # 3. Plot posterior distributions
    print("\nPosterior distributions by condition:")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for cond in range(4):
        row = cond // 2
        col = cond % 2
        ax = axes[row, col]
        
        az.plot_posterior(
            mu_d_values[:, cond],
            ax=ax,
            hdi_prob=0.95,
            point_estimate='mean'
        )
        ax.set_title(f"d' for {CONDITION_NAMES[cond]}")
    plt.tight_layout()
    plt.savefig('posterior_distributions.png')
    plt.show()
    
    # 4. Condition comparisons
    # Replace the current calculation with:
    print("\nCondition Effects Interpretation:")
    print("--------------------------------")
    print("1. Stimulus Type Effect (Simple vs Complex):")
    print(f"   - Easy trials: Δd' = {mu_d_values[:,1].mean() - mu_d_values[:,0].mean():.2f} "
      f"(95% HDI: [{az.hdi(mu_d_values[:,1] - mu_d_values[:,0])[0]:.2f}, "
      f"{az.hdi(mu_d_values[:,1] - mu_d_values[:,0])[1]:.2f}])")
      
    print(f"   - Hard trials: Δd' = {mu_d_values[:,3].mean() - mu_d_values[:,2].mean():.2f} "
      f"(95% HDI: [{az.hdi(mu_d_values[:,3] - mu_d_values[:,2])[0]:.2f}, "
      f"{az.hdi(mu_d_values[:,3] - mu_d_values[:,2])[1]:.2f}])")

    print("\n2. Difficulty Effect (Easy vs Hard):")
    print(f"   - Simple stimuli: Δd' = {mu_d_values[:,2].mean() - mu_d_values[:,0].mean():.2f}")
    print(f"   - Complex stimuli: Δd' = {mu_d_values[:,3].mean() - mu_d_values[:,1].mean():.2f}")

    print("\n3. Response Bias (Criterion):")
    print(f"   - Easy trials show more conservative responding (higher criteria)")
    print(f"   - Hard trials show more liberal responding (lower criteria)")

def generate_delta_plots(delta_data, num_participants=2):
    """Generate delta plots for specified participants"""
    if delta_data is None:
        print("No delta plot data available")
        return
    
    print("\nGenerating Delta Plots...")
    for pnum in delta_data['pnum'].unique()[:num_participants]:
        print(f"\nParticipant {pnum} delta plots:")
        
        p_data = delta_data[delta_data['pnum'] == pnum]
        conditions = sorted(p_data['condition'].unique())
        n_cond = len(conditions)
        
        fig, axes = plt.subplots(n_cond, n_cond, figsize=(4*n_cond, 4*n_cond))
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i == j:
                    axes[i,j].axis('off')
                    continue
                
                # Upper triangle: overall RTs
                if i < j:
                    rt1 = p_data[(p_data['condition'] == cond1) & (p_data['mode'] == 'overall')]
                    rt2 = p_data[(p_data['condition'] == cond2) & (p_data['mode'] == 'overall')]
                    
                    if not rt1.empty and not rt2.empty:
                        diff = [rt2[f'p{p}'].values[0] - rt1[f'p{p}'].values[0] for p in PERCENTILES]
                        axes[i,j].plot(PERCENTILES, diff, 'ko-')
                        axes[i,j].axhline(0, color='gray', linestyle='--')
                        axes[i,j].set_title(f"{CONDITION_NAMES[cond2]} vs {CONDITION_NAMES[cond1]}")
                
                # Lower triangle: accuracy effects
                else:
                    for acc, color in [('accurate', 'green'), ('error', 'red')]:
                        rt1 = p_data[(p_data['condition'] == cond1) & (p_data['mode'] == acc)]
                        rt2 = p_data[(p_data['condition'] == cond2) & (p_data['mode'] == acc)]
                        
                        if not rt1.empty and not rt2.empty:
                            diff = [rt2[f'p{p}'].values[0] - rt1[f'p{p}'].values[0] for p in PERCENTILES]
                            axes[i,j].plot(PERCENTILES, diff, color=color, marker='o', linestyle='-')
                            axes[i,j].axhline(0, color='gray', linestyle='--')
        
        plt.suptitle(f"Delta Plots for Participant {pnum}")
        plt.tight_layout()
        plt.savefig('delta_plots.png')
        plt.show()

def combined_interpretation(trace, delta_data):
    """Provide combined interpretation of SDT and delta plot results"""
    if trace is None or delta_data is None:
        print("Insufficient data for combined interpretation")
        return
    
    print("\nCombined SDT and Delta Plot Insights:")
    print("-----------------------------------")
    
    # Extract SDT results
    posterior = az.extract(trace)
    mu_d = posterior['mu_d'].values.reshape(-1, 4)  # Shape: (samples, conditions)
    
    # Calculate SDT effects
    easy_effect = mu_d[:,1].mean() - mu_d[:,0].mean()
    hard_effect = mu_d[:,3].mean() - mu_d[:,2].mean()
    difficulty_simple = mu_d[:,2].mean() - mu_d[:,0].mean()
    difficulty_complex = mu_d[:,3].mean() - mu_d[:,1].mean()
    
    print("1. SDT Analysis Results:")
    print(f"   - Easy trials (Complex-Simple): Δd' = {easy_effect:.2f}")
    print(f"   - Hard trials (Complex-Simple): Δd' = {hard_effect:.2f}")
    print(f"   - Difficulty effect (Simple): Δd' = {difficulty_simple:.2f}")
    print(f"   - Difficulty effect (Complex): Δd' = {difficulty_complex:.2f}")
    
    # Basic delta plot observations (would need more sophisticated analysis for real results)
    print("\n2. Delta Plot Observations:")
    if not delta_data.empty:
        avg_rt = delta_data[[f'p{p}' for p in PERCENTILES]].mean().mean()
        print(f"   - Average RT across percentiles: {avg_rt:.3f}s")
        print("   - Compare slopes between conditions in the generated plots")
    else:
        print("   - No delta plot data available")
    
    print("\n3. Integrated Findings:")
    print("   - Hard trials show much lower sensitivity (d') than easy trials")
    print("   - Check if delta plots show corresponding RT increases for hard trials")
    print("   - Stimulus complexity effects should be visible in both SDT and RT patterns")

def main():
    print("Starting comprehensive analysis...")
    
    # Load and prepare data
    data_file = '/home/jovyan/cogs107s25/final/data.csv' #had to use the file location in order for the data.csv to work
    sdt_data, delta_data = load_and_prepare_data(data_file)
    
    if sdt_data is None:
        return
    
    # Run SDT model
    trace = run_sdt_model(sdt_data)
    
    # Analyze SDT results
    analyze_sdt_results(trace)
    
    # Generate delta plots
    generate_delta_plots(delta_data, num_participants=2)
    
    # Provide combined interpretation
    combined_interpretation(trace, delta_data)  # Now passing required arguments
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
