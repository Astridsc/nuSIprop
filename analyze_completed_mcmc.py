#!/usr/bin/env python3
"""
Analyze completed MCMC results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import emcee
import os
import glob

class MockSampler:
    """
    Mock sampler class to work with CSV data
    """
    def __init__(self, chain, log_prob=None, acceptance_fraction=None):
        self._chain = chain
        self._log_prob = log_prob
        self._acceptance_fraction = acceptance_fraction
    
    def get_chain(self, discard=0, flat=False):
        """Get the chain data"""
        if flat:
            # Return flattened chain
            n_steps, n_walkers, n_params = self._chain.shape
            return self._chain[discard:].reshape(-1, n_params)
        else:
            # Return chain with burn-in discarded
            return self._chain[discard:]
    
    def get_log_prob(self, discard=0, flat=False):
        """Get the log probability data"""
        if self._log_prob is None:
            raise AttributeError("No log probability data available")
        
        if flat:
            # Return flattened log probability
            n_steps, n_walkers = self._log_prob.shape
            return self._log_prob[discard:].flatten()
        else:
            # Return log probability with burn-in discarded
            return self._log_prob[discard:]
    
    @property
    def acceptance_fraction(self):
        """Get acceptance fractions"""
        if self._acceptance_fraction is None:
            raise AttributeError("No acceptance fraction data available")
        return self._acceptance_fraction



def analyze_completed_mcmc(sampler, save_results=True, folder_name=None):
    """
    Comprehensive analysis of completed MCMC results
    """
    print("=== COMPLETED MCMC ANALYSIS ===")
    
    # Get all available data
    chain = sampler.get_chain()
    n_steps, n_chains, n_params = chain.shape
    
    print(f"Final results:")
    print(f"  Total iterations completed: {n_steps}")
    print(f"  Number of walkers: {n_chains}")
    print(f"  Number of parameters: {n_params}")
    
    # Get likelihood information
    if hasattr(sampler, 'get_log_prob'):
        log_prob = sampler.get_log_prob()
        print(f"  Likelihood range: {np.min(log_prob):.2f} to {np.max(log_prob):.2f}")
        
        # Find best fit
        best_idx = np.unravel_index(np.argmax(log_prob), log_prob.shape)
        best_iteration, best_walker = best_idx
        best_params = chain[best_iteration, best_walker, :]
        best_likelihood = log_prob[best_iteration, best_walker]
        
        print(f"  Best fit found at iteration {best_iteration} by walker {best_walker}")
        print(f"  Best likelihood: {best_likelihood:.2f}")
        print(f"  Best parameters: {best_params}")
    
    # Acceptance rates
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rates = sampler.acceptance_fraction
        mean_acc = np.mean(acc_rates)
        std_acc = np.std(acc_rates)
        
        print(f"\nAcceptance rates:")
        print(f"  Mean: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  Range: {np.min(acc_rates):.3f} to {np.max(acc_rates):.3f}")
        
        # Individual walker acceptance rates
        print(f"  Individual walker rates:")
        for i, acc in enumerate(acc_rates):
            print(f"    Walker {i}: {acc:.3f}")
    
    # Parameter statistics
    #param_names = ["M_phi", "g", "si", "norm"]
    param_names = ["M_phi", "g", "si"]
    
    print(f"\nParameter statistics (all iterations):")
    for i, name in enumerate(param_names):
        param_data = chain[:, :, i].flatten()
        print(f"  {name}:")
        print(f"    Mean: {np.mean(param_data):.2e}")
        print(f"    Std:  {np.std(param_data):.2e}")
        print(f"    Min:  {np.min(param_data):.2e}")
        print(f"    Max:  {np.max(param_data):.2e}")
        print(f"    Median: {np.median(param_data):.2e}")
    
    # Convergence diagnostics
    print(f"\nConvergence diagnostics:")
    
    # Gelman-Rubin statistic (if multiple chains)
    # OBS! Should not be used for emcee, since the chains are not independent. See https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """if n_chains > 1:
        try:
            R_hat = gelman_rubin(sampler)
            print(f"  Gelman-Rubin R-hat: {R_hat}")
            print(f"  Converged (R-hat < 1.1): {np.all(R_hat < 1.1)}")
        except:
            print("  Gelman-Rubin calculation failed")"""
    
    # Autocorrelation
    try:
        tau = sampler.get_autocorr_time()
        n_eff = len(chain.flatten()) / np.max(tau)
        print(f"  Autocorrelation times: {tau}")
        print(f"  Effective sample size: {n_eff:.0f}")
    except:
        print("  Autocorrelation calculation failed")
    
    # Save results
    if save_results:
        save_mcmc_results(sampler, param_names, folder_name)
    
    return {
        'chain': chain,
        'log_prob': log_prob if hasattr(sampler, 'get_log_prob') else None,
        'acceptance_rates': acc_rates if hasattr(sampler, 'acceptance_fraction') else None,
        'best_params': best_params if 'best_params' in locals() else None,
        'best_likelihood': best_likelihood if 'best_likelihood' in locals() else None
    }
    
    

def gelman_rubin(sampler, discard=0):
    """Calculate Gelman-Rubin statistic"""
    chains = sampler.get_chain(discard=discard)
    n_steps, n_chains, n_params = chains.shape
    
    # Within-chain variance
    W = np.var(chains, axis=0, ddof=1).mean(axis=1)
    
    # Between-chain variance
    chain_means = np.mean(chains, axis=0)
    B = n_steps * np.var(chain_means, axis=0, ddof=1)
    
    # Pooled variance
    V = (n_steps - 1) / n_steps * W + (n_chains + 1) / (n_chains * n_steps) * B
    
    # R-hat statistic
    R_hat = np.sqrt(V / W)
    
    return R_hat



def save_mcmc_results(sampler, param_names, folder_name=None):
    """Save MCMC results to files"""
    print("\nSaving results to files...")
    
    # Save chain
    chain = sampler.get_chain()
    np.save(f'{folder_name}/mcmc_chain_final.npy', chain)
    print(f"  Chain saved to '{folder_name}/mcmc_chain_final.npy'")
    
    # Save likelihood
    if hasattr(sampler, 'get_log_prob'):
        log_prob = sampler.get_log_prob()
        np.save(f'{folder_name}/mcmc_log_prob_final.npy', log_prob)
        print(f"  Log probability saved to '{folder_name}/mcmc_log_prob_final.npy'")
    
    # Save acceptance rates
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rates = sampler.acceptance_fraction
        np.save(f'{folder_name}/mcmc_acceptance_rates.npy', acc_rates)
        print(f"  Acceptance rates saved to '{folder_name}/mcmc_acceptance_rates.npy'")
    
    # Save parameter statistics as CSV
    chain_flat = sampler.get_chain(flat=True)
    param_stats = pd.DataFrame(chain_flat, columns=param_names)
    param_stats.to_csv(f'{folder_name}/mcmc_parameter_samples.csv', index=False)
    print(f"  Parameter samples saved to '{folder_name}/mcmc_parameter_samples.csv'")
    
    # Save summary statistics
    summary_stats = param_stats.describe()
    summary_stats.to_csv(f'{folder_name}/mcmc_summary_statistics.csv')
    print(f"  Summary statistics saved to '{folder_name}/mcmc_summary_statistics.csv'")



def plot_comprehensive_results(sampler, param_names=["M_phi", "g", "si", "norm"], discard=100, title=None, folder_name=None, save_plots=True):
    """Create comprehensive plots of MCMC results"""
    print("\nCreating comprehensive plots...")
    
    chain = sampler.get_chain()
    n_steps, n_chains, n_params = chain.shape
    
    # 1. Chain evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        
        # Plot all chains
        for j in range(min(n_chains, 10)):
            ax.plot(chain[:, j, i], alpha=0.3, linewidth=0.5)
            
        # Plot mean across all walkers at each iteration
        mean_chain = np.mean(chain[:, :, i], axis=1)
        ax.plot(mean_chain, 'k-', linewidth=2, label='Mean across walkers')
        
        # Plot standard deviation envelope
        std_chain = np.std(chain[:, :, i], axis=1)
        ax.fill_between(np.arange(n_steps), mean_chain - std_chain, mean_chain + std_chain, 
                        alpha=0.2, color='gray', label='±1σ')
        
        ax.set_title(f'{param_names[i]} Chain Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for parameters that span many orders of magnitude
        if i in [0, 1, 3]:  # M_phi, g, norm
            ax.set_yscale('log')
        if i == 3:
            ax.set_ylim(1e-25, 1e-17)
        
        # Add some statistics to the plot
        final_mean = np.mean(chain[-1, :, i])
        final_std = np.std(chain[-1, :, i])
        ax.text(0.02, 0.98, f'Final: {final_mean:.2e} ± {final_std:.2e}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{folder_name}/mcmc_chain_evolution.png', dpi=150, bbox_inches='tight')
        print(f"  Chain evolution plot saved to '{folder_name}/mcmc_chain_evolution.png'")
    plt.show()

    
    
    # 2. Corner plot
    samples = sampler.get_chain(discard=discard, flat=True)  # Discard first 100 as burn-in
    if title is not None:
        title = title
    else:
        title = None
    fig = corner.corner(samples, labels=param_names, 
                       quantiles=[0.68, 0.95, 0.997],
                       show_titles=True, title_kwargs={"fontsize": 12}, 
                       axes_scale=['log', 'log', 'linear'], title_fmt='.2e')
    
    if save_plots:
        plt.savefig(f'{folder_name}/mcmc_corner_plot.png', dpi=150, bbox_inches='tight')
        print(f"  Corner plot saved to '{folder_name}/mcmc_corner_plot.png'")
    plt.show()
    
    # 3. Likelihood evolution
    if hasattr(sampler, 'get_log_prob'):
        log_prob = sampler.get_log_prob()
        print('log prob shape', log_prob.shape)
        
        # Check for infinite values
        n_infinite = np.sum(~np.isfinite(log_prob))
        n_total = log_prob.size
        print(f'Infinite log probabilities: {n_infinite}/{n_total} ({100*n_infinite/n_total:.1f}%)')
        
        if n_infinite > 0:
            print('Some walkers are exploring regions with zero likelihood')
            print('This is normal in MCMC - filtering out infinite values for mean calculation')
        
        plt.figure(figsize=(12, 6))
        
        # Plot individual walkers
        for i in range(min(n_chains, 10)):
            plt.plot(log_prob[:, i], alpha=0.3, linewidth=0.5)
        
        # Plot mean (filter out infinite values)
        finite_log_prob = np.ma.masked_invalid(log_prob)
        mean_log_prob = np.ma.mean(finite_log_prob, axis=1)
        print('mean log prob', mean_log_prob)
        plt.plot(mean_log_prob, 'k-', linewidth=1.5, label='Mean (finite only)')
        
        # Also plot median as an alternative robust statistic
        median_log_prob = np.ma.median(finite_log_prob, axis=1)
        plt.plot(median_log_prob, color = 'gray', linestyle='--', linewidth=1, label='Median (finite only)')
        
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Likelihood Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(f'{folder_name}/mcmc_likelihood_evolution.png', dpi=150, bbox_inches='tight')
            print(f"  Likelihood evolution plot saved to '{folder_name}/mcmc_likelihood_evolution.png'")
        plt.show()
    
    # 4. Acceptance rate distribution
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rates = sampler.acceptance_fraction
        
        plt.figure(figsize=(10, 6))
        plt.hist(acc_rates, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(acc_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(acc_rates):.3f}')
        plt.xlabel('Acceptance Rate')
        plt.ylabel('Number of Walkers')
        plt.title('Acceptance Rate Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(f'{folder_name}/mcmc_acceptance_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  Acceptance rate distribution saved to '{folder_name}/mcmc_acceptance_distribution.png'")
        plt.show()



def plot_walker_acceptance_evolution(sampler, save_plot=True):
    """
    Plot acceptance rate evolution for each walker
    """
    print("Creating acceptance rate evolution plot...")
    
    if not hasattr(sampler, 'acceptance_fraction'):
        print("No acceptance fraction data available")
        return
    
    acc_rates = sampler.acceptance_fraction
    n_walkers = len(acc_rates)
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual walker acceptance rates
    walker_indices = np.arange(n_walkers)
    plt.bar(walker_indices, acc_rates, alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_acc = np.mean(acc_rates)
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f}')
    
    # Add target acceptance rate range
    plt.axhspan(0.2, 0.5, alpha=0.2, color='green', label='Target range (0.2-0.5)')
    
    plt.xlabel('Walker Index')
    plt.ylabel('Acceptance Rate')
    plt.title('Acceptance Rates by Walker')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, f'Mean: {mean_acc:.3f}\nStd: {np.std(acc_rates):.3f}\nMin: {np.min(acc_rates):.3f}\nMax: {np.max(acc_rates):.3f}', 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_plot:
        plt.savefig('walker_acceptance_rates.png', dpi=150, bbox_inches='tight')
        print("Walker acceptance rates plot saved to 'walker_acceptance_rates.png'")
    
    plt.show()
    
    

def plot_parameter_statistics(sampler, param_names=["M_phi", "g", "si", "norm"], save_plot=True):
    """
    Plot detailed parameter statistics
    """
    print("Creating parameter statistics plots...")
    
    chain = sampler.get_chain()
    n_chains, n_steps, n_params = chain.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        
        # Get parameter data for all walkers and iterations
        param_data = chain[:, :, i].flatten()
        
        # Create histogram
        ax.hist(param_data, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Add vertical lines for statistics
        mean_val = np.mean(param_data)
        median_val = np.median(param_data)
        std_val = np.std(param_data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2e}')
        ax.axvline(mean_val + std_val, color='gray', linestyle=':', linewidth=1, label=f'+1σ: {mean_val + std_val:.2e}')
        ax.axvline(mean_val - std_val, color='gray', linestyle=':', linewidth=1, label=f'-1σ: {mean_val - std_val:.2e}')
        
        ax.set_title(f'{param_names[i]} Distribution')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for appropriate parameters
        if i in [0, 1, 3]:  # M_phi, g, norm
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('parameter_statistics.png', dpi=150, bbox_inches='tight')
        print("Parameter statistics plot saved to 'parameter_statistics.png'")
    
    plt.show()



def print_detailed_chain_info(sampler):
    """
    Print detailed information about the chain
    """
    print("=== DETAILED CHAIN INFORMATION ===")
    
    chain = sampler.get_chain()
    n_chains, n_steps, n_params = chain.shape
    
    print(f"Chain dimensions: {chain.shape}")
    print(f"Number of walkers: {n_chains}")
    print(f"Number of iterations: {n_steps}")
    print(f"Number of parameters: {n_params}")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(chain)):
        print("⚠️  WARNING: Found NaN or infinite values in chain!")
        nan_positions = np.where(~np.isfinite(chain))
        print(f"NaN positions: {list(zip(nan_positions[0], nan_positions[1], nan_positions[2]))}")
    else:
        print("✅ All chain values are finite")
    
    # Parameter ranges
    param_names = ["M_phi", "g", "si", "norm"]
    print(f"\nParameter ranges across all iterations:")
    for i, name in enumerate(param_names):
        param_data = chain[:, :, i]
        print(f"  {name}:")
        print(f"    Min: {np.min(param_data):.2e}")
        print(f"    Max: {np.max(param_data):.2e}")
        print(f"    Mean: {np.mean(param_data):.2e}")
        print(f"    Std: {np.std(param_data):.2e}")
    
    # Check if parameters are changing
    print(f"\nParameter evolution check (first vs last iteration):")
    for i, name in enumerate(param_names):
        first_vals = chain[:, 0, i]
        last_vals = chain[:, -1, i]
        change = np.abs(last_vals - first_vals)
        mean_change = np.mean(change)
        print(f"  {name}: Mean change = {mean_change:.2e}")
        
        if mean_change < 1e-6:
            print(f"    ⚠️  {name} appears stuck!")
        elif mean_change < 1e-4:
            print(f"    ⚠️  {name} may be getting stuck")
        else:
            print(f"    ✅ {name} is evolving")
            


def check_for_backend_files():
    """Check if there are any backend files with additional data"""
    print("\nChecking for backend files...")
    
    backend_files = glob.glob("*.h5") + glob.glob("*.hdf5")
    
    if backend_files:
        print(f"Found backend files: {backend_files}")
        for file in backend_files:
            try:
                backend = emcee.backends.HDFBackend(file)
                print(f"  {file}: {backend.iteration} iterations, {backend.shape}")
            except:
                print(f"  {file}: Could not read")
    else:
        print("No backend files found")
    
    # Check for other result files
    result_files = glob.glob("mcmc_*.npy") + glob.glob("mcmc_*.csv")
    if result_files:
        print(f"Found result files: {result_files}")



def print_best_fit_summary(results):
    """Print a summary of the best fit"""
    if results['best_params'] is not None:
        print("\n=== BEST FIT SUMMARY ===")
        param_names = ["M_phi", "g", "si", "norm"]
        
        print("Best fit parameters:")
        for i, name in enumerate(param_names):
            print(f"  {name}: {results['best_params'][i]:.2e}")
        
        print(f"Best log-likelihood: {results['best_likelihood']:.2f}")
        
        # Calculate credible intervals
        chain_flat = sampler.get_chain(flat=True)
        percentiles = [16, 50, 84]  # 1-sigma intervals
        
        print("\nParameter credible intervals (16%, 50%, 84%):")
        for i, name in enumerate(param_names):
            param_data = chain_flat[:, i]
            quantiles = np.percentile(param_data, percentiles)
            print(f"  {name}: {quantiles[0]:.2e}, {quantiles[1]:.2e}, {quantiles[2]:.2e}")



def load_sampler_from_csv(csv_file="mcmc_parameter_samples.csv", n_walkers=32):
    """
    Load MCMC data from CSV file and create a mock sampler object
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing parameter samples
    n_walkers : int
        Number of walkers used in the original MCMC
    
    Returns:
    --------
    sampler : MockSampler
        Mock sampler object compatible with analysis functions
    """
    print(f"Loading MCMC data from {csv_file}...")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} not found")
    
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Check if the CSV has the expected format
    expected_columns = ["M_phi", "g", "si", "norm"]
    if not all(col in df.columns for col in expected_columns):
        print(f"Warning: Expected columns {expected_columns} not found in CSV")
        print(f"Available columns: {list(df.columns)}")
    
    # Extract parameter data
    param_data = df[expected_columns].values
    
    # Calculate number of steps
    n_total_samples = len(param_data)
    n_steps = n_total_samples // n_walkers
    
    if n_total_samples % n_walkers != 0:
        print(f"Warning: Total samples ({n_total_samples}) not divisible by n_walkers ({n_walkers})")
        print(f"Truncating to {n_steps * n_walkers} samples")
        param_data = param_data[:n_steps * n_walkers]
    
    # Reshape into chain format (n_steps, n_walkers, n_params)
    chain = param_data.reshape(n_steps, n_walkers, len(expected_columns))
    
    print(f"Loaded chain with shape: {chain.shape}")
    print(f"Number of steps: {n_steps}")
    print(f"Number of walkers: {n_walkers}")
    print(f"Number of parameters: {len(expected_columns)}")
    
    # Create mock sampler
    sampler = MockSampler(chain)
    
    return sampler



def load_sampler_with_log_prob(csv_file="mcmc_parameter_samples.csv", 
                              log_prob_file="mcmc_log_prob.csv", 
                              n_walkers=32):
    """
    Load MCMC data from CSV files including log probability data
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing parameter samples
    log_prob_file : str
        Path to the CSV file or .npy file containing log probability data
    n_walkers : int
        Number of walkers used in the original MCMC
    
    Returns:
    --------
    sampler : MockSampler
        Mock sampler object with log probability data
    """
    # Load parameter data
    sampler = load_sampler_from_csv(csv_file, n_walkers)
    
    # Load log probability data if available
    if os.path.exists(log_prob_file):
        print(f"Loading log probability data from {log_prob_file}...")
        
        # Check if it's a .npy file
        if log_prob_file.endswith('.npy'):
            log_prob = np.load(log_prob_file)
            print(f"Loaded log probability from .npy with shape: {log_prob.shape}")
            
            # Check if the shape matches our expectations
            if len(log_prob.shape) == 2:
                # Should be (n_steps, n_walkers)
                n_steps, n_walkers_loaded = log_prob.shape
                if n_walkers_loaded != n_walkers:
                    print(f"Warning: Expected {n_walkers} walkers, but log_prob has {n_walkers_loaded}")
                    print("Reshaping log_prob to match expected walker count...")
                    # Try to reshape if possible
                    total_samples = log_prob.size
                    n_steps = total_samples // n_walkers
                    if total_samples % n_walkers == 0:
                        log_prob = log_prob.flatten()[:n_steps * n_walkers].reshape(n_steps, n_walkers)
                    else:
                        print("Cannot reshape log_prob to match walker count")
                        return sampler
                
                sampler._log_prob = log_prob
                print(f"Final log probability shape: {log_prob.shape}")
                
            elif len(log_prob.shape) == 1:
                # Flattened array, need to reshape
                total_samples = len(log_prob)
                n_steps = total_samples // n_walkers
                
                if total_samples % n_walkers == 0:
                    log_prob = log_prob.reshape(n_steps, n_walkers)
                    sampler._log_prob = log_prob
                    print(f"Reshaped log probability to: {log_prob.shape}")
                else:
                    print(f"Warning: Log probability data length ({total_samples}) not compatible with n_walkers ({n_walkers})")
                    print("Truncating to fit...")
                    log_prob = log_prob[:n_steps * n_walkers].reshape(n_steps, n_walkers)
                    sampler._log_prob = log_prob
                    print(f"Truncated and reshaped log probability to: {log_prob.shape}")
            else:
                print(f"Warning: Unexpected log_prob shape: {log_prob.shape}")
                
        else:
            # Handle CSV file (existing code)
            log_prob_df = pd.read_csv(log_prob_file)
            
            if 'log_prob' in log_prob_df.columns:
                log_prob_data = log_prob_df['log_prob'].values
                n_total = len(log_prob_data)
                n_steps = n_total // n_walkers
                
                if n_total % n_walkers == 0:
                    log_prob = log_prob_data.reshape(n_steps, n_walkers)
                    sampler._log_prob = log_prob
                    print(f"Loaded log probability with shape: {log_prob.shape}")
                else:
                    print(f"Warning: Log probability data length ({n_total}) not compatible with n_walkers ({n_walkers})")
            else:
                print("Warning: 'log_prob' column not found in log probability CSV")
    else:
        print(f"Log probability file {log_prob_file} not found")
    
    return sampler



def load_sampler_from_npy(chain_file="mcmc_chain.npy", 
                         log_prob_file="mcmc_log_prob_final.npy", 
                         n_walkers=32):
    """
    Load MCMC data from .npy files
    
    Parameters:
    -----------
    chain_file : str
        Path to the .npy file containing parameter chain data
    log_prob_file : str
        Path to the .npy file containing log probability data
    n_walkers : int
        Number of walkers used in the original MCMC (if needed for reshaping)
    
    Returns:
    --------
    sampler : MockSampler
        Mock sampler object with chain and log probability data
    """
    print("Loading MCMC data from .npy files...")
    
    # Load chain data
    if os.path.exists(chain_file):
        chain = np.load(chain_file)
        print(f"Loaded chain from {chain_file} with shape: {chain.shape}")
        
        # Check if chain needs reshaping
        if len(chain.shape) == 3:
            # Already in correct format (n_steps, n_walkers, n_params)
            pass
        elif len(chain.shape) == 2:
            # (n_samples, n_params) - need to reshape
            n_samples, n_params = chain.shape
            n_steps = n_samples // n_walkers
            if n_samples % n_walkers == 0:
                chain = chain.reshape(n_steps, n_walkers, n_params)
                print(f"Reshaped chain to: {chain.shape}")
            else:
                print(f"Warning: Chain samples ({n_samples}) not divisible by n_walkers ({n_walkers})")
                chain = chain[:n_steps * n_walkers].reshape(n_steps, n_walkers, n_params)
                print(f"Truncated and reshaped chain to: {chain.shape}")
        else:
            print(f"Warning: Unexpected chain shape: {chain.shape}")
    else:
        print(f"Chain file {chain_file} not found")
        return None
    
    # Load log probability data
    log_prob = None
    if os.path.exists(log_prob_file):
        log_prob = np.load(log_prob_file)
        print(f"Loaded log probability from {log_prob_file} with shape: {log_prob.shape}")
        
        # Check if log_prob needs reshaping
        if len(log_prob.shape) == 2:
            # Should be (n_steps, n_walkers)
            n_steps, n_walkers_loaded = log_prob.shape
            if n_walkers_loaded != n_walkers:
                print(f"Warning: Expected {n_walkers} walkers, but log_prob has {n_walkers_loaded}")
                # Try to reshape if possible
                total_samples = log_prob.size
                n_steps = total_samples // n_walkers
                if total_samples % n_walkers == 0:
                    log_prob = log_prob.flatten()[:n_steps * n_walkers].reshape(n_steps, n_walkers)
                else:
                    print("Cannot reshape log_prob to match walker count")
                    log_prob = None
        elif len(log_prob.shape) == 1:
            # Flattened array, need to reshape
            total_samples = len(log_prob)
            n_steps = total_samples // n_walkers
            if total_samples % n_walkers == 0:
                log_prob = log_prob.reshape(n_steps, n_walkers)
            else:
                print(f"Warning: Log probability data length ({total_samples}) not compatible with n_walkers ({n_walkers})")
                log_prob = log_prob[:n_steps * n_walkers].reshape(n_steps, n_walkers)
        
        if log_prob is not None:
            print(f"Final log probability shape: {log_prob.shape}")
    else:
        print(f"Log probability file {log_prob_file} not found")
    
    # Create mock sampler
    sampler = MockSampler(chain, log_prob)
    
    return sampler




# Usage instructions
print("To analyze your completed MCMC:")
print("1. Run: results = analyze_completed_mcmc(sampler)")
print("2. Run: plot_comprehensive_results(sampler)")
print("3. Run: check_for_backend_files()")
print("4. Run: print_best_fit_summary(results)")
print()
print("To load MCMC data from CSV files:")
print("1. Load from parameter samples: sampler = load_sampler_from_csv('mcmc_parameter_samples.csv')")
print("2. Load with log probability: sampler = load_sampler_with_log_prob('mcmc_parameter_samples.csv', 'mcmc_log_prob.csv')")
print("3. Then use any analysis function: results = analyze_completed_mcmc(sampler)")
print()
print("To load MCMC data from .npy files:")
print("1. Load from .npy files: sampler = load_sampler_from_npy('mcmc_chain.npy', 'mcmc_log_prob_final.npy')")
print("2. Load with just log probability .npy: sampler = load_sampler_with_log_prob('mcmc_parameter_samples.csv', 'mcmc_log_prob_final.npy')") 


