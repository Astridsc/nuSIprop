#!/usr/bin/env python3
"""
Script to check if MCMC sampling is stuck and provide diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import os

def check_mcmc_status(sampler, iteration_threshold=50):
    """
    Check if MCMC sampling appears to be stuck
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object
    iteration_threshold : int
        Number of iterations to check for stuck behavior
    
    Returns:
    --------
    dict : Status information
    """
    
    # Get current chain state
    chain = sampler.get_chain()
    n_chains, n_steps, n_params = chain.shape
    
    print(f"=== MCMC STATUS CHECK ===")
    print(f"Total iterations completed: {n_steps}")
    print(f"Number of walkers: {n_chains}")
    print(f"Number of parameters: {n_params}")
    
    # Check if sampling is still running
    if hasattr(sampler, 'running') and sampler.running:
        print("Status: SAMPLING IS STILL RUNNING")
    else:
        print("Status: SAMPLING COMPLETED")
    
    # Check for stuck behavior in recent iterations
    if n_steps >= iteration_threshold:
        recent_chain = chain[:, -iteration_threshold:, :]
        
        # Calculate parameter changes in recent iterations
        param_changes = np.abs(recent_chain[:, -1, :] - recent_chain[:, 0, :])
        mean_changes = np.mean(param_changes, axis=0)
        
        print(f"\nParameter changes in last {iteration_threshold} iterations:")
        param_names = ["M_phi", "g", "si", "norm"]
        for i, (name, change) in enumerate(zip(param_names, mean_changes)):
            print(f"  {name}: {change:.2e}")
        
        # Check if parameters are barely changing
        stuck_threshold = 1e-6  # Adjust based on your parameter scales
        stuck_params = mean_changes < stuck_threshold
        if np.any(stuck_params):
            print(f"\n⚠️  WARNING: Parameters appear stuck:")
            for i, stuck in enumerate(stuck_params):
                if stuck:
                    print(f"    {param_names[i]} (change: {mean_changes[i]:.2e})")
        else:
            print(f"\n✅ Parameters are still evolving")
    
    # Check acceptance rates
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rates = sampler.acceptance_fraction
        mean_acc = np.mean(acc_rates)
        print(f"\nAcceptance rate: {mean_acc:.3f} ± {np.std(acc_rates):.3f}")
        
        if mean_acc < 0.1:
            print("⚠️  WARNING: Very low acceptance rate - sampling may be stuck")
        elif mean_acc > 0.8:
            print("⚠️  WARNING: Very high acceptance rate - may not be exploring enough")
        else:
            print("✅ Acceptance rate looks reasonable")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(chain)):
        print("⚠️  WARNING: Found NaN or infinite values in chain")
    else:
        print("✅ All chain values are finite")
    
    return {
        'n_steps': n_steps,
        'n_chains': n_chains,
        'n_params': n_params,
        'is_running': hasattr(sampler, 'running') and sampler.running,
        'acceptance_rate': mean_acc if 'mean_acc' in locals() else None
    }

def plot_chain_diagnostics(sampler, save_plot=True):
    """
    Create diagnostic plots to visualize chain behavior
    """
    chain = sampler.get_chain()
    n_chains, n_steps, n_params = chain.shape
    param_names = ["M_phi", "g", "si", "norm"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        
        # Plot all chains for this parameter
        for j in range(min(n_chains, 10)):  # Plot first 10 chains to avoid clutter
            ax.plot(chain[j, :, i], alpha=0.5, linewidth=0.5)
        
        # Plot mean across chains
        mean_chain = np.mean(chain[:, :, i], axis=0)
        ax.plot(mean_chain, 'k-', linewidth=2, label='Mean')
        
        ax.set_title(f'{param_names[i]}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        
        # Use log scale for parameters that span many orders of magnitude
        if i in [0, 1, 3]:  # M_phi, g, norm
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('mcmc_chain_diagnostics.png', dpi=150, bbox_inches='tight')
        print("Diagnostic plot saved as 'mcmc_chain_diagnostics.png'")
    
    plt.show()

def check_likelihood_evolution(sampler):
    """
    Check if likelihood values are evolving or stuck
    """
    if hasattr(sampler, 'get_log_prob'):
        log_prob = sampler.get_log_prob()
        n_chains, n_steps = log_prob.shape
        
        print(f"\n=== LIKELIHOOD EVOLUTION ===")
        print(f"Likelihood range: {np.min(log_prob):.2f} to {np.max(log_prob):.2f}")
        
        # Check recent likelihood changes
        if n_steps >= 50:
            recent_log_prob = log_prob[:, -50:]
            recent_changes = np.abs(recent_log_prob[:, -1] - recent_log_prob[:, 0])
            mean_change = np.mean(recent_changes)
            
            print(f"Mean likelihood change in last 50 iterations: {mean_change:.2f}")
            
            if mean_change < 0.1:
                print("⚠️  WARNING: Likelihood barely changing - may be stuck")
            else:
                print("✅ Likelihood is still evolving")
        
        # Plot likelihood evolution
        plt.figure(figsize=(10, 6))
        for i in range(min(n_chains, 10)):
            plt.plot(log_prob[i, :], alpha=0.5, linewidth=0.5)
        
        # Filter out infinite values before taking mean
        finite_log_prob = np.ma.masked_invalid(log_prob)
        mean_log_prob = np.ma.mean(finite_log_prob, axis=0)
        plt.plot(mean_log_prob, 'k-', linewidth=2, label='Mean (finite only)')
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Likelihood Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('likelihood_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # If you have a saved sampler, you can load it here
    # sampler = emcee.backends.HDFBackend("mcmc_backend.h5").get_sampler()
    
    print("To use this script:")
    print("1. Run this in the same notebook cell as your MCMC")
    print("2. Call: check_mcmc_status(sampler)")
    print("3. Call: plot_chain_diagnostics(sampler)")
    print("4. Call: check_likelihood_evolution(sampler)") 