#!/usr/bin/env python3
"""
MCMC Rescue Script - Fixes for stuck MCMC
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt

def rescue_stuck_mcmc(sampler, strategy='restart_better'):
    """
    Attempt to rescue a stuck MCMC
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        The stuck sampler
    strategy : str
        'restart_better' : Restart with better parameters
        'continue_adaptive' : Continue with adaptive step sizes
        'save_and_restart' : Save current state and restart
    """
    
    print(f"Attempting to rescue MCMC using strategy: {strategy}")
    
    if strategy == 'restart_better':
        return restart_with_better_params(sampler)
    elif strategy == 'continue_adaptive':
        return continue_with_adaptive_steps(sampler)
    elif strategy == 'save_and_restart':
        return save_and_restart(sampler)
    else:
        print("Unknown strategy. Using 'restart_better'")
        return restart_with_better_params(sampler)

def restart_with_better_params(sampler):
    """
    Restart MCMC with better parameters
    """
    print("Restarting with improved parameters...")
    
    # Get current best positions
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    # Find best walkers (highest likelihood)
    best_walkers = np.argmax(log_prob, axis=0)
    best_positions = chain[best_walkers, -1, :]
    
    # Create new initial positions with better spread
    n_walkers, n_params = best_positions.shape
    
    # Use log-space for better exploration
    log_best = np.log(best_positions)
    
    # Create new positions with wider spread
    new_positions = np.zeros_like(best_positions)
    
    # Different strategies for different parameters
    for i in range(n_params):
        if i in [0, 1, 3]:  # M_phi, g, norm - use log-space
            # Wider spread in log-space
            spread = 0.5  # Increased from 0.25
            new_positions[:, i] = np.exp(log_best[i] + spread * np.random.randn(n_walkers))
        else:  # si - use linear space
            # Wider spread in linear space
            spread = 0.3  # Increased from 0.15
            new_positions[:, i] = best_positions[i] + spread * np.random.randn(n_walkers)
    
    # Ensure parameters stay within bounds
    new_positions[:, 1] = np.clip(new_positions[:, 1], 1e-4, 1)  # g
    new_positions[:, 2] = np.clip(new_positions[:, 2], 2.0, 3.0)  # si
    new_positions[:, 3] = np.clip(new_positions[:, 3], 1e-19, 1e-17)  # norm
    
    print("New initial positions created with wider spread")
    print(f"Parameter ranges:")
    param_names = ["M_phi", "g", "si", "norm"]
    for i, name in enumerate(param_names):
        print(f"  {name}: {np.min(new_positions[:, i]):.2e} to {np.max(new_positions[:, i]):.2e}")
    
    return new_positions

def continue_with_adaptive_steps(sampler):
    """
    Continue MCMC with adaptive step sizes
    """
    print("Continuing with adaptive step sizes...")
    
    # Get current chain
    chain = sampler.get_chain()
    n_walkers, n_steps, n_params = chain.shape
    
    # Calculate current parameter variances
    current_vars = np.var(chain[:, -50:, :], axis=(0, 1))  # Last 50 iterations
    
    # Create new positions with adaptive step sizes
    current_positions = chain[:, -1, :]  # Current positions
    
    # Adaptive step sizes based on current variance
    step_sizes = np.sqrt(current_vars) * 2.4 / np.sqrt(n_params)  # Optimal step size
    
    new_positions = current_positions + step_sizes * np.random.randn(n_walkers, n_params)
    
    print("Adaptive step sizes calculated:")
    param_names = ["M_phi", "g", "si", "norm"]
    for i, name in enumerate(param_names):
        print(f"  {name} step size: {step_sizes[i]:.2e}")
    
    return new_positions

def save_and_restart(sampler):
    """
    Save current state and restart with completely new positions
    """
    print("Saving current state and restarting...")
    
    # Save current results
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    # Save to file
    np.save('mcmc_chain_backup.npy', chain)
    np.save('mcmc_log_prob_backup.npy', log_prob)
    print("Current state saved to 'mcmc_chain_backup.npy' and 'mcmc_log_prob_backup.npy'")
    
    # Create completely new initial positions
    n_walkers, n_params = chain.shape[0], chain.shape[2]
    
    # Use wider, more conservative initial positions
    initial_guess = [20, 0.01, 2.5, 1e-18]  # Your original guess
    
    new_positions = np.zeros((n_walkers, n_params))
    
    # Much wider initial spread
    for i in range(n_walkers):
        # M_phi: wider range around 20
        new_positions[i, 0] = np.random.uniform(5, 50)
        
        # g: wider range around 0.01
        new_positions[i, 1] = np.random.uniform(1e-3, 0.1)
        
        # si: wider range around 2.5
        new_positions[i, 2] = np.random.uniform(2.0, 3.0)
        
        # norm: wider range around 1e-18
        new_positions[i, 3] = np.random.uniform(1e-19, 1e-17)
    
    print("New initial positions created with much wider spread")
    return new_positions

def create_improved_sampler(new_positions, n_iterations=300):
    """
    Create an improved sampler with better settings
    """
    print("Creating improved sampler...")
    
    # Import your functions (you'll need to adjust this)
    # from your_notebook import log_posterior, your_args
    
    # Create sampler with backend for saving progress
    backend = emcee.backends.HDFBackend("mcmc_backend_improved.h5")
    backend.reset(len(new_positions), new_positions.shape[1])
    
    # You'll need to replace this with your actual log_posterior function
    # sampler = emcee.EnsembleSampler(
    #     len(new_positions), new_positions.shape[1], log_posterior,
    #     args=your_args, backend=backend
    # )
    
    print("Improved sampler created with backend saving")
    print("Backend file: mcmc_backend_improved.h5")
    
    return backend  # Return backend for now

def quick_diagnosis(sampler):
    """
    Quick diagnosis of current state
    """
    chain = sampler.get_chain()
    n_chains, n_steps, n_params = chain.shape
    
    print(f"Quick diagnosis:")
    print(f"  Iterations: {n_steps}")
    print(f"  Walkers: {n_chains}")
    
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rate = np.mean(sampler.acceptance_fraction)
        print(f"  Acceptance rate: {acc_rate:.3f}")
        
        if acc_rate < 0.1:
            print("  ⚠️  Low acceptance rate - likely stuck")
        elif acc_rate > 0.8:
            print("  ⚠️  High acceptance rate - may not be exploring")
        else:
            print("  ✅ Acceptance rate looks okay")
    
    # Check recent parameter changes
    if n_steps >= 20:
        recent_chain = chain[:, -20:, :]
        changes = np.abs(recent_chain[:, -1, :] - recent_chain[:, 0, :])
        mean_changes = np.mean(changes, axis=0)
        
        param_names = ["M_phi", "g", "si", "norm"]
        print(f"  Recent parameter changes:")
        for i, name in enumerate(param_names):
            print(f"    {name}: {mean_changes[i]:.2e}")
            
            if mean_changes[i] < 1e-6:
                print(f"      ⚠️  {name} appears stuck!")

# Usage instructions
print("To rescue your stuck MCMC:")
print("1. Run: new_positions = rescue_stuck_mcmc(sampler, 'restart_better')")
print("2. Run: quick_diagnosis(sampler)  # Check current state")
print("3. Use new_positions to restart your MCMC")
print("4. Consider using a backend to save progress") 