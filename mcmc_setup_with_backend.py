#!/usr/bin/env python3
"""
MCMC setup with backend for progress recovery and monitoring
"""

import numpy as np
import emcee
import os
import time
from datetime import datetime

def setup_mcmc_with_backend(log_posterior, initial_positions, backend_filename="mcmc_backend.h5", 
                           n_walkers=None, n_dim=None):
    """
    Set up MCMC sampler with backend for progress recovery
    
    Parameters:
    -----------
    log_posterior : function
        Your log posterior function
    initial_positions : array
        Initial positions for walkers
    backend_filename : str
        Filename for the backend
    n_walkers : int, optional
        Number of walkers (inferred from initial_positions if not provided)
    n_dim : int, optional
        Number of parameters (inferred from initial_positions if not provided)
    
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        Configured sampler with backend
    backend : emcee.backends.HDFBackend
        Backend object for monitoring
    """
    
    if n_walkers is None:
        n_walkers = len(initial_positions)
    if n_dim is None:
        n_dim = initial_positions.shape[1]
    
    # Create backend to store results
    backend = emcee.backends.HDFBackend(backend_filename)
    
    # Reset backend (clear any previous data)
    backend.reset(n_walkers, n_dim)
    
    # Create sampler with backend
    sampler = emcee.EnsembleSampler(
        n_walkers,  # number of walkers
        n_dim,      # number of parameters
        log_posterior,  # your log posterior function
        backend=backend  # this stores the results
    )
    
    print(f"MCMC sampler created with backend: {backend_filename}")
    print(f"Number of walkers: {n_walkers}")
    print(f"Number of parameters: {n_dim}")
    print(f"Backend will save progress automatically")
    
    return sampler, backend

def run_mcmc_with_monitoring(sampler, initial_positions, n_iterations, 
                           progress=True, save_interval=100, 
                           checkpoint_interval=500):
    """
    Run MCMC with progress monitoring and checkpointing
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        Sampler with backend
    initial_positions : array
        Initial positions
    n_iterations : int
        Number of iterations to run
    progress : bool
        Show progress bar
    save_interval : int
        How often to save progress info (iterations)
    checkpoint_interval : int
        How often to print detailed status (iterations)
    """
    
    print(f"Starting MCMC with {n_iterations} iterations...")
    print(f"Progress will be saved every {save_interval} iterations")
    print(f"Checkpoint status every {checkpoint_interval} iterations")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get current iteration if resuming
    current_iteration = sampler.backend.iteration
    if current_iteration > 0:
        print(f"Resuming from iteration {current_iteration}")
        remaining_iterations = n_iterations - current_iteration
        if remaining_iterations <= 0:
            print("MCMC already completed!")
            return sampler
        print(f"Remaining iterations: {remaining_iterations}")
        n_iterations = remaining_iterations
    
    # Run MCMC with monitoring
    start_time = time.time()
    
    try:
        sampler.run_mcmc(initial_positions, n_iterations, progress=progress)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nMCMC completed successfully!")
        print(f"Total iterations: {sampler.backend.iteration}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average time per iteration: {total_time/sampler.backend.iteration:.2f} seconds")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\nMCMC interrupted by user at iteration {sampler.backend.iteration}")
        print(f"Progress saved to backend. You can resume later.")
        print(f"Time elapsed: {(time.time() - start_time)/3600:.2f} hours")
    
    return sampler

def check_mcmc_progress(backend_filename="mcmc_backend.h5"):
    """
    Check the progress of an MCMC run
    
    Parameters:
    -----------
    backend_filename : str
        Backend filename
    
    Returns:
    --------
    dict : Progress information
    """
    if not os.path.exists(backend_filename):
        print(f"Backend file {backend_filename} not found")
        return None
    
    backend = emcee.backends.HDFBackend(backend_filename)
    
    print(f"=== MCMC PROGRESS CHECK ===")
    print(f"Backend file: {backend_filename}")
    print(f"Current iteration: {backend.iteration}")
    print(f"Number of walkers: {backend.shape[0]}")
    print(f"Number of parameters: {backend.shape[1]}")
    
    if backend.iteration > 0:
        # Get some basic statistics
        chain = backend.get_chain()
        log_prob = backend.get_log_prob()
        
        print(f"Chain shape: {chain.shape}")
        print(f"Log probability shape: {log_prob.shape}")
        
        # Check for infinite values
        n_infinite = np.sum(~np.isfinite(log_prob))
        n_total = log_prob.size
        print(f"Infinite log probabilities: {n_infinite}/{n_total} ({100*n_infinite/n_total:.1f}%)")
        
        # Show recent log probability statistics
        if log_prob.shape[0] > 0:
            recent_log_prob = log_prob[-1, :]  # Last iteration
            finite_log_prob = recent_log_prob[np.isfinite(recent_log_prob)]
            if len(finite_log_prob) > 0:
                print(f"Recent log probability - Mean: {np.mean(finite_log_prob):.2f}, Std: {np.std(finite_log_prob):.2f}")
    
    return {
        'iteration': backend.iteration,
        'shape': backend.shape,
        'filename': backend_filename
    }

def resume_mcmc(backend_filename="mcmc_backend.h5", target_iterations=None):
    """
    Resume an interrupted MCMC run
    
    Parameters:
    -----------
    backend_filename : str
        Backend filename
    target_iterations : int, optional
        Total target iterations (if not provided, will run indefinitely)
    
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        Resumed sampler
    """
    if not os.path.exists(backend_filename):
        print(f"Backend file {backend_filename} not found")
        return None
    
    backend = emcee.backends.HDFBackend(backend_filename)
    sampler = backend.get_sampler()
    
    print(f"Resuming MCMC from iteration {backend.iteration}")
    
    if target_iterations is not None:
        remaining = target_iterations - backend.iteration
        if remaining <= 0:
            print("MCMC already completed!")
            return sampler
        print(f"Will run {remaining} more iterations to reach {target_iterations}")
        return run_mcmc_with_monitoring(sampler, None, remaining)
    else:
        print("Will run indefinitely (use Ctrl+C to stop)")
        return run_mcmc_with_monitoring(sampler, None, 1000000)  # Large number

def save_mcmc_results_from_backend(backend_filename="mcmc_backend.h5", 
                                  output_dir="mcmc_results",
                                  discard=100):
    """
    Save MCMC results from backend to various formats
    
    Parameters:
    -----------
    backend_filename : str
        Backend filename
    output_dir : str
        Output directory for results
    discard : int
        Number of initial iterations to discard as burn-in
    """
    if not os.path.exists(backend_filename):
        print(f"Backend file {backend_filename} not found")
        return
    
    backend = emcee.backends.HDFBackend(backend_filename)
    sampler = backend.get_sampler()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving MCMC results from {backend_filename}...")
    
    # Get chain and log probability
    chain = backend.get_chain(discard=discard)
    log_prob = backend.get_log_prob(discard=discard)
    
    # Save as numpy arrays
    np.save(f"{output_dir}/mcmc_chain_final.npy", chain)
    np.save(f"{output_dir}/mcmc_log_prob_final.npy", log_prob)
    
    # Save flattened samples as CSV
    samples = backend.get_chain(discard=discard, flat=True)
    param_names = ["M_phi", "g", "si", "norm"]
    
    import pandas as pd
    df = pd.DataFrame(samples, columns=param_names)
    df.to_csv(f"{output_dir}/mcmc_parameter_samples.csv", index=False)
    
    # Save summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv(f"{output_dir}/mcmc_summary_statistics.csv")
    
    print(f"Results saved to {output_dir}/")
    print(f"  - mcmc_chain_final.npy")
    print(f"  - mcmc_log_prob_final.npy") 
    print(f"  - mcmc_parameter_samples.csv")
    print(f"  - mcmc_summary_statistics.csv")

# Example usage
if __name__ == "__main__":
    print("=== MCMC SETUP WITH BACKEND ===")
    print()
    print("To use this in your notebook:")
    print()
    print("1. Setup with backend:")
    print("   sampler, backend = setup_mcmc_with_backend(log_posterior, initial_positions)")
    print()
    print("2. Run with monitoring:")
    print("   sampler = run_mcmc_with_monitoring(sampler, initial_positions, 5000)")
    print()
    print("3. Check progress anytime:")
    print("   check_mcmc_progress('mcmc_backend.h5')")
    print()
    print("4. Resume if interrupted:")
    print("   sampler = resume_mcmc('mcmc_backend.h5', target_iterations=5000)")
    print()
    print("5. Save results:")
    print("   save_mcmc_results_from_backend('mcmc_backend.h5')") 