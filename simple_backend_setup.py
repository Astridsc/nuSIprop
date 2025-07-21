#!/usr/bin/env python3
"""
Simple backend setup for MCMC - just add backend to existing code
"""

import emcee
import os
import numpy as np

def create_sampler_with_backend(nwalkers, ndim, log_posterior, backend_filename="mcmc_backend.h5"):
    """
    Create sampler with backend - drop-in replacement for your existing code
    
    Parameters:
    -----------
    nwalkers : int
        Number of walkers
    ndim : int
        Number of parameters
    log_posterior : function
        Your log posterior function
    backend_filename : str
        Backend filename
    
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        Sampler with backend
    """
    
    # Create backend
    backend = emcee.backends.HDFBackend(backend_filename)
    
    # Reset backend (clear any previous data)
    backend.reset(nwalkers, ndim)
    
    # Create sampler with backend (just like your existing code)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend)
    
    print(f"Sampler created with backend: {backend_filename}")
    print(f"Progress will be saved automatically")
    
    return sampler

def check_progress(backend_filename="mcmc_backend.h5"):
    """
    Quick progress check
    
    Parameters:
    -----------
    backend_filename : str
        Backend filename
    """
    if not os.path.exists(backend_filename):
        print(f"Backend file {backend_filename} not found")
        return
    
    backend = emcee.backends.HDFBackend(backend_filename)
    
    print(f"Current iteration: {backend.iteration}")
    print(f"Walkers: {backend.shape[0]}")
    print(f"Parameters: {backend.shape[1]}")
    
    if backend.iteration > 0:
        # Get recent log probability
        log_prob = backend.get_log_prob()
        if log_prob.shape[0] > 0:
            recent_log_prob = log_prob[-1, :]
            finite_log_prob = recent_log_prob[np.isfinite(recent_log_prob)]
            if len(finite_log_prob) > 0:
                print(f"Recent log prob - Mean: {np.mean(finite_log_prob):.2f}")

def resume_from_backend(backend_filename="mcmc_backend.h5", target_iterations=None):
    """
    Resume MCMC from backend
    
    Parameters:
    -----------
    backend_filename : str
        Backend filename
    target_iterations : int, optional
        Target total iterations
    
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
    
    print(f"Resuming from iteration {backend.iteration}")
    
    if target_iterations is not None:
        remaining = target_iterations - backend.iteration
        if remaining <= 0:
            print("MCMC already completed!")
            return sampler
        print(f"Will run {remaining} more iterations")
        return sampler, remaining
    else:
        return sampler, None

# Example usage - replace your existing code with this:
if __name__ == "__main__":
    print("=== SIMPLE BACKEND SETUP ===")
    print()
    print("Replace your existing code:")
    print()
    print("OLD CODE:")
    print("sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)")
    print("sampler.run_mcmc(pos, 2000, progress=True)")
    print()
    print("NEW CODE:")
    print("sampler = create_sampler_with_backend(nwalkers, ndim, log_posterior)")
    print("sampler.run_mcmc(pos, 2000, progress=True)")
    print()
    print("Check progress anytime:")
    print("check_progress('mcmc_backend.h5')")
    print()
    print("Resume if interrupted:")
    print("sampler, remaining = resume_from_backend('mcmc_backend.h5', 2000)")
    print("if remaining:")
    print("    sampler.run_mcmc(None, remaining, progress=True)") 