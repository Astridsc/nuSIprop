#!/usr/bin/env python3
"""
Real-time monitoring for MCMC runs
"""

import numpy as np
import matplotlib.pyplot as plt
#import time
from tqdm import tqdm

class MCMCMonitor:
    """Real-time monitor for MCMC progress"""
    
    def __init__(self, check_interval=10):
        self.check_interval = check_interval
        self.acceptance_history = []
        self.iteration_history = []
        self.parameter_history = []
        self.likelihood_history = []
        
    def monitor_step(self, sampler, current_iteration):
        """Monitor one step of MCMC"""
        if current_iteration % self.check_interval == 0:
            # Get current acceptance rates
            if hasattr(sampler, 'acceptance_fraction'):
                acc_rates = sampler.acceptance_fraction
                mean_acc = np.mean(acc_rates)
                std_acc = np.std(acc_rates)
                
                self.acceptance_history.append(mean_acc)
                self.iteration_history.append(current_iteration)
                
                print(f"Iteration {current_iteration}: Acceptance rate = {mean_acc:.3f} ± {std_acc:.3f}")
                
                # Check for stuck behavior
                if mean_acc < 0.01:
                    print("🚨 WARNING: Very low acceptance rate - sampling may be stuck!")
                elif mean_acc < 0.1:
                    print("⚠️  WARNING: Low acceptance rate")
                elif mean_acc > 0.8:
                    print("⚠️  WARNING: Very high acceptance rate")
                else:
                    print("✅ Acceptance rate looks good")
                
                # Get current parameter statistics
                chain = sampler.get_chain()
                if chain.shape[1] > 0:
                    #current_params = chain[:, -1, :]  # Latest parameter values
                    current_params = chain[-1, :, :]  # Latest parameter values
                    param_means = np.mean(current_params, axis=1)
                    param_stds = np.std(current_params, axis=1)
                    
                    self.parameter_history.append(param_means)
                    
                    param_names = ["M_phi", "g", "si", "norm"]
                    print("  Current parameter means:")
                    for i, name in enumerate(param_names):
                        print(f"    {name}: {param_means[i]:.2e} ± {param_stds[i]:.2e}")
                
                # Get likelihood information
                if hasattr(sampler, 'get_log_prob'):
                    log_prob = sampler.get_log_prob()
                    if log_prob.shape[1] > 0:
                        #current_likelihoods = log_prob[:, -1]
                        current_likelihoods = log_prob[-1, :]
                        mean_likelihood = np.mean(current_likelihoods)
                        max_likelihood = np.max(current_likelihoods)
                        
                        self.likelihood_history.append(mean_likelihood)
                        
                        print(f"  Likelihood: mean = {mean_likelihood:.2f}, max = {max_likelihood:.2f}")
    
    def plot_monitoring_history(self):
        """Plot the monitoring history"""
        if not self.iteration_history:
            print("No monitoring data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Acceptance rate evolution
        axes[0, 0].plot(self.iteration_history, self.acceptance_history, 'b-', linewidth=2)
        axes[0, 0].axhline(0.2, color='green', linestyle='--', alpha=0.7, label='Target min')
        axes[0, 0].axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Target max')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Mean Acceptance Rate')
        axes[0, 0].set_title('Acceptance Rate Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter evolution
        if self.parameter_history:
            param_history = np.array(self.parameter_history)
            param_names = ["M_phi", "g", "si", "norm"]
            
            for i, name in enumerate(param_names):
                axes[0, 1].plot(self.iteration_history, param_history[:, i], 
                               label=name, linewidth=2)
            
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Parameter Value')
            axes[0, 1].set_title('Parameter Evolution')
            axes[0, 1].legend()
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Likelihood evolution
        if self.likelihood_history:
            axes[1, 0].plot(self.iteration_history, self.likelihood_history, 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Mean Log Likelihood')
            axes[1, 0].set_title('Likelihood Evolution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Acceptance rate distribution
        if self.acceptance_history:
            #axes[1, 1].hist(self.acceptance_history, bins=20, alpha=0.7, edgecolor='black')
            #axes[1, 1].axvline(np.mean(self.acceptance_history), color='red', linestyle='--', 
            #                  label=f'Mean: {np.mean(self.acceptance_history):.3f}')
            axes[1, 1].plot(self.iteration_history, self.acceptance_history, 'b-', linewidth=2, label='Mean = ' + str(np.mean(self.acceptance_history)))
            axes[1, 1].set_xlabel('Acceptance Rate')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Acceptance Rate Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mcmc_monitoring_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def run_mcmc_with_monitoring(sampler, initial_positions, n_iterations, monitor_interval=10):
    """
    Run MCMC with real-time monitoring
    """
    monitor = MCMCMonitor(check_interval=monitor_interval)
    
    print(f"Starting MCMC with {len(initial_positions)} walkers for {n_iterations} iterations")
    print(f"Monitoring every {monitor_interval} iterations")
    
    # Run MCMC with monitoring
    for i in tqdm(range(n_iterations), desc="MCMC Progress"):
        sampler.run_mcmc(initial_positions, 1, progress=False)
        monitor.monitor_step(sampler, i + 1)
    
    print(f"\nMCMC completed! Total iterations: {sampler.get_chain().shape[1]}")
    
    # Plot monitoring history
    monitor.plot_monitoring_history()
    
    return sampler, monitor

def quick_acceptance_check(sampler):
    """
    Quick check of acceptance rates
    """
    if hasattr(sampler, 'acceptance_fraction'):
        acc_rates = sampler.acceptance_fraction
        mean_acc = np.mean(acc_rates)
        std_acc = np.std(acc_rates)
        
        print(f"=== ACCEPTANCE RATE CHECK ===")
        print(f"Mean acceptance rate: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"Range: {np.min(acc_rates):.3f} to {np.max(acc_rates):.3f}")
        
        if mean_acc == 0:
            print("🚨 CRITICAL: 0% acceptance rate - MCMC is completely stuck!")
            print("Possible causes:")
            print("1. Likelihood function returning -inf")
            print("2. Step sizes too large")
            print("3. Numerical issues in calculations")
            print("4. All proposals outside prior bounds")
        elif mean_acc < 0.01:
            print("🚨 CRITICAL: Very low acceptance rate - MCMC is stuck!")
        elif mean_acc < 0.1:
            print("⚠️  WARNING: Low acceptance rate - may be stuck")
        elif mean_acc > 0.8:
            print("⚠️  WARNING: Very high acceptance rate - may not be exploring enough")
        else:
            print("✅ Acceptance rate looks reasonable")
        
        # Show individual walker rates
        print(f"\nIndividual walker acceptance rates:")
        for i, acc in enumerate(acc_rates):
            status = "❌" if acc == 0 else "⚠️" if acc < 0.1 else "✅"
            print(f"  Walker {i}: {acc:.3f} {status}")
        
        return mean_acc
    else:
        print("No acceptance fraction data available")
        return None

# Usage instructions
print("To use real-time monitoring:")
print("1. Create monitor: monitor = MCMCMonitor(check_interval=10)")
print("2. Run with monitoring: sampler, monitor = run_mcmc_with_monitoring(sampler, pos, 300)")
print("3. Quick check: quick_acceptance_check(sampler)")
print("4. Plot history: monitor.plot_monitoring_history()")