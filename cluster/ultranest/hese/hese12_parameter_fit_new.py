import numpy as np
import pandas as pd
import ultranest
import corner
import nuSIprop
# Try to import matplotlib with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for clusters
    import matplotlib.pyplot as plt
    print("Successfully imported matplotlib")
except ImportError as e:
    print(f"Warning: Could not import matplotlib: {e}")
    print("Trying to install missing dependencies...")
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm
from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy.integrate import quad
import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import time


# Try to import MPI for distributed computing
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. MPI support disabled.")
# Get the project root directory dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
print('project_root', project_root)


def bin_centers_to_edges(bin_centers):
    log_centers = np.log10(bin_centers)
    dlog = np.diff(log_centers)
    log_edges = np.zeros(len(bin_centers) + 1)
    log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
    log_edges[0] = log_centers[0] - dlog[0]/2
    log_edges[-1] = log_centers[-1] + dlog[-1]/2
    return 10**log_edges


def bin_edges_to_centers(bin_edges):
    log_edges = np.log10(bin_edges)
    dlog = np.diff(log_edges)
    log_centers = np.zeros(len(bin_edges) - 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    return 10**log_centers


energy_bins = np.logspace(4, 7, 3*20+1)
energy_centers = bin_edges_to_centers(energy_bins)
energy_bins_low_resolution = np.logspace(4, 7, 20+1)
energy_centers_low_resolution = bin_edges_to_centers(energy_bins_low_resolution)

livetime12 = 12*365*24*3600

"""
# Use relative paths from project root
data_dir = project_root / '4_to_7_HESE12'

hese12_events_df = pd.read_csv(data_dir / '20bins' / 'hese12_20bins_df.csv', index_col=0)

background_df = pd.read_csv(data_dir / '20bins' / 'background_20bins_df.csv', index_col=0)
background_df.rename(columns={'background': 'events'}, inplace=True)

effective_area_df = pd.read_csv(data_dir / '60bins' / 'effective_area_4_to_7.csv', index_col=0)
"""

hese12_events_df = pd.read_csv('hese12_20bins_df.csv', index_col=0)

background_df = pd.read_csv('background_20bins_df.csv', index_col=0)
background_df.rename(columns={'background': 'events'}, inplace=True)

effective_area_df = pd.read_csv('effective_area_4_to_7.csv', index_col=0)


def initialize_evolver(M_phi=25, g_phi=0.01, mntot=0.1, si=2.5, norm=4.0*1e-18):
    # Initialize flux evolver object with arbitrary parameters and return energies, bin edges, and delta E
    # Only called once, as these are the same for all parameters

    evolver = nuSIprop.pyprop(mphi = M_phi*1e6, # Mediator mass [eV]
                g = g_phi, # Coupling
                mntot = mntot, # Sum of neutrino masses [eV]
                si = si, # Spectral index
                norm = norm, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
                majorana = True, # Majorana neutrinos? [Default = True]
                non_resonant = True, # Include non s-channel contributions? Relevant for couplings g>~0.1 [Default = True]
                normal_ordering = True, # Normal neutrino mass ordering? [Default = True]
                N_bins_E = 300, # Number of energy bins, uniformly distributed in log space [Default = 300]
                lEmin = 13, # log_10 (E_min/eV) [Default = 13]
                lEmax = 16, # log_10 (E_max/eV) [Default = 17]
                zmax = 5, # Largest redshift at which sources are included [Default = 5]
                flav = 2, # Flavor of interacting neutrinos [0=e, 1=mu, 2=tau. Default = 2]
                phiphi = False # Consider double-scalar production? If set to true, the files xsec/alpha_phiphi.bin and xsec/alphatilde_phiphi.bin must exist [Default = False]
                            )
    evolver.evolve()

    energy_centers_high_resolution = evolver.get_energies()
    bin_edges_high_resolution = bin_centers_to_edges(energy_centers_high_resolution)
    delta_E_high_resolution = np.diff(bin_edges_high_resolution)    # Delta E in eV   
    
    return evolver, energy_centers_high_resolution, bin_edges_high_resolution, delta_E_high_resolution

evolver, energies, bin_edges_high_resolution, delta_E = initialize_evolver()




def nuSIprop_events(flx, eff, livetime, norm, delta_E, save_to_csv=False):
    # Interpolate `flx` to the same energy bins as `eff`
    # Want to interpolate effective area as flux but for some reason, it messed up when I change it, so just do a 'rockad'
    # Therefore, send eff as flx, and vice versa!!
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)
    
    if eff['nu_e'].any() < 0:
        print('Negative effective area found')
    if eff['nu_mu'].any() < 0:
        print('Negative effective area found')
    if eff['nu_tau'].any() < 0:
        print('Negative effective area found')

    total_events_df = flx_interpolated * eff * livetime * norm 
    negative_mask = total_events_df < 0
    total_events_df[negative_mask] = 0
    total_events_df['total_events'] = delta_E * (total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau'])

    return total_events_df



def _apply_energy_smearing_serial(energies, events, resolution):
    """Serial version of energy smearing"""
    smeared_events = np.zeros_like(events)  
    
    for i, E_true in enumerate(energies):
        # Calculate sigma in linear space (resolution is fractional)
        sigma = resolution * E_true
        
        # Create Gaussian in linear space
        gaussian = np.exp(-0.5 * ((energies - E_true) / sigma) ** 2)
        
        # Normalize the Gaussian
        gaussian_sum = np.sum(gaussian)
        if gaussian_sum > 0:  # Avoid division by zero
            gaussian /= gaussian_sum
        
        # Redistribute events
        smeared_events += events[i] * gaussian
    
    # Verify event conservation
    total_events_before = np.sum(events)
    total_events_after = np.sum(smeared_events)
    if not np.isclose(total_events_before, total_events_after, rtol=1e-10):
        print(f"Warning: Event conservation violated! Before: {total_events_before}, After: {total_events_after}")
    
    return smeared_events




def prior_transform_nuSIprop(cube):
    """Vectorized prior transform: maps [0,1]^3 -> (Mphi, g, si)"""
    # Handle both single cube and multiple cubes
    """if cube.ndim == 1:
        print('SINGLE CUBE', cube.shape)
        cube = cube.reshape(1, -1)
        
    u_Mphi, u_g, u_si = cube[:, 0], cube[:, 1], cube[:, 2]
    """
    u_Mphi, u_g, u_mntot, u_si = cube


    # Mphi log-uniform in [0.1, 1000]
    Mphi_min, Mphi_max = 0.03, 100
    #Mphi = 10**(Mphi_min + (Mphi_max - Mphi_min) * u_Mphi)
    Mphi = Mphi_min * (Mphi_max / Mphi_min) ** u_Mphi

    # g log-uniform in [1e-4, 1]
    g_min, g_max = 1e-4, 1.0
    #g = 10**(g_min + (g_max - g_min) * u_g)
    g = g_min * (g_max / g_min) ** u_g
    
    # mntot uniform in [0.06, 0.12]
    #mntot_min, mntot_max = 0.06, 0.12
    #mntot = mntot_min + (mntot_max - mntot_min) * u_mntot

    # si uniform in [2.0, 3.0]
    si = 2.0 + (4.0 - 2.0) * u_si

    # Return with the same shape as input
    return Mphi, g, si




def log_likelihood_poisson(data, predicted):
    # From Poisson distribution 
    return np.sum(-predicted + data*np.log(predicted) - np.log(factorial(data)))

def log_likelihood_gaussian(data, predicted, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (data - predicted)**2 / sigma**2)




def create_custom_corner_plot(samples, param_names=['Mphi', 'g', 'si'], title='corner_plot_custom.png'):
    """Create a custom corner plot with log scaling for Mphi and g"""
    try:  
        # Create corner plot
        fig = corner.corner(samples, labels=param_names, 
                           quantiles=[0.68, 0.95, 0.997],
                           show_titles=True, title_kwargs={"fontsize": 12})
        
        # Apply log scaling to Mphi and g
        axes = fig.get_axes()
        n_params = len(param_names)
        
        for i in range(n_params):
            for j in range(n_params):
                ax_idx = i * n_params + j
                if ax_idx < len(axes):
                    ax = axes[ax_idx]
                    
                    # Apply log scaling based on parameter
                    if j < len(param_names) and param_names[j] in ['Mphi', 'g']:
                        ax.set_xscale('log')
                    if i < len(param_names) and param_names[i] in ['Mphi', 'g']:
                        ax.set_yscale('log')
        
        plt.savefig(title, bbox_inches='tight')
        print("Saved custom corner_plot_custom.png with log scaling")
        plt.close(fig)
        
    except ImportError:
        print("Corner package not available, skipping custom corner plot")
    except Exception as e:
        print(f"Error creating custom corner plot: {e}")


def create_manual_run_plot(sampler, result):
    """Create a manual run plot using UltraNest data"""
    try:
        # Get log likelihood data
        logl = result.get('logl', [])
        if len(logl) == 0:
            print("No log likelihood data available for run plot")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(logl, 'b-', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Log Likelihood')
        ax.set_title('UltraNest Run Plot')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if len(logl) > 0:
            ax.axhline(y=np.max(logl), color='r', linestyle='--', alpha=0.7, 
                      label=f'Max: {np.max(logl):.2f}')
            ax.axhline(y=np.mean(logl), color='g', linestyle='--', alpha=0.7, 
                      label=f'Mean: {np.mean(logl):.2f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('run_plot_manual.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Saved manual run_plot_manual.png")
        
    except Exception as e:
        print(f"Error creating manual run plot: {e}")


def create_manual_trace_plot(sampler, result):
    """Create a manual trace plot using UltraNest data"""
    try:
        # Get samples and parameter names
        samples = result.get('samples', [])
        param_names = ['Mphi', 'g', 'si']
        
        if len(samples) == 0:
            print("No samples available for trace plot")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 8))
        if len(param_names) == 1:
            axes = [axes]
            
        for i, param in enumerate(param_names):
            if i < len(axes):
                ax = axes[i]
                ax.plot(samples[:, i], 'b-', alpha=0.7, linewidth=0.5)
                ax.set_ylabel(param)
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                if len(samples) > 0:
                    ax.axhline(y=np.mean(samples[:, i]), color='r', linestyle='--', alpha=0.7,
                              label=f'Mean: {np.mean(samples[:, i]):.3f}')
                    ax.legend()
        
        axes[-1].set_xlabel('Sample')
        plt.suptitle('UltraNest Trace Plot')
        plt.tight_layout()
        plt.savefig('trace_plot_manual.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Saved manual trace_plot_manual.png")
        
    except Exception as e:
        print(f"Error creating manual trace plot: {e}")


def log_likelihood(theta):
    Mphi, g, si = theta


    # update rank-local evolver
    #evolver.set_parameters(mphi=Mphi*1e6, g=g, si=si)
    evolver.set_parameters(mphi=Mphi*1e6, g=g, si=si)
    evolver.evolve()
    flux = evolver.get_flux_fla()

    flx_df = pd.DataFrame(flux.T, index=energies, columns=['nu_e', 'nu_mu', 'nu_tau'])
    flx_df.index = flx_df.index / 1e9

    nuSIprop_df = nuSIprop_events(
        flx=effective_area_df, eff=flx_df,
        livetime=livetime12, norm=1e-4, delta_E=delta_E
    )

    nuSIprop_smeared = _apply_energy_smearing_serial(
        energies=nuSIprop_df.index.values,
        events=nuSIprop_df['total_events'].values,
        resolution=0.1
    )

    nuSIprop_binned_events, _ = np.histogram(
        nuSIprop_df.index.values,
        weights=nuSIprop_smeared,
        bins=energy_bins_low_resolution
    )

    predicted = nuSIprop_binned_events + background_df['events'].values

    data = hese12_events_df['events'].values.astype(int)
    # Poisson log-likelihood
    return log_likelihood_poisson(data=data, predicted=predicted)


def run_ultranest(param_names=['Mphi', 'g', 'si'], loglike=log_likelihood, prior_transform=prior_transform_nuSIprop, log_dir='ultranest_results/hese/nuSIprop/Majorana_NO/mntot_0065', resume='subfolder'):
    # Valid for all flux models
    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        loglike,
        prior_transform,
        log_dir=log_dir,
        resume=resume
    )

    print('Running sampler...')
    result = sampler.run(dlogz=0.2, dKL=0.2)
    #result = sampler.run()
    if MPI_AVAILABLE and MPI.COMM_WORLD.rank == 0:
        print('Result: ', result)
        sampler.print_results()
        
        sampler.plot_run()
        sampler.plot_trace()
        sampler.plot_corner()
        
    elif not MPI_AVAILABLE:
        sampler.print_results()
        print('not MPI_AVAILABLE')
    return result


# --- Run UltraNest ---
if __name__ == "__main__":
    
    run_ultranest(param_names=['Mphi', 'g', 'si'], loglike=log_likelihood, prior_transform=prior_transform_nuSIprop, log_dir='ultranest_results/hese/nuSIprop/Majorana_NO/mntot_0065', resume='subfolder')

    
    """sampler = ultranest.ReactiveNestedSampler(
        ["Mphi", "g", "mntot", "si"],
        log_likelihood,
        prior_transform,
    )


    result = sampler.run(min_num_live_points=800)
    

    # Print results (only from rank 0 in MPI runs)
    if MPI_AVAILABLE and MPI.COMM_WORLD.rank == 0:
        print('Result: ', result)
        print('hej')
        sampler.print_results()
        print('MPI_AVAILABLE and MPI.COMM_WORLD.rank == 0')
        """sampler.plot_corner()
        plt.savefig('corner_plot_mpi.png', dpi=300, bbox_inches='tight')
        plt.close()"""

        samples = sampler.results['samples']
        print('samples: ', samples)

        # Transform selected axes to log10
        log_samples = samples.copy()
        log_samples[:, 0] = np.log10(samples[:, 0])   # log scale for Mphi
        log_samples[:, 1] = np.log10(samples[:, 1])   # log scale for g

        fig1 = corner.corner(
            log_samples,
            labels=[r"$\log_{10} M_\phi$", r"$\log_{10} g$", r"$\sum m_{\nu}$", r"$\gamma$"],
            range=[(np.log10(0.1), np.log10(1000)),   # log range
                (np.log10(1e-4), np.log10(1.0)),   # log range
                (0.06, 0.12),                       # linear range
                (2.0, 4.0)],                       # linear range
            quantiles=[0.68, 0.95, 0.997],
            show_titles=True,
            title_fmt=".2f"
        )
        fig1.savefig("corner_log_transformed_samples.png")
        #fig1.close()
        
        fig2 = create_custom_corner_plot(samples, param_names=['Mphi', 'g', 'mntot', 'si'], title='corner_custom.png')
        #fig2.close()
        
        run_plot = sampler.plot_run()
        run_plot.savefig("run_plot.png")
        #run_plot.close()
        
        trace_plot = sampler.plot_trace()
        trace_plot.savefig("trace_plot.png")
        #trace_plot.close()
        
        create_manual_run_plot(sampler, result)
        create_manual_trace_plot(sampler, result)
        
        print('hej2')
        
        weights = result["weights"]        # or sampler.results['weights']
        ess = (weights.sum())**2 / (weights**2).sum()
        print("Effective Sample Size:", ess)


    elif not MPI_AVAILABLE:
        sampler.print_results()
        print('not MPI_AVAILABLE')"""




"""def run_ultranest(output_dir=None, max_ncalls=None, n_cores=None, use_mpi=False):
    # --- Run UltraNest ---
    if output_dir is None:
        output_dir = Path.cwd() / 'ultranest_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Handle MPI setup
    if use_mpi and MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank: {rank}, size: {size}")
        
        if rank == 0:
            print(f"Running with MPI: {size} processes")
        else:
            print(f"MPI process {rank}/{size} starting...")
        
        # Adjust output directory for MPI
        if size > 1:
            output_dir = output_dir / f"rank_{rank}"
            output_dir.mkdir(exist_ok=True)
    else:
        rank = 0
        size = 1
        print("Running without MPI")
    
    # Print system information
    print(f"System info:")
    print(f"  CPU cores: {mp.cpu_count()} logical cores")
    print(f"  Process ID: {os.getpid()}")
    print(f"  Working directory: {os.getcwd()}")
    
    sampler = ultranest.ReactiveNestedSampler(
        ['Mphi', 'g', 'si'],  # parameter names
        loglike,
        prior_transform,
        log_dir=str(output_dir),
        vectorized=True  # Enable UltraNest's built-in parallelization
    )
    
    # Set maximum number of function calls if specified
    if max_ncalls is not None:
        result = sampler.run(max_ncalls=max_ncalls)
    else:
        result = sampler.run()
    
    # Print results to stdout (will be captured in job output)
    sampler.print_results()
    
    # Save results to files
    try:
        import json
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to: {output_dir / 'results.json'}")
    except Exception as e:
        print(f"Warning: Could not save results.json: {e}")
    
    # Save corner plot to file (non-interactive)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        sampler.plot_corner()
        plt.savefig(str(output_dir / 'corner_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Corner plot saved to: {output_dir / 'corner_plot.png'}")
    except Exception as e:
        print(f"Warning: Could not save corner plot: {e}")
    
    # Save summary statistics
    try:
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("UltraNest Results Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Log-evidence: {result['logz']:.3f} ± {result['logzerr']:.3f}\n")
        f.write(f"Number of live points: {result['ncall']}\n")
        f.write(f"Number of iterations: {result['niter']}\n")
        f.write("\nParameter estimates:\n")
        for i, param in enumerate(['Mphi', 'g', 'si']):
            f.write(f"{param}: {result['posterior']['mean'][i]:.6f} ± {result['posterior']['stdev'][i]:.6f}\n")
        print(f"Summary saved to: {output_dir / 'summary.txt'}")
    except Exception as e:
        print(f"Warning: Could not save summary.txt: {e}")
    
    print(f"Results saved to: {output_dir}")
    return result



if __name__ == "__main__":
    # Parse command line arguments for cluster usage
    parser = argparse.ArgumentParser(description='Run UltraNest parameter fitting for HESE12 data')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for results (default: ./ultranest_results)')
    parser.add_argument('--max-ncalls', type=int, default=None,
                       help='Maximum number of function calls (for time-limited jobs)')
    parser.add_argument('--n-cores', type=int, default=None,
                       help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--use-mpi', action='store_true',
                       help='Use MPI for distributed computing (requires mpi4py)')
    
    args = parser.parse_args()
    
    print("Starting UltraNest parameter fitting...")
    print(f"Output directory: {args.output_dir or './ultranest_results'}")
    if args.max_ncalls:
        print(f"Maximum function calls: {args.max_ncalls}")
    if args.n_cores:
        print(f"Using {args.n_cores} CPU cores")
    else:
        print(f"Using all available CPU cores ({mp.cpu_count()})")
    
    if args.use_mpi:
        if MPI_AVAILABLE:
            print("MPI support enabled")
        else:
            print("Warning: MPI requested but mpi4py not available. Running without MPI.")
            args.use_mpi = False
    
    result = run_ultranest(output_dir=args.output_dir, max_ncalls=args.max_ncalls, 
                          n_cores=args.n_cores, use_mpi=args.use_mpi)
    print("UltraNest fitting completed successfully!")"""
