"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm
from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy.integrate import quad
import os
import sys
sys.path.append('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release')
from Astrid.effective_area import bin_edges_to_centers, bin_centers_to_edges, apply_energy_smearing
from simple_backend_setup import create_sampler_with_backend
import nuSIprop"""

import numpy as np
import pandas as pd
import ultranest
#import arviz as az
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

hese12_events_df = pd.read_csv('hese12_20bins_df.csv', index_col=0)

background_df = pd.read_csv('background_20bins_df.csv', index_col=0)
background_df.rename(columns={'background': 'events'}, inplace=True)

effective_area_df = pd.read_csv('effective_area_4_to_7.csv', index_col=0)



#energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
#energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]
# Energy bins within sensitivity range of SPL model
energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]
energy_edges_MESE = np.asarray(energy_edges_MESE)
energy_centers_MESE = np.asarray(energy_centers_MESE)
energy_edges_MESE *= 1e3 # GeV
energy_centers_MESE *= 1e3 # GeV
print(energy_edges_MESE)

"""norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
#sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
#sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.19, 0.0, 0.16, 0.0]
sigma_upper = [5.2, 5.8, 2.0, 0.93, 0.72, 0.56, 0.50, 0.38, 0.42, 0.45, 0.218, 0.19, 0.76]
sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.183, 0.187, 0.0, 0.157, 0.0]"""
norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
#sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
#sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.19, 0.0, 0.16, 0.0]
sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16, 0.0]
# physical pivot used by the paper (100 TeV). Convert to GeV.
# observed y (E^2 phi) and asymmetric errors in the same units
E0 = 1e5    # if E in GeV; 100 TeV = 1e5 GeV
y_obs = np.asarray(norm_segmented) * 1e-18 * E0**2
sigma_upper = np.asarray(sigma_upper) * 1e-18 * E0**2
sigma_lower = np.asarray(sigma_lower) * 1e-18 * E0**2

sigma1 = 2 * sigma_upper * sigma_lower / (sigma_upper + sigma_lower)    # Acts as an 'average width'
sigma2 = (sigma_upper - sigma_lower) / (sigma_upper + sigma_lower)       # Encodes the asymmetry




def initialize_evolver(M_phi=0.25, g_phi=1e-35, mntot=0.15, si=2.5, norm_=1):
    # Initialize flux evolver object with arbitrary parameters and return energies, bin edges, and delta E
    # Only called once, as these are the same for all parameters

    evolver = nuSIprop.pyprop(mphi = M_phi*1e6, # Mediator mass [eV]
                g = g_phi, # Coupling
                mntot = mntot, # Sum of neutrino masses [eV]
                si = si, # Spectral index
                norm = norm_*1e-18, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
                majorana = True, # Majorana neutrinos? [Default = True]
                non_resonant = True, # Include non s-channel contributions? Relevant for couplings g>~0.1 [Default = True]
                normal_ordering = True, # Normal neutrino mass ordering? [Default = True]
                N_bins_E = 300, # Number of energy bins, uniformly distributed in log space [Default = 300]
                lEmin = 12, # log_10 (E_min/eV) [Default = 13]
                lEmax = 17, # log_10 (E_max/eV) [Default = 17]
                zmax = 5, # Largest redshift at which sources are included [Default = 5]
                flav = 2, # Flavor of interacting neutrinos [0=e, 1=mu, 2=tau. Default = 2]
                phiphi = True # Consider double-scalar production? If set to true, the files xsec/alpha_phiphi.bin and xsec/alphatilde_phiphi.bin must exist [Default = False]
                            )
    evolver.evolve()

    energy_centers_high_resolution = evolver.get_energies()
    bin_edges_high_resolution = bin_centers_to_edges(energy_centers_high_resolution)
    delta_E_high_resolution = np.diff(bin_edges_high_resolution)    # Delta E in eV   
    
    return evolver, energy_centers_high_resolution, bin_edges_high_resolution, delta_E_high_resolution




def integrate_model_E2phi(theta, bin_edges):
    #norm, si1, si2, E_break = theta

    y_bin_avg = []
    for i in range(len(bin_edges)-1):
        Emin, Emax = bin_edges[i], bin_edges[i+1]
        Es = np.logspace(np.log10(Emin), np.log10(Emax), 200)
        if len(theta) == 2:
            norm, si = theta
            vals = SPL_flux(norm, si, energies=Es) * Es**2
        elif len(theta) == 4:
            norm, si1, si2, E_break = theta
            vals = BPL_flux(norm, si1, si2, E_break, energies=Es) * Es**2
        elif len(theta) == 3:
            norm, alpha, beta = theta
            vals = LP_flux(norm, alpha, beta, energies=Es) * Es**2

        integral = np.trapezoid(vals, Es)
        avg = integral / (Emax - Emin)   # <-- critical
        y_bin_avg.append(avg)
    return np.array(y_bin_avg)


def integrate_nuSIprop_E2phi(theta, interaction=True):
    #Mphi, g, si = theta
    #Mphi, g, mntot, si, norm = theta
    if interaction:
        Mphi, g, si, norm = theta
        evolver.set_parameters(mphi=Mphi*1e6, g=g, si=si, norm=norm*1e-18)
    else:
        si, norm = theta
        evolver.set_parameters(si=si, norm=norm*1e-18)
    #evolver.set_parameters(mphi=Mphi*1e6, g=g, mntot=mntot, si=si, norm=norm)
    
    evolver.evolve()
    flux = evolver.get_flux_fla()                      # shape (nE, flavors)
    E_samples = evolver.get_energies() / 1e9           # ensure same units as energy_edges_MESE (GeV)
    perflavor = (flux[0,:] + flux[1,:] + flux[2,:]) / 3.0  # or flux.T/columns depending on shape

    preds = []
    for Emin, Emax in zip(energy_edges_MESE[:-1], energy_edges_MESE[1:]):
        # select indices inside the bin (include endpoints carefully)
        mask = (E_samples >= Emin) & (E_samples <= Emax)
        Es = E_samples[mask]
        phis = perflavor[mask]

        if Es.size == 0:
            # fallback: interpolate phi on a fine grid and integrate, or use nearest neighbor
            preds.append(0.0)
            continue

        # integrate E^2 * phi(E) over the bin using trapezoid rule
        integrand = (Es**2) * phis
        integral = np.trapezoid(integrand, Es)   # approximates ∫ E^2 phi dE
        preds.append(integral / (Emax - Emin))   # bin-averaged E^2 phi

    return np.array(preds)


def plot_nuSIprop_E2phi():
    nuSIprop_flx_bin_averaged = integrate_nuSIprop_E2phi([5.5, 0.035, 2.85])

    evolver.set_parameters(mphi=5.5*1e6, g=0.035, si=2.85)
    evolver.evolve()
    flux = evolver.get_flux_fla()
    E = evolver.get_energies() / 1e9
    perflavor = (flux[0,:] + flux[1,:] + flux[2,:]) / 3.0

    plt.scatter(E, perflavor * E**2, label='nuSIprop, raw')
    plt.scatter(energy_centers_MESE, nuSIprop_flx_bin_averaged, label='nuSIprop, bin averaged')
    plt.errorbar(energy_centers_MESE, norm_segmented, 
                yerr=[sigma_lower/E0**2, sigma_upper/E0**2], 
                fmt='x', color='blue', label='Events', capsize=3, markersize=8, linestyle='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('$E^2 \phi$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    plt.legend()
    plt.show()



def SPL_flux(norm=2.13, si=2.548, energies=energy_centers_MESE):
    # SPL model flux
    # Normalized at E = 100TeV
    # Best fit MESE: norm = 2.13^{0.18}_{-0.17}, si = 2.548^{0.039}_{-0.041}
    #si, norm = theta
    norm = norm*1e-18
    return  norm * (energies / E0)**(-si)
    
    
def BPL_flux(norm=2.28, si1=1.72, si2=2.839, E_break=10**4.524, energies=energy_centers_MESE):
    # BPL model flux
    # Normalized at E = 100TeV = 1e5 GeV
    # Best fit MESE: norm = 2.28, si1 = 1.72, si2 = 2.839, E_break = 10**(4.524) GeV
    norm = norm*1e-18
    if E_break > 1e5:
        norm_break  = norm * (E_break / E0)**(-si1)
    else:
        norm_break  = norm * (E_break / E0)**(-si2)
    
    # If used for integration within the bin, energies will be a single value.
    if isinstance(energies, float):
        if energies < E_break:
            flux = norm_break * (energies / E0)**(-si1)
        else:
            flux = norm_break * (energies / E0)**(-si2)
    else:
        flux = np.zeros(len(energies))
        for i, E in enumerate(energies):
            if E < E_break:
                flux[i] = norm_break * (E / E_break)**(-si1)
            else:
                flux[i] = norm_break * (E / E_break)**(-si2)
    return flux 


def LP_flux(norm=2.58, alpha=2.669, beta=0.359, energies=energy_centers_MESE):
    # Log Parabola model flux
    # Normalized at E = 100TeV
    # Best fit MESE: norm = 2.42, alpha = 2.05, beta = 2.54
    norm = norm*1e-18
    return  norm * (energies / E0)**(-alpha - beta * np.log10(energies / E0)) 



def prior_transform_nuSIprop(cube):
    # --- Prior transform: maps [0,1]^3 -> (Mphi, g, si) ---
    #u_Mphi, u_g, u_mntot, u_si, u_norm = cube
    u_Mphi, u_g, u_si, u_norm = cube

    # Mphi log-uniform in [0.1, 1000]
    Mphi_min, Mphi_max = 0.03, 100   # OBS: Vid duble scalar production verkar det som att allt flux blir NAN för Mphi <~ 0.05
    #Mphi = 10**(Mphi_min + (Mphi_max - Mphi_min) * u_Mphi)
    Mphi = Mphi_min * (Mphi_max / Mphi_min) ** u_Mphi

    # g log-uniform in [1e-4, 1]
    g_min, g_max = 1e-5, 1.0
    #g = 10**(g_min + (g_max - g_min) * u_g)
    g = g_min * (g_max / g_min) ** u_g

    # mntot uniform in [0.06, 0.12]
    #mntot_min, mntot_max = 0.06, 0.16
    #mntot = mntot_min + (mntot_max - mntot_min) * u_mntot

    # si uniform in [2.0, 3.0]
    si = 2.0 + (3.5 - 2.0) * u_si

    # norm log-uniform in [1e-19, 1e-16]
    #norm_min, norm_max = 1e-19, 5*1e-17
    #norm = norm_min * (norm_max / norm_min) ** u_norm
    norm = 1 + (30.0 - 1) * u_norm

    return Mphi, g, si, norm


def prior_transform_SPL(cube):
    # --- Prior transform: maps [0,1]^2 -> (si, norm) ---
    # This prior transform also applies to nuSIprop flux without interaction (i.e g-->0)
    u_si, u_norm = cube
    
    si = 2.0 + (3.5 - 2.0) * u_si
    
    norm_min, norm_max = 0.1, 10
    norm = norm_min + (norm_max - norm_min) * u_norm
    
    return si, norm


def prior_transform_BPL(cube):
    # --- Prior transform: maps [0,1]^4 -> (norm, si1, si2, E_break) ---
    u_norm, u_si1, u_si2, u_E_break = cube
    
    norm_min, norm_max = 0.1, 10
    norm = norm_min + (norm_max - norm_min) * u_norm
    
    si1_min, si1_max = 1.0, 2.5
    si1 = si1_min + (si1_max - si1_min) * u_si1
    
    si2_min, si2_max = 2.0, 3.5
    si2 = si2_min + (si2_max - si2_min) * u_si2
    
    E_break_min, E_break_max = 1e2, 1e5
    E_break = E_break_min*(E_break_max /E_break_min) ** u_E_break
    return norm, si1, si2, E_break


def prior_transform_LP(cube):
    # --- Prior transform: maps [0,1]^3 -> (norm, alpha, beta) ---
    u_norm, u_alpha, u_beta = cube
    
    norm_min, norm_max = 0.5, 5
    norm = norm_min + (norm_max - norm_min) * u_norm
    
    alpha_min, alpha_max = 1.0, 3.5
    alpha = alpha_min + (alpha_max - alpha_min) * u_alpha
    
    beta_min, beta_max = 0.05, 1.5
    beta = beta_min + (beta_max - beta_min) * u_beta
    
    return norm, alpha, beta


def loglike_nuSIprop(theta):
    #Mphi, g, mntot, si, norm = theta
    Mphi, g, si, norm = theta
    y_pred = integrate_nuSIprop_E2phi(theta, interaction=True)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    if np.any(y_err <= 0):
        print('Zeros in y_err for parameters: Mphi=', Mphi,', g=', g, ' si=', si,', norm=', norm)
        print('y_err: ', y_err)
        print('y_pred: ', y_pred)
        #y_err[y_err <= 0] = 1e-11
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    return ll

"""
def loglike_SPL(theta, raw=True):
    # log likelihood for SPL model
    # theta = (norm, si1, si2, E_break)
    #norm, si1, si2, E_break = theta
    si, norm = theta
    if raw:
        y_pred = integrate_model_E2phi(theta, energy_edges_MESE)
    else:
        y_pred = integrate_nuSIprop_E2phi(theta, interaction=False)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    return ll


def loglike_BPL(theta):
    # log likelihood for BPL model
    # theta = (norm, si1, si2, E_break)
    #norm, si1, si2, E_break = theta
    norm, si1, si2, E_break = theta

    y_pred_bin_avg = []
    for i in range(len(energy_edges_MESE)-1):
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i+1]
        Es = np.logspace(np.log10(Emin), np.log10(Emax), 200)
        vals = BPL_flux(norm, si1, si2, E_break, energies=Es) * Es**2
        integral = np.trapezoid(vals, Es)
        avg = integral / (Emax - Emin)   # <-- critical
        y_pred_bin_avg.append(avg)
        
    y_pred = np.array(y_pred_bin_avg)
    
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2)  #- np.sum(np.log(y_err))
    return ll


def loglike_LP(theta):
    # log likelihood for LP model
    # theta = (norm, alpha, beta)
    norm, alpha, beta = theta
    y_pred = integrate_model_E2phi(theta, energy_edges_MESE)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    return ll"""


def loglike(theta):
    # Valid for all flux models
    """if interaction:
        y_pred = integrate_nuSIprop_E2phi(theta, interaction=True)
    else:"""
    y_pred = integrate_model_E2phi(theta, energy_edges_MESE)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    return ll

def create_custom_corner_plot(samples, param_names=['Mphi', 'g', 'mntot', 'si', 'norm'], title='corner_plot_custom.png'):
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
                    if j < len(param_names) and param_names[j] in ['Mphi', 'g', 'norm']:
                        ax.set_xscale('log')
                    if i < len(param_names) and param_names[i] in ['Mphi', 'g', 'norm']:
                        ax.set_yscale('log')
        
        plt.savefig(title, bbox_inches='tight')
        print("Saved custom corner_plot_custom.png with log scaling")
        plt.close(fig)
        
    except ImportError:
        print("Corner package not available, skipping custom corner plot")
    except Exception as e:
        print(f"Error creating custom corner plot: {e}")
        

evolver, energies, bin_edges_high_resolution, delta_E = initialize_evolver()
    

# --- Run UltraNest ---
if __name__ == "__main__":
   
    """
    sampler = ultranest.ReactiveNestedSampler(
        ["Mphi", "g", "mntot", "si", "norm"],
        loglike=loglike,
        transform=prior_transform_nuSIprop,
        vectorized=False,
        log_dir='ultranest_results/nuSIprop'
    )
    labels_ = [r"$\log_{10} M_\phi$", r"$\log_{10} g$", r"$\sum m_{\nu}$", r"$\gamma$", r"$log_{10} N$"]
    log_dir = 'ultranest_results/nuSIprop'
    
    
    # --- Run for BPL model without secret interactions ---
    sampler = ultranest.ReactiveNestedSampler(
        ['norm', 'si1', 'si2', 'E_break'],
        loglike_BPL,
        prior_transform_BPL,
        log_dir='ultranest_results/BPL'
    )
    # --- Run for LP model without secret interactions ---
    sampler = ultranest.ReactiveNestedSampler(
        ['norm', 'alpha', 'beta'],
        loglike_LP,
        prior_transform_LP,
        log_dir='ultranest_results/LP'
    )  """
    
    sampler = ultranest.ReactiveNestedSampler(
        ['norm', 'si1', 'si2', 'E_break'],
        loglike,
        prior_transform_BPL,
        log_dir='ultranest_results/BPL',
    )

    print('Running sampler...')
    result = sampler.run()
    

    # Print results (only from rank 0 in MPI runs)
    if MPI_AVAILABLE and MPI.COMM_WORLD.rank == 0:
        print('MPI_AVAILABLE and MPI.COMM_WORLD.rank == 0')
        
        print('Result: ', result)
        print('hej')
        
        # Print diagnostic information
        #print(f"Number of likelihood evaluations: {result['ncall']}")
        #print(f"Number of live points: {result['nlive']}")
        #print(f"Evidence estimate: {result['logz']:.3f} ± {result['logzerr']:.3f}")
        #print(f"Convergence criteria met: {result.get('converged', 'Unknown')}")
        
        sampler.print_results()
        
        """sampler.plot_corner()
        plt.savefig('corner_plot_mpi.png', dpi=300, bbox_inches='tight')
        plt.close()"""

        samples = sampler.results['samples']
        print('samples: ', samples)
        """
        fig1 = corner.corner(samples, labels=['si', 'norm'],
                           quantiles=[0.68, 0.95, 0.997],
                           show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})


        
                           
                # Transform selected axes to log10
        log_samples = samples.copy()
        log_samples[:, 0] = np.log10(samples[:, 0])   # log scale for Mphi
        log_samples[:, 1] = np.log10(samples[:, 1])   # log scale for g
        #log_samples[:, 3] = np.log10(samples[:, 3])   # log scale for E_break

        fig1 = corner.corner(
            log_samples,
            labels=labels_,
            #range=[(np.log10(0.01), np.log10(1000)),   # log range
            #    (np.log10(1e-5), np.log10(1.0)),   # log range
            #    (2.0, 3.5),                       # linear range
            #    (np.log10(1e-19), np.log10(5*1e-17))],                       # linear range
            quantiles=[0.68, 0.95, 0.997],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12}
        )
        
        
        fig1 = corner.corner(samples, labels=['norm', 'alpha', 'beta'],
                           quantiles=[0.68, 0.95, 0.997],
                           show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})"""

        #fig1.close()
        sampler.plot_run()
        sampler.plot_trace()
        sampler.plot_corner()
        
        #fig1.savefig(log_dir + "/corner.png")
        #fig2 = create_custom_corner_plot(samples, param_names=['Mphi', 'g', 'si', 'norm'], title='corner_custo_mese.png')

        #fig2.close()
        
        
        #run_plot = sampler.plot_run()
        #run_plot.savefig("run_plot.png")
        #run_plot.close()
        
        #trace_plot = sampler.plot_trace()
        #trace_plot.savefig("trace_plot.png")
        #trace_plot.close()
        
        #create_manual_run_plot(sampler, result)
        #create_manual_trace_plot(sampler, result)
        
        print('hej2')
        


    elif not MPI_AVAILABLE:
        sampler.print_results()
        print('not MPI_AVAILABLE')

