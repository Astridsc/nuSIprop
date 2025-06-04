import numpy as np
import scipy
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import nuSIprop

from joblib import Parallel, delayed
from joblib import parallel_backend

import os
import sys
sys.path.append('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release')
import binning

#sys.path.append('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/Astrid')
from Astrid import effective_area
from Astrid import data_processing
from Astrid.data_processing import get_weights


class SuppressOutput:
    """Suppress both Python and C++ prints to stdout and stderr."""
    def __enter__(self):
        # Save original stdout and stderr file descriptors
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)

        # Open a null file descriptor (redirected to /dev/null)
        self._devnull = os.open(os.devnull, os.O_WRONLY)

        # Override stdout and stderr with the null file descriptor
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original stdout and stderr file descriptors
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)

        # Close the null file descriptor and original saved descriptors
        os.close(self._devnull)
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)

"""
def total_events(flx, eff, livetime, norm, delta_E, save_to_csv=False):
    # Interpolate `flx` to the same energy bins as `eff`
    #print('flx: ', flx)
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)
    #print('flx interpolated: ', flx_interpolated)

    total_events_df = flx_interpolated * eff * livetime * norm
    total_events_df['total_events'] = delta_E * (total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau'])
    total_events_df.index = eff.index

    if save_to_csv==True:
        total_events_df.to_csv('total_events.csv')

    return total_events_df


def apply_energy_smearing(energies, events, resolution):

    smeared_events = np.zeros_like(events)  # Initialize array for smeared events
    
    for i, E_true in enumerate(energies):
        # Gaussian width depends on resolution and energy
        logE = np.log10(energies)
        logE_true = np.log10(E_true)
        sigma_log = resolution  # Now resolution is fractional in log10(E)
        gaussian = np.exp(-0.5 * ((logE - logE_true) / sigma_log) ** 2)
        #sigma = resolution * E_true  
        #gaussian = np.exp(-0.5 * ((energies - E_true) / sigma) ** 2)

        gaussian_sum = np.sum(gaussian)
        gaussian /= gaussian_sum  # Normalize Gaussian for proper redistribution

        # Redistribute current bin's events according to the Gaussian
        smeared_events += events[i] * gaussian
        if i == 0 or i == len(energies)-1:
            print(f"Edge bin {i}: gaussian sum = {gaussian_sum}")
    
    return smeared_events


def rebinning(total_events, Edep):
    # Same procedure as used by HESE
    emin, emax = Edep[0], Edep[-1]
    nbins = len(Edep)
    width = (np.log10(emax) - np.log10(emin)) / nbins
    e_edges, _, _ = binning.get_bins(emin, emax, ewidth=width, eedge=emin)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:]))) # Oklart varför dom börjar med index 1 istället för 0

    # Group data into logarithmic bins
    total_events_binned = total_events.groupby(pd.cut(total_events.index, e_edges, include_lowest=True)).sum()
    #total_events_binned = pd.cut(total_events['total'], e_edges, include_lowest=True)
    #print('total_events_binned: ', total_events_binned)

    # Compute the midpoint (center) of each logarithmic interval
    # Geometric mean for log step midpoints
    total_events_binned['interval_center'] = [
        (interval.left * interval.right) ** 0.5 for interval in total_events_binned.index
    ]
    #total_events_binned['interval_center'] = bin_centers

    return total_events_binned
"""

def compute_chi2(df1, df2):
    """
    Compute the Chi-squared statistic between two DataFrames.
    
    Parameters:
    df1 : pandas.DataFrame
        Observed values DataFrame. 
    df2 : pandas.DataFrame
        Expected values DataFrame. Must have the same structure as df1.
    
    Returns:
    chi2_value : float
        The chi-squared value.
    p_value : float
        The p-value corresponding to the chi-squared test.
    """
    # Ensure DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape")

    # Degrees of freedom = number of rows - 1
    dof = len(df1) - 1

    # Flatten DataFrames to compare all rows/columns
    observed = df2.values.flatten()
    expected = df1.values.flatten()
    # Avoid division by zero
    expected = np.where(expected == 0, 1e-6, expected)

    # Avoid division by zero in chi2 calculation
    if np.any(expected == 0):
        raise ValueError("Expected values must not contain zeros")

    # Compute chi-squared statistic
    chi2_value = np.sum((observed - expected) ** 2 / expected)

    # Calculate p-value
    p_value = chi2.sf(chi2_value, dof)
    
    return chi2_value, p_value



mc1 = pd.read_csv('~/nuSIprop/HESE_MC_events/mc_Gen1.csv', index_col=0)
mc2 = pd.read_csv('~/nuSIprop/HESE_MC_events/mc_Gen2.csv', index_col=0)

# Oklar normalisering?.....
#mc1['total_events'] *= 
#mc2['total_events'] *= 10   # Gen2 has 10 times more events than Gen1

print('mc1: ', mc1)
print('mc2: ', mc2)

eff1 = pd.read_csv('effective_areas/effective_areas_by_flavor_gen1.csv', index_col=0)
eff2 = pd.read_csv('effective_areas/effective_areas_by_flavor_gen2.csv', index_col=0)
energies1 = np.asarray(eff1.index)
energies2 = np.asarray(eff2.index)

Edep1 = np.logspace(4, 5, num=20+1)
Edep2 = np.logspace(5, 7, num=40+1)

# Get the effective area as originally provided by HESE (1e4 to 1e7), [m2]
# Compute limited/extrapolated effective area and energy bins 
"""eff1 = effective_area.get_effective_area_dataframe(Edep1, gen2=False)
eff2 = effective_area.get_effective_area_dataframe(Edep2, gen2=True)
#eff1.to_csv('effective_areas_by_flavor_gen1.csv')
#eff2.to_csv('effective_areas_by_flavor_gen2.csv')
energies1 = np.asarray(eff1.index)
energies2 = np.asarray(eff2.index)

mc1, weights1 = get_weights(Edep1, livetime=LIVETIME3, gen2=False)
mc2, weights2 = get_weights(Edep2, livetime=LIVETIME2, gen2=True)
mc_df1 = pd.DataFrame({
        'Edep': mc1['recoDepositedEnergy'],
        'weights': np.asarray(weights1[0]),
    })
mc_df2 = pd.DataFrame({
        'Edep': mc2['recoDepositedEnergy'],
        'weights': np.asarray(weights2[0]),
    })

mc_df1 = rebinning(mc_df1, Edep_new)
mc_df2 = rebinning(mc_df2, Edep_new)
mc_df1.set_index(Edep1[1:], inplace=True)
mc_df2.set_index(Edep2[1:], inplace=True)
"""

livetime15 = 15*365*24*3600
livetime10 = 10*365*24*3600

n_iter = 7
si_grid = np.linspace(2.0, 3.0, n_iter)  
si_marginalized_1 = []
si_marginalized_2 = []

g_phi = np.logspace(-4, 0, num=n_iter)
M_min = 4*1e-1
M_max = 2*1e2
M_phi = np.logspace(np.log10(M_min), np.log10(M_max), num=n_iter)

evolver1 = nuSIprop.pyprop(mphi = M_phi[0]*1e6, # Mediator mass [eV]
			  g = g_phi[0], # Coupling
			  mntot = 0.1, # Sum of neutrino masses [eV]
			  si = 2.5, # Spectral index
			  norm = 5*1e-18, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
			  majorana = True, # Majorana neutrinos? [Default = True]
			  non_resonant = True, # Include non s-channel contributions? Relevant for couplings g>~0.1 [Default = True]
			  normal_ordering = True, # Normal neutrino mass ordering? [Default = True]
			  N_bins_E = 300, # Number of energy bins, uniformly distributed in log space [Default = 300]
			  lEmin = 13, # log_10 (E_min/eV) [Default = 13]
			  lEmax = 14, # log_10 (E_max/eV) [Default = 17]
			  zmax = 5, # Largest redshift at which sources are included [Default = 5]
			  flav = 2, # Flavor of interacting neutrinos [0=e, 1=mu, 2=tau. Default = 2]
			  phiphi = False # Consider double-scalar production? If set to true, the files xsec/alpha_phiphi.bin and xsec/alphatilde_phiphi.bin must exist [Default = False]
                          )

evolver2 = nuSIprop.pyprop(mphi = M_phi[0]*1e6, # Mediator mass [eV]
			  g = g_phi[0], # Coupling
			  mntot = 0.1, # Sum of neutrino masses [eV]
			  si = 2.5, # Spectral index
			  norm = 5*1e-18, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
			  majorana = True, # Majorana neutrinos? [Default = True]
			  non_resonant = True, # Include non s-channel contributions? Relevant for couplings g>~0.1 [Default = True]
			  normal_ordering = True, # Normal neutrino mass ordering? [Default = True]
			  N_bins_E = 300, # Number of energy bins, uniformly distributed in log space [Default = 300]
			  lEmin = 14, # log_10 (E_min/eV) [Default = 13]
			  lEmax = 17, # log_10 (E_max/eV) [Default = 17]
			  zmax = 5, # Largest redshift at which sources are included [Default = 5]
			  flav = 2, # Flavor of interacting neutrinos [0=e, 1=mu, 2=tau. Default = 2]
			  phiphi = False # Consider double-scalar production? If set to true, the files xsec/alpha_phiphi.bin and xsec/alphatilde_phiphi.bin must exist [Default = False]
                          )

chi2_1 = np.zeros(shape=(len(g_phi), len(M_phi)))
chi2_2 = np.zeros(shape=(len(g_phi), len(M_phi)))

def compute_for_params(g_, M_):
    chi2_1_si = []
    chi2_2_si = []
    for si_ in si_grid:
        print(f'g = {g_}, M = {M_}, si = {si_}')

        with SuppressOutput():
            # Checking for zmax=2 since most know sources of astrophysical neutrinos come from z<2?
            #evolver1 = nuSIprop.pyprop(mphi = M_*1e6, g = g_, mntot = 0.1, si = si_, norm = 5*1e-18, lEmin = 13, lEmax = 14, zmax=2, phiphi = False)
            evolver1.set_parameters(g=g_, mphi=M_*1e6, si=si_)
            evolver1.evolve()
            #evolver2 = nuSIprop.pyprop(mphi = M_*1e6, g = g_, mntot = 0.1, si = si_, norm = 5*1e-18, lEmin = 14, zmax=2, lEmax = 16)
            evolver2.set_parameters(g=g_, mphi=M_*1e6, si=si_)
            evolver2.evolve()

        flx1 = evolver1.get_flux_fla()
        flx2 = evolver2.get_flux_fla()         
        flx1_df = pd.DataFrame(flx1.T, index=evolver1.get_energies(), columns=['nu_e', 'nu_mu', 'nu_tau'])
        flx2_df = pd.DataFrame(flx2.T, index=evolver2.get_energies(), columns=['nu_e', 'nu_mu', 'nu_tau'])
        delta_E_1 = np.diff(flx1_df.index.values)
        delta_E_2 = np.diff(flx2_df.index.values)
        delta_E_1 = np.append(delta_E_1, delta_E_1[-1])  # Append last value to match length
        delta_E_2 = np.append(delta_E_2, delta_E_2[-1])  # Append last value to match length
        flx1_df.index = flx1_df.index / 1e9    # Convert to [GeV]
        flx2_df.index = flx2_df.index / 1e9  

        #norm = 5*1e-18  # Normalization of the free-streaming flux at 100 TeV
        total_events1 = effective_area.total_events(eff=flx1_df, flx=eff1, livetime=livetime15, norm=1, delta_E=delta_E_1)
        total_events2 = effective_area.total_events(eff=flx2_df, flx=eff2, livetime=livetime10, norm=1, delta_E=delta_E_2)

        # Apply energy smearing 
        smeared_events1 = effective_area.apply_energy_smearing(energies=np.asarray(total_events1.index), 
                                                events=np.asarray(total_events1['total_events']), 
                                                resolution=0.1)
        smeared_events2 = effective_area.apply_energy_smearing(energies=np.asarray(total_events2.index), 
                                                events=np.asarray(total_events2['total_events']), 
                                                resolution=0.1)
        total_events1['with_resolution'] = smeared_events1        
        total_events2['with_resolution'] = smeared_events2

        total_events1 = effective_area.rebinning(total_events1, Edep1)
        total_events2 = effective_area.rebinning(total_events2, Edep2)

        print('total_events 1 binned: ', total_events1)
        print('total_events 2 binned: ', total_events2)
        print(len(total_events1), len(total_events2))

        # Compute chi2 values
        chi2_1_, p_val_1 = compute_chi2(mc1['weights'], total_events1['with_resolution'])
        chi2_2_, p_val_2 = compute_chi2(mc2['weights'], total_events2['with_resolution'])
        chi2_1_si.append(chi2_1_)
        chi2_2_si.append(chi2_2_)

        #print('Chi2 1: ', chi2_1)
        #print('Chi2 2: ', chi2_2)
    # Marginalize: take minimum chi2 over si
    g_index = np.where(g_phi == g_)[0][0]
    M_index = np.where(M_phi == M_)[0][0]
    print(f'Index g: {g_index}, Index M: {M_index}')
    chi2_1[g_index][M_index] = np.min(chi2_1_si)
    chi2_2[g_index][M_index] = np.min(chi2_2_si)
    si_min_1 = si_grid[np.argmin(chi2_1_si)]
    si_min_2 = si_grid[np.argmin(chi2_2_si)]
    print(f'Minimum si for g={g_}, M={M_}: si_1 = {si_min_1}, si_2 = {si_min_2}')
    si_marginalized_1.append(si_min_1)
    si_marginalized_2.append(si_min_2)
    
    print('Chi2 1 marginalized:', chi2_1[g_index][M_index])
    print('Chi2 2 marginalized:', chi2_2[g_index][M_index])
    return np.min(chi2_1_si), np.min(chi2_2_si), si_grid[np.argmin(chi2_1_si)], si_grid[np.argmin(chi2_2_si)]

with parallel_backend('loky'):
    results = Parallel(n_jobs=1)(delayed(compute_for_params)(g_, M_) for g_ in g_phi for M_ in M_phi)
    print(results)

print(len(chi2_1), len(chi2_1[0]))
print('chi2, 1: ', chi2_1)
print('chi2, 2: ', chi2_2)
print('si marginalized 1: ', si_marginalized_1)
print('si marginalized 2: ', si_marginalized_2)

# Combine chi2 values from both generations
chi2_combined = chi2_1 + chi2_2
chi2_combined_df = pd.DataFrame(
    chi2_combined,
    index=g_phi,
    columns=M_phi
)
print('chi2 combined: ', chi2_combined_df)

# Compute the critical chi2 value for 2 sigma exclusion. Degrees of freedom for each generation
chi2_critical_1 = chi2.ppf(0.95, df=20-1)
chi2_critical_2 = chi2.ppf(0.95, df=40-1)
chi2_critical_combined = chi2.ppf(0.95, df=20+40-2)  # adjust df as appropriate
print('chi2 critical combined: ', chi2_critical_combined)


# Computing the exclusion limit for g
# where chi2_combined exceeds the critical value
exclusion_g = []
for m_idx, M in enumerate(M_phi):
    above = np.where(chi2_combined[:, m_idx] > chi2_critical_combined)[0]
    if len(above) > 0:
        exclusion_g.append(g_phi[above[0]])
        print('Exclusion g: ', g_phi[above[0]])
        print(above[0])
    else:
        exclusion_g.append(np.nan)
print('Exclusion g: ', exclusion_g)


# Plotting the exclusion limit
plt.figure(figsize=(7,5))
plt.plot(M_phi, exclusion_g, linestyle='-', marker='o', label=r'2$\sigma$ exclusion, 60-2 dof')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_\phi$')
plt.ylabel(r'$g_\phi$')
plt.title('Exclusion Limit (2$\sigma$)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()