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
    #observed = df1.values.flatten()
    #expected = df2.values.flatten()
    observed = df1
    expected = df2

    # Avoid division by zero
    observed = np.where(observed == 0, 1e-6, observed)

    # Avoid division by zero in chi2 calculation
    if np.any(observed == 0):
        raise ValueError("Expected values must not contain zeros")

    # Compute chi-squared statistic
    chi2_value = np.sum((observed - expected) ** 2 / observed)

    # Calculate p-value
    p_value = chi2.sf(chi2_value, dof)
    
    return chi2_value, p_value

Edep1 = np.logspace(4, 5, num=20+1)
Edep2 = np.logspace(5, 7, num=40+1)

livetime15 = 15*365*24*3600
livetime10 = 10*365*24*3600


mc1 = pd.read_csv('~/nuSIprop/HESE_MC_events/mc_Gen1_smeared_si25_norm5.csv', index_col=0)
mc2 = pd.read_csv('~/nuSIprop/HESE_MC_events/mc_Gen2_smeared_si25_norm5.csv', index_col=0)

# Oklar normalisering?.....
#mc1['total_events'] *= 
#mc2['total_events'] *= 10   # Gen2 has 10 times more events than Gen1

print('mc1: ', mc1)
print('mc2: ', mc2)

data = pd.read_csv('HESE_data.csv', index_col=0)
print('data: ', data)

#eff1 = pd.read_csv('effective_areas/effective_areas_by_flavor_gen1.csv', index_col=0)
#eff2 = pd.read_csv('effective_areas/effective_areas_by_flavor_gen2.csv', index_col=0)
eff1 = pd.read_csv('effective_areas/eff_Gen1.csv', index_col=0)
eff2 = pd.read_csv('effective_areas/eff_Gen2.csv', index_col=0)
energies1 = np.asarray(eff1.index)
energies2 = np.asarray(eff2.index)
bin_centers1 = effective_area.bin_edges_to_centers(energies1)
bin_centers2 = effective_area.bin_edges_to_centers(energies2)
print('bin_centers1: ', bin_centers1)
print('bin_centers2: ', bin_centers2)

n_iter = 5
si_grid = np.linspace(2.0, 3.0, n_iter)  
si_marginalized_1 = []
si_marginalized_2 = []

g_phi = np.logspace(-4, 0, num=n_iter)
M_phi = np.logspace(np.log10(4*1e-1), np.log10(2*1e2), num=n_iter)

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
			  N_bins_E = 600, # Number of energy bins, uniformly distributed in log space [Default = 300]
			  lEmin = 14, # log_10 (E_min/eV) [Default = 13]
			  lEmax = 16, # log_10 (E_max/eV) [Default = 17]
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
        
        bin_centers1 = flx1_df.index.values
        bin_centers2 = flx2_df.index.values
        #print('bin_centers1: ', bin_centers1)
        #print('bin_centers2: ', bin_centers2)
        bin_edges1 = effective_area.bin_centers_to_edges(bin_centers1)        
        bin_edges2 = effective_area.bin_centers_to_edges(bin_centers2)
        delta_E1 = np.diff(bin_edges1)
        delta_E2 = np.diff(bin_edges2)
        
        flx1_df.index = flx1_df.index / 1e9    # Convert to [GeV]
        flx2_df.index = flx2_df.index / 1e9  

        #norm = 1e-4 to account for cm to m conversion
        total_events1 = effective_area.total_events(eff=flx1_df, flx=eff1, livetime=livetime15, norm=1e-4, delta_E=delta_E1)
        total_events2 = effective_area.total_events(eff=flx2_df, flx=eff2, livetime=livetime10, norm=1e-4, delta_E=delta_E2)

        # Apply energy smearing 
        total_events1['with_resolution'] = effective_area.apply_energy_smearing(energies=np.asarray(total_events1.index), 
                                                events=np.asarray(total_events1['total_events']), 
                                                resolution=0.1)
        total_events2['with_resolution'] = effective_area.apply_energy_smearing(energies=np.asarray(total_events2.index), 
                                                events=np.asarray(total_events2['total_events']), 
                                                resolution=0.1)
 

        #total_events1 = effective_area.rebinning(total_events1, Edep1)
        #total_events2 = effective_area.rebinning(total_events2, Edep2)
        counts1, edges1 = np.histogram(total_events1['with_resolution'], Edep1)
        counts2, edges2 = np.histogram(total_events2['with_resolution'], Edep2)

        print('total_events 1 binned: ', total_events1)
        print('total_events 2 binned: ', total_events2)
        print(len(total_events1), len(total_events2))

        # Compute chi2 values
        chi2_1_, p_val_1 = compute_chi2(mc1['events'], counts1)
        chi2_2_, p_val_2 = compute_chi2(mc2['events'], counts2)
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

# Save chi2 values to csv
chi2_combined_df.to_csv('chi2_combined.csv')


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

