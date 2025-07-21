import numpy as np
import pandas as pd
import nuSIprop
from scipy.stats import chi2
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release')

from Astrid import effective_area



livetime12 = 12*365*24*3600

energy_bins = np.logspace(4, 7, 60+1)
n_iter = 5
g_phi = np.logspace(-4, 0, num=n_iter)
M_phi = np.logspace(np.log10(4*1e-1), np.log10(2*1e2), num=n_iter)

si_grid = np.linspace(2.0, 3.0, n_iter)  
si_marginalized = []


evolver = nuSIprop.pyprop(mphi = M_phi[0]*1e6, # Mediator mass [eV]
			  g = g_phi[0], # Coupling
			  mntot = 0.1, # Sum of neutrino masses [eV]
			  si = 2.5, # Spectral index
			  norm = 5*1e-18, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
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



eff = pd.read_csv('eff_4_to_7.csv', index_col=0)
data = pd.read_csv('HESE12_events.csv', index_col=0)
hese12_counts, hese12_edges = np.histogram(data['energy'], energy_bins)
print(energy_bins - hese12_edges)
print('data: ', data)
print(max(data['energy']), min(data['energy']))

chi2_grid = np.zeros(shape=(len(g_phi), len(M_phi)))

for g_idx, g_phi_ in enumerate(g_phi):
    for M_idx, M_phi_ in enumerate(M_phi):
        chi2_si = []
        for si_ in si_grid:
            evolver.set_parameters(g=g_phi_, mphi=M_phi_*1e6, si=si_)
            evolver.evolve()
            flx = evolver.get_flux_fla()
            flx_df = pd.DataFrame(flx.T, index=evolver.get_energies(), columns=['nu_e', 'nu_mu', 'nu_tau'])
            bin_centers = flx_df.index.values
            bin_edges = effective_area.bin_centers_to_edges(bin_centers)
            delta_E = np.diff(bin_edges)
            
            total_events = effective_area.total_events(eff=flx_df, flx=eff, livetime=livetime12, norm=1, delta_E=delta_E)
            total_events['with_resolution'] = effective_area.apply_energy_smearing(energies=np.asarray(total_events.index), 
                                                events=np.asarray(total_events['total_events']), 
                                                resolution=0.1)
            nuSI_counts, edges = np.histogram(total_events['with_resolution'], energy_bins)
            chi2_si.append(np.sum((nuSI_counts - hese12_counts)**2 / hese12_counts))
        si_marginalized.append(si_grid[np.argmin(chi2_si)])
        chi2_grid[g_idx, M_idx] = np.min(chi2_si)
        print(f'Minimum chi2 for g={g_phi_}, M={M_phi_}: {np.min(chi2_si)}')
        print(f'Minimum si for g={g_phi_}, M={M_phi_}: {si_marginalized[-1]}')
       
       
chi2_df = pd.DataFrame(
    chi2_grid,
    index=g_phi,
    columns=M_phi
)     

chi2_critical = chi2.ppf(0.95, df=60-1)


# Computing the exclusion limit for g
# where chi2_combined exceeds the critical value
exclusion_g = []
for m_idx, M in enumerate(M_phi):
    above = np.where(chi2_grid[:, m_idx] > chi2_critical)[0]
    if len(above) > 0:
        exclusion_g.append(g_phi[above[0]])
        print('Exclusion g: ', g_phi[above[0]])
        print(above[0])
    else:
        exclusion_g.append(np.nan)
print('Exclusion g: ', exclusion_g)


# Plotting the exclusion limit
plt.figure(figsize=(7,5))
plt.plot(M_phi, exclusion_g, linestyle='-', marker='o', label=r'2$\sigma$ exclusion, 60-1 dof')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_\phi$')
plt.ylabel(r'$g_\phi$')
plt.title('Exclusion Limit (2$\sigma$)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
            


            







