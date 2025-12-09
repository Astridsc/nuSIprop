import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#import seaborn as sns
from scipy.stats import chi2, norm
from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy.integrate import quad
import os
import sys
sys.path.append('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release')
from Astrid.effective_area import bin_edges_to_centers, bin_centers_to_edges, apply_energy_smearing
import nuSIprop

import ultranest
import h5py
import corner
import gc



energy_bins = np.logspace(4, 7, 3*20+1)
energy_centers = bin_edges_to_centers(energy_bins)
energy_bins_low_resolution = np.logspace(4, 7, 20+1)
energy_centers_low_resolution = bin_edges_to_centers(energy_bins_low_resolution)

livetime12 = 12*365*24*3600


hese12_events_df = pd.read_csv('4_to_7_HESE12/20bins/hese12_20bins_df.csv', index_col=0)

background_df = pd.read_csv('4_to_7_HESE12/20bins/background_20bins_df.csv', index_col=0)
background_df.rename(columns={'background': 'events'}, inplace=True)


effective_area_df = pd.read_csv('4_to_7_HESE12/60bins/effective_area_4_to_7.csv', index_col=0)

energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]
energy_edges_MESE = np.asarray(energy_edges_MESE)
energy_centers_MESE = np.asarray(energy_centers_MESE)
energy_edges_MESE *= 1e3 # GeV
energy_centers_MESE *= 1e3 # GeV


norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
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
print('sigma1: ', sigma1)
print('sigma2: ', sigma2)




def log_likelihood_gaussian(data, predicted, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (data - predicted)**2 / sigma**2)


def log_likelihood_HESE(theta, evolver):

    Mphi, g, si = theta

    evolver.set_parameters(mphi=Mphi*1e6, g=g, si=si)
    evolver.evolve()
    flux = evolver.get_flux_fla()

    flx_df = pd.DataFrame(flux.T, index=energies, columns=['nu_e', 'nu_mu', 'nu_tau'])
    flx_df.index = flx_df.index / 1e9

    nuSIprop_df = total_events(flx=effective_area_df, eff=flx_df, livetime=livetime12, norm=1e-4, delta_E=delta_E)
    nuSIprop_smeared = apply_energy_smearing(energies=nuSIprop_df.index.values, events=nuSIprop_df['total_events'].values, resolution=0.1)
    nuSIprop_binned_events, _ = np.histogram(nuSIprop_df.index.values, weights=nuSIprop_smeared, bins=energy_bins_low_resolution)

    predicted = nuSIprop_binned_events + background_df['events'].values

    return log_likelihood_poisson(data=hese12_events_df['events'].values.astype(int), predicted=predicted)



def SPL_flux(norm=2.13, si=2.548, energies=energy_centers_MESE):
    # SPL model flux
    # Normalized at E = 100TeV
    # Best fit MESE: norm = 2.13^{0.18}_{-0.17}, si = 2.548^{0.039}_{-0.041}
    
    return  norm * (energies / E0)**(-si)
    
    
def BPL_flux(norm=2.28, si1=1.72, si2=2.839, E_break=10**4.524, energies=energy_centers_MESE):
    # BPL model flux
    # Normalized at E = 100TeV = 1e5 GeV
    # Best fit MESE: norm = 2.28, si1 = 1.72, si2 = 2.839, E_break = 10**(4.524) GeV
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
    return  norm * (energies / E0)**(-alpha - beta * np.log10(energies / E0)) 


def integrate_nuSIprop_E2phi(theta, evolver):
    Mphi, g, mntot, si, norm = theta
    #Mphi, g, si, norm = theta
    evolver.set_parameters(mphi=Mphi*1e6, g=g, mntot=mntot, si=si, norm=norm)
    evolver.evolve()
    flux = evolver.get_flux_fla()                      # shape (nE, flavors)
    E_samples = evolver.get_energies() / 1e9           # ensure same units as energy_edges_MESE (GeV)
    perflavor = (flux[0,:] + flux[1,:] + flux[2,:]) / 3.0  # or flux.T/columns depending on shape
    #print('perflavor: ', perflavor)

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
        integral = np.trapz(integrand, Es)   # approximates ∫ E^2 phi dE
        preds.append(integral / (Emax - Emin))   # bin-averaged E^2 phi

    return np.array(preds), E_samples, perflavor

"""def integrate_nuSIprop_E2phi(theta):
    #Mphi, g, si = theta
    Mphi, g, mntot, si, norm = theta
    evolver.set_parameters(mphi=Mphi*1e6, g=g, mntot=mntot, si=si)
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
        integral = np.trapz(integrand, Es)   # approximates ∫ E^2 phi dE
        preds.append(integral / (Emax - Emin))   # bin-averaged E^2 phi

    return np.array(preds), E_samples, perflavor"""

def integrate_model_nuSIprop_E2phi(theta, bin_edges, evolver):
    #norm, si1, si2, E_break = theta
    Mphi, g, mntot, si, norm = theta
    y_bin_avg = []
    for i in range(len(bin_edges)-1):
        Emin, Emax = bin_edges[i], bin_edges[i+1]
        Es = np.logspace(np.log10(Emin), np.log10(Emax), 200)
        
        """evolver.interp_flux_el(Es)
        evolver.interp_flux_mu(Es)
        evolver.interp_flux_ta(Es)"""
        vals = (evolver.interp_flux_el(Es) + evolver.interp_flux_mu(Es) + evolver.interp_flux_ta(Es)) * (Es)**2

        integral = np.trapz(vals, Es)
        avg = integral / (Emax - Emin)   # <-- critical
        y_bin_avg.append(avg)
    return np.array(y_bin_avg)


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

        integral = np.trapz(vals, Es)
        avg = integral / (Emax - Emin)   # <-- critical
        y_bin_avg.append(avg)
    return np.array(y_bin_avg)



def loglike_LP(theta):
    # log likelihood for LP model
    # theta = (norm, alpha, beta)
    norm, alpha, beta = theta
    y_pred = integrate_model_E2phi(theta, energy_edges_MESE)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    return ll


def loglike_BPL(theta):
    # log likelihood for BPL model
    # theta = (norm, si1, si2, E_break)
    norm, si1, si2, E_break = theta
    y_pred = model_predict_E2phi_per_bin(theta, BPL_flux, energy_edges_MESE)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2)
    return ll


def loglike_nuSIprop(theta):
    #Mphi, g, mntot, si, norm = theta
    Mphi, g, mntot, si, norm = theta
    y_pred = integrate_nuSIprop_E2phi(theta, interaction=True)
    y_err = sigma1 + sigma2 * (y_pred - y_obs)
    if np.any(y_err <= 0):
        print('Zeros in y_err for parameters: Mphi=', Mphi,', g=', g, ' mntot=', mntot, ' si=', si,', norm=', norm)
        print('y_err: ', y_err)
        print('y_pred: ', y_pred)
        y_err[y_err <= 0] = 1e-11
    ll = -0.5 * np.sum((y_obs - y_pred)**2 / y_err**2) #- np.sum(np.log(y_err))
    print('log likelihood nuSIprop: ', -0.5 * (y_obs - y_pred)**2 / y_err**2)
    return ll


def initialize_evolver(Mphi, g, mntot, si, norm, Majorana=True, Normal_ordering=True):
    evolver = nuSIprop.pyprop(mphi = Mphi*1e6, # Mediator mass [eV]
            g = g, # Coupling
            mntot = mntot, # Sum of neutrino masses [eV]
            si = si, # Spectral index
            norm = norm, # Normalization of the free-streaming flux at 100 TeV [Default = 1]
            majorana = Majorana, # Majorana neutrinos? [Default = True]
            non_resonant = True, # Include non s-channel contributions? Relevant for couplings g>~0.1 [Default = True]
            normal_ordering = Normal_ordering, # Normal neutrino mass ordering? [Default = True]
            N_bins_E = 300, # Number of energy bins, uniformly distributed in log space [Default = 300]
            lEmin = 12-0.01, # log_10 (E_min/eV) [Default = 13]
            lEmax = 16+np.sqrt(3.3), # log_10 (E_max/eV) [Default = 17]
            zmax = 5, # Largest redshift at which sources are included [Default = 5]
            flav = 2, # Flavor of interacting neutrinos [0=e, 1=mu, 2=tau. Default = 2]
            phiphi = True # Consider double-scalar production? If set to true, the files xsec/alpha_phiphi.bin and xsec/alphatilde_phiphi.bin must exist [Default = False]
                        )
    evolver.evolve()
    return evolver

theta_MNO = [5.87678306429217, 0.11338592237807571, 0.06065117287842304, 2.0022282739159496, 6.437079025724532*1e-18] # theta = Mphi, g, mntot, si, norm
evolver_MNO = initialize_evolver(Mphi=theta_MNO[0], g=theta_MNO[1], mntot=theta_MNO[2], si=theta_MNO[3], norm=theta_MNO[4], Majorana=True, Normal_ordering=True)
"""energies = evolver_MNO.get_energies()
energies = energies / 1e9
flux_MNO = evolver_MNO.get_flux_fla()
flux_MNO_tot = flux_MNO[0,:] + flux_MNO[1,:] + flux_MNO[2,:]"""
MNO_bin_avg, energies, MNO_flux = integrate_nuSIprop_E2phi(theta=theta_MNO, evolver=evolver_MNO)
print('MNO_bin_avg: ', MNO_bin_avg)
del evolver_MNO
gc.collect()
theta_DNO =  [0.44727279432446193, 0.015519735890792644, 0.059045210101645366, 2.6115632949445837, 10.20198720559154*1e-18] # theta = Mphi, g, mntot, si, norm
evolver_DNO = initialize_evolver(Mphi=theta_DNO[0], g=theta_DNO[1], mntot=theta_DNO[2], si=theta_DNO[3], norm=theta_DNO[4], Majorana=False, Normal_ordering=True)
"""flux_DNO = evolver_DNO.get_flux_fla()
flux_DNO_tot = flux_DNO[0,:] + flux_DNO[1,:] + flux_DNO[2,:]"""
DNO_bin_avg, energies, DNO_flux = integrate_nuSIprop_E2phi(theta=theta_DNO, evolver=evolver_DNO)
print('DNO_bin_avg: ', DNO_bin_avg)
del evolver_DNO
gc.collect()

theta_MIO = [6.146778500918771, 0.10363516170173048, 0.10638092522721163, 2.1226924452202147, 7.747454734748619*1e-18] # theta = Mphi, g, mntot, si, norm
evolver_MIO = initialize_evolver(Mphi=theta_MIO[0], g=theta_MIO[1], mntot=theta_MIO[2], si=theta_MIO[3], norm=theta_MIO[4], Majorana=True, Normal_ordering=False)
"""flux_MIO = evolver_MIO.get_flux_fla()
flux_MIO_tot = flux_MIO[0,:] + flux_MIO[1,:] + flux_MIO[2,:]"""
MIO_bin_avg, energies, MIO_flux = integrate_nuSIprop_E2phi(theta=theta_MIO, evolver=evolver_MIO)
print('MIO_bin_avg: ', MIO_bin_avg)
del evolver_MIO
gc.collect()
theta_DIO =  [0.8666222255525288, 0.027758096433915552, 0.10010012813679053, 2.589493436530274, 14.374786484680799*1e-18] # theta = Mphi, g, mntot, si, norm
evolver_DIO = initialize_evolver(Mphi=theta_DIO[0], g=theta_DIO[1], mntot=theta_DIO[2], si=theta_DIO[3], norm=theta_DIO[4], Majorana=False, Normal_ordering=False)
"""flux_DIO = evolver_DIO.get_flux_fla()
flux_DIO_tot = flux_DIO[0,:] + flux_DIO[1,:] + flux_DIO[2,:]"""
DIO_bin_avg, energies, DIO_flux = integrate_nuSIprop_E2phi(theta=theta_DIO, evolver=evolver_DIO)
print('DIO_bin_avg: ', DIO_bin_avg)
del evolver_DIO
gc.collect()



plt.plot(energies, MNO_flux * energies**2, color='mediumorchid', linestyle='--', alpha=1)
plt.plot(energies, DNO_flux * energies**2, color='olivedrab', linestyle='--', alpha=1)
plt.scatter(energy_centers_MESE, MNO_bin_avg, alpha=1, color='mediumorchid', label='Majorana')
plt.scatter(energy_centers_MESE, DNO_bin_avg, alpha=1, color='olivedrab', label='Dirac')

plt.errorbar(energy_centers_MESE, y_obs, 
             yerr=[sigma_lower, sigma_upper], 
             fmt='x', color='dimgrey', label='Events', capsize=3, markersize=6, linestyle='none', alpha=0.9)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('$E^2 \phi$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
plt.legend()
plt.title('Best fit models (Normal ordering)')
#plt.show()
plt.savefig('best_fit_models_NO.png')
plt.close()


plt.plot(energies, MIO_flux * energies**2, color='mediumorchid', linestyle='--', alpha=1)
plt.plot(energies, DIO_flux * energies**2, color='olivedrab', linestyle='--', alpha=1)
plt.scatter(energy_centers_MESE, MIO_bin_avg, alpha=1, color='mediumorchid', label='Majorana')
plt.scatter(energy_centers_MESE, DIO_bin_avg, alpha=1, color='olivedrab', label='Dirac')
plt.errorbar(energy_centers_MESE, y_obs, 
             yerr=[sigma_lower, sigma_upper], 
             fmt='x', color='dimgrey', label='Events', capsize=3, markersize=6, linestyle='none', alpha=0.9)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('$E^2 \phi$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
plt.legend()
plt.title('Best fit models (Inverted ordering)')
#plt.show()
plt.savefig('best_fit_models_IO.png')
plt.close()