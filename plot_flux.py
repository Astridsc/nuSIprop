import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nuSIprop  
import scipy
from scipy import signal
from scipy import stats


def plot_fig2(df):
    n = int(len(df.index))
    norm = 1e28

    plt.plot(df.index[:n]/1e9, 
             (df['nu_e'].iloc[:n] + df['nu_mu'].iloc[:n] + df['nu_tau'].iloc[:n])*df.index[:n]**2/norm, 
             label=r'$\sum m_{\nu} = 0.3 eV$' + ', NO')
    
    plt.xscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel(r'$E_{\nu}^2 \times d\phi/dE_{\nu}$')

    plt.legend()
    plt.show()

#df = pd.read_csv('flux_results.csv', index_col=0)
#plot_fig2(df)



def plot_fig3a(df):
    n = int(len(df.index)/2)
    norm = 1e28
    plt.plot(df.index[:n]/1e9, df['nu_e'].iloc[:n], label=r'$\nu_e$')

    """plt.plot(df.index[:n]/1e9, df['nu_e'].iloc[:n]*df.index[:n]**2/norm, label=r'$\nu_e$')
    plt.plot(df.index[:n]/1e9, df['nu_mu'].iloc[:n]*df.index[:n]**2/norm, label=r'$\nu_\mu$')
    plt.plot(df.index[:n]/1e9, df['nu_tau'].iloc[:n]*df.index[:n]**2/norm, label=r'$\nu_\tau$')"""

    #plt.plot(df.index[:n]/1e9, df.index[:n]*df['nu_e'].iloc[:n]/np.log10(df.index[:n])/2.3/norm, label=r'$\nu_e$')
    #plt.plot(df.index[:n]/1e9, df.index[:n]*df['nu_mu'].iloc[:n]/np.log10(df.index[:n])/2.3/norm, label=r'$\nu_\mu$')
    #plt.plot(df.index[:n]/1e9, df.index[:n]*df['nu_tau'].iloc[:n]/np.log10(df.index[:n])/2.3/norm, label=r'$\nu_\tau$')

    plt.xscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel(r'$E_{\nu}^2 \times d\phi/dE_{\nu}$')

    plt.legend()
    plt.show()

#df = pd.read_csv('flux_results.csv', index_col=0)
#plot_fig3a(df)

def plot_fig3b(df_majorana, df_dirac):
    n = int(len(df_majorana.index)/2)
    norm = 1e28

    plt.plot(df_majorana.index[:n]/1e9, 
             (df_majorana['nu_e'].iloc[:n] + df_majorana['nu_mu'].iloc[:n] + df_majorana['nu_tau'].iloc[:n])*df_majorana.index[:n]**2/norm, 
             label=r'Majorana')
    plt.plot(df_dirac.index[:n]/1e9, 
             (df_dirac['nu_e'].iloc[:n] + df_dirac['nu_mu'].iloc[:n] + df_dirac['nu_tau'].iloc[:n])*df_dirac.index[:n]**2/norm, 
             label=r'Dirac')

    plt.xscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel(r'$E_{\nu}^2 \times d\phi/dE_{\nu}$')
    plt.title(r'Flux at $z_{max}=2.5$ for $g=0.1$ and $M_{\phi} = 5 MeV$')

    plt.legend()
    plt.show()

df_majorana = pd.read_csv('flux/flux_majorana_z_2_5.csv', index_col=0)
df_dirac = pd.read_csv('flux/flux_dirac_z_2_5.csv', index_col=0)
#plot_fig3b(df_majorana, df_dirac)


def plot_fig4():
    m_phi = 1e6
    g = 0.1
    gamma = g**2 * m_phi / 16 / np.pi
    #s1 = np.linspace(1e-2, 1-0.001, 100)
    #s2 = np.linspace(1+0.001, 4-0.001, 50)
    #s = (s1 + s2)*m_phi**2
    #s1 = np.linspace(1e-2, 1-0.1, 50)*m_phi**2
    #s2 = np.linspace(1+0.1, 1e2, 50)*m_phi**2
    s1 = np.logspace(-2, 0) * m_phi**2
    s2 = np.logspace(0, 2) * m_phi**2
    s1, s2 = s1[:-1], s2[1:]

    log1 = np.log10(m_phi**2 / (m_phi**2 + s1))
    log2 = np.log10(m_phi**2 / (m_phi**2 + s2))
    sigma_s1 = s1 / ((s1-m_phi**2)**2 + gamma**2 * m_phi**2)
    sigma_s2 = s2 / ((s2-m_phi**2)**2 + gamma**2 * m_phi**2)

    tot1_1 = 2*((s1+2*m_phi**2)/s1/(s1+m_phi**2))
    tot1_2 = 4*(m_phi**2/s1**2)*log1
    tot2 = -2*(m_phi**2/s1 -1)*((s1 + m_phi**2 * log1)/ ((s1-m_phi**2)**2 + m_phi**2 * gamma**2))
    tot3 = (1/s1 + (2*m_phi**2 *(m_phi**2 + s1)/s1**2 /(2*m_phi**2 +s1))*log1)
    plt.plot(s1/m_phi**2, g**4 * m_phi**2 * tot1_1 / 16 / np.pi, label='tot1.1', color ='black')
    plt.plot(s1/m_phi**2, g**4 * m_phi**2 * tot1_2 / 16 / np.pi, label='tot1.2', color ='purple')
    plt.plot(s1/m_phi**2, g**4 * m_phi**2 * tot2 / 16 / np.pi, label='tot2', color='yellow')
    plt.plot(s1/m_phi**2, g**4 * m_phi**2 * tot3 / 16 / np.pi, label='tot3', color='green')

    sigma_tot1 = sigma_s1 + 2*((s1+2*m_phi**2)/s1/(s1+m_phi**2) + 2*(m_phi**2/s1**2)*log1) -2*(m_phi**2/s1 -1)*((s1 + m_phi**2 * log1)/ ((s1-m_phi**2)**2 + m_phi**2 * gamma**2)) + (1/s1 + (2*m_phi**2 *(m_phi**2 + s1)/s1**2 /(2*m_phi**2 +s1))*log1)
    sigma_tot2 = sigma_s2 + 2*((s2+2*m_phi**2)/s2/(s2+m_phi**2) + 2*(m_phi**2/s2**2)*log2) -2*(m_phi**2/s2 -1)*((s2 + m_phi**2 * log2)/ ((s2-m_phi**2)**2 + m_phi**2 * gamma**2)) + (1/s2 + (2*m_phi**2 *(m_phi**2 + s2)/s2**2 /(2*m_phi**2 +s2))*log2)

    for i, s in enumerate(s2):
        if s/m_phi**2 > 4.001:
            sigma_phiphi = ((s**2-4*m_phi**2 *s + 6*m_phi**4)/(s-2*m_phi**2) *  np.log10(((np.sqrt(s*(s-4*m_phi**2)) +s -2*m_phi**2)/(np.sqrt(s*(s-4*m_phi**2)) -s +2*m_phi**2))**2) - 6*np.sqrt(s*(s-4*m_phi**2)))/2/s**2
        else:
            sigma_phiphi = 0
        sigma_tot2[i] += sigma_phiphi

    #plt.plot(s1/m_phi**2, g**4 * sigma_tot1 * m_phi**2 /16 / np.pi, c='r', label=r'$\sigma_{tot}$')
    #plt.plot(s2/m_phi**2, g**4 * sigma_tot2 * m_phi**2 /16 / np.pi, c='r')

    plt.plot(s1/m_phi**2, g**4 * m_phi**2 * sigma_s1 /16 / np.pi, c='b', label=r'$\sigma_{s}$')
    plt.plot(s2/m_phi**2, g**4 * m_phi**2 * sigma_s2 /16 / np.pi, c='b')
    #plt.plot(s2/m_phi**2, g**4 * sigma_s2 * m_phi**2 /16 / np.pi, c='b')

    plt.plot(s1/m_phi**2, g**4 * sigma_tot1 * m_phi**2 /16 / np.pi, c='r', label=r'$\sigma_{tot}$')
    plt.plot(s2/m_phi**2, g**4 * sigma_tot2 * m_phi**2 /16 / np.pi, c='r')

    #plt.plot(s1/m_phi**2, 2*((s1+2*m_phi**2)/(s1**2+s1*m_phi**2) + 2*(m_phi/s1)**2 *log1), c='b')
    #plt.plot(s1/m_phi**2, -2*(m_phi**2/s1 -1)*((s1 + m_phi**2 * log1)/ ((s1-m_phi**2)**2 + m_phi**2 * gamma**2)), c='r')
    #plt.plot(s1/m_phi**2, (1/s1 + (2*m_phi**2 *(m_phi**2 + s1)/s1**2 /(2*m_phi**2 +s1))*log1), c='g')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$s/M_{\phi}^2 = 2E_{\nu}m_{\nu}/M_{\phi}^2$')
    plt.ylabel(r'$\sigma \times M_{\phi}^2$')
    #plt.ylabel(r'$\sigma_{tot}/\sigma_s$')
    plt.legend()
    #plt.savefig('fig4.png')
    plt.show()

#plot_fig4()


def plot_fig5(df_full, df_s_channel):
    n = int(len(df_full.index)/2)
    norm = 1e28

    plt.plot(df_full.index[:n]/1e9, 
             (df_full['nu_e'].iloc[:n] + df_full['nu_mu'].iloc[:n] + df_full['nu_tau'].iloc[:n])*df_full.index[:n]**2/norm, 
             label='g = 0.1, Full computation')
    plt.plot(df_s_channel.index[:n]/1e9, 
             (df_s_channel['nu_e'].iloc[:n] + df_s_channel['nu_mu'].iloc[:n] + df_s_channel['nu_tau'].iloc[:n])*df_s_channel.index[:n]**2/norm, 
             label='g = 0.1, s-channel')
    delta = signal.unit_impulse(df_s_channel.index)

    #plt.plot(df_s_channel.index[:n]/1e9, stats.chauchy())
    
    plt.xscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel(r'$E_{\nu}^2 \times d\phi/dE_{\nu}$')

    plt.legend()
    plt.show()

#df_full = pd.read_csv('flux_results.csv', index_col=0)
#df_s_channel = pd.read_csv('flux_s_channel.csv', index_col=0)
#plot_fig5(df_full, df_s_channel)


def plot_fig8(df):
    flux = df['nu_e'] + df['nu_mu'] + df['nu_tau']
    print('flux, ', flux)
    DA = 10 # To get total nof events for Gen2, multiply by 10 for effective area
    Dt = 10*365*24*3600
    N = flux * Dt * DA
    print('N ', len(N), type(N), N)
    hist = N.hist(bins=30)

    """#plt.plot(df.index/1e9, flux, label = r'$\nu SI$', c='r')
    plt.hist(x=df.index,
             bins=30,
             weigths=N)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel('Number of events')
    plt.show()"""

    hist, bins = np.histogram(N, bins=30)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(N, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#df_Gen2 = pd.read_csv('flux_Fig8.csv', index_col=0)
#print(df_Gen2.index)
#plot_fig8(df_Gen2)


def plot_phiphi_diff(df_phiphi, df_no_phiphi):
    norm = 1e28
    phiphi = df_phiphi['nu_e'] + df_phiphi['nu_mu'] + df_phiphi['nu_tau']
    no_phiphi = df_no_phiphi['nu_e'] + df_no_phiphi['nu_mu'] + df_no_phiphi['nu_tau']

    plt.plot(df_phiphi.index/1e9, 
             (phiphi - no_phiphi)*df_phiphi.index**2/norm, 
             label='g = 0.01, difference')
    
    plt.xscale('log')
    plt.xlabel(r'$E_{\nu} [GeV]$')
    plt.ylabel(r'$E_{\nu}^2 \times d\phi/dE_{\nu}$')

    plt.legend()
    plt.show()

#df_phiphi = pd.read_csv('flux_results_with_phiphi.csv', index_col=0)
#df_no_phiphi = pd.read_csv('flux_results_no_phiphi.csv', index_col=0)
#plot_phiphi_diff(df_phiphi, df_no_phiphi)


if __name__ == "__main__":
    plot_fig3b(df_majorana, df_dirac)


