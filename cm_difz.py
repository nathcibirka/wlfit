from __future__ import division

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from scipy import integrate, special
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(H0=100, Om0=0.27)
h = cosmo.h

clusters = np.loadtxt('clusters_z04_l60.dat').T

z_mean_cluster = np.mean(clusters[3])

z_lo = 0.23
z_cl = 0.35


####### prada c-m ########
# m_200, c_200, WMAP

c_0 = 3.681
c_1 = 5.033
alpha = 6.948
x_0 = 0.424
sig_0 = 1.047
sig_1 = 1.646
beta = 7.386
x_1 = 0.526

def c_prada(m, z):

    a = (1+z)**(-1)
    x = ((0.73/0.27)**(1/3)) * a

    d_a = (5 / 2) * ((0.73 / 0.27) ** (1 / 3)) * (np.sqrt(1 + x ** 3) / x ** (3 / 2)) * (
    integrate.quad(lambda x: x ** (3 / 2) / (1 + x ** 3) ** (3 / 2), 0, x)[0])

    y = ((10**m)/(1e12*(1/h)))**(-1)

    sig_ma = d_a*(16.9*(y**0.41)/(1+1.102*(y**0.2)+6.22*(y**0.333)))

    c_min = lambda t: c_0 + (c_1-c_0)*((1/np.pi)*np.arctan(alpha*(t-x_0))+1/2)
    sig_min = lambda w: sig_0 + (sig_1-sig_0)*((1/np.pi)*np.arctan(beta*(w-x_1))+1/2)

    B_0 = c_min(x)/c_min(1.393)
    B_1 = sig_min(x)/sig_min(1.393)

    A = 2.881
    b = 1.257
    c = 1.022
    d = 0.060

    sig_lin = B_1*sig_ma
    C = A*((sig_lin/b)**c + 1)*np.exp(d/(sig_lin**2))

    return B_0*C


##### klypin 2016 c-m ######
## parameters for 200*rho_cr, planck cosmology
z_vec = [0.0, 0.35, 0.50, 1.0, 1.44, 2.15, 2.50, 2.90, 4.10, 5.40]
Co_ve = [7.40, 6.25, 5.65, 4.30, 3.53, 2.70, 2.42, 2.20, 1.92, 1.65]
gamma_vec = [0.120, 0.117, 0.115, 0.110, 0.095, 0.085, 0.080, 0.080, 0.080, 0.080]
Mo_vec = [5.5e5, 1.0e5, 2.0e4, 900, 300, 42, 17, 8.5, 2.0, 0.3]

Cointerp = interp1d(z_vec, Co_ve, kind='cubic')
gammainterp = interp1d(z_vec, gamma_vec, kind='cubic')
Mointerp = interp1d(z_vec, Mo_vec, kind='cubic')

def coef_klypin(z):
    Co = Cointerp(z)
    gamma = gammainterp(z)
    Mo = Mointerp(z)*1e12*(1/h)
    return Co, gamma, Mo


###### Dutton e Maccio 14 m-c #######
## parameters for c_200 and m_200, NFW, planck(?)
def coef_duma(z):
    b = -0.101 + 0.026*z
    a_ = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z**1.21)
    return a_, b


m_200 = np.arange(2e14, 2.8e15, 1e13)

c_200_duffy_lo = (5.71/(1+0.23)**0.47) * (m_200/(2e12*(1/h)))**-0.084
c_200_prada_lo = c_prada(np.log10(m_200), z_lo)
c_200_klypin_lo = coef_klypin(z_lo)[0] * (((m_200)/(1e12*(1/h)))**(-coef_klypin(z_lo)[1]))*(1+((m_200)/(coef_klypin(z_lo)[2]))**0.4)
c_200_DuMa_lo = 10**(coef_duma(z_lo)[0]+ coef_duma(z_lo)[1]*np.log10((m_200)/(1e12*(1/h))))

c_200_duffy_cl = (5.71/(1+0.23)**0.47) * (m_200/(2e12*(1/h)))**-0.084
c_200_prada_cl = c_prada(np.log10(m_200), z_cl)
c_200_klypin_cl = coef_klypin(z_cl)[0] * (((m_200)/(1e12*(1/h)))**(-coef_klypin(z_cl)[1]))*(1+((m_200)/(coef_klypin(z_cl)[2]))**0.4)
c_200_DuMa_cl = 10**(coef_duma(z_cl)[0]+ coef_duma(z_cl)[1]*np.log10((m_200)/(1e12*(1/h))))

c_200_duffy_me = (5.71/(1+0.23)**0.47) * (m_200/(2e12*(1/h)))**-0.084
c_200_prada_me = c_prada(np.log10(m_200), z_mean_cluster)
c_200_klypin_me = coef_klypin(z_mean_cluster)[0] * (((m_200)/(1e12*(1/h)))**(-coef_klypin(z_mean_cluster)[1]))*(1+((m_200)/(coef_klypin(z_mean_cluster)[2]))**0.4)
c_200_DuMa_me = 10**(coef_duma(z_mean_cluster)[0]+ coef_duma(z_mean_cluster)[1]*np.log10((m_200)/(1e12*(1/h))))

# duffy_lo, = plt.plot(np.log10(m_200), c_200_duffy_lo, '--', linewidth = 3, color = 'r', label = 'Duffy et al. 2008')
# prada_lo, = plt.plot(np.log10(m_200), c_200_prada_lo, '--', linewidth = 3, color = 'g', label = 'Prada et al. 2012')
# klypin_lo, = plt.plot(np.log10(m_200), c_200_klypin_lo, '--', linewidth = 3, color = 'm', label = 'Klypin et al. 2016')
#
# duffy_cl, = plt.plot(np.log10(m_200), c_200_duffy_cl, '--', linewidth = 3, color = 'r', label = 'Duffy et al. 2008')
# prada_cl, = plt.plot(np.log10(m_200), c_200_prada_cl, '--', linewidth = 3, color = 'g', label = 'Prada et al. 2012')
# klypin_cl, = plt.plot(np.log10(m_200), c_200_klypin_cl, '--', linewidth = 3, color = 'm', label = 'Klypin et al. 2016')
#
# duffy_me, = plt.plot(np.log10(m_200), c_200_duffy_me, '--', linewidth = 3, color = 'r', label = 'Duffy et al. 2008')
# prada_me, = plt.plot(np.log10(m_200), c_200_prada_me, '--', linewidth = 3, color = 'g', label = 'Prada et al. 2012')
# klypin_me, = plt.plot(np.log10(m_200), c_200_klypin_me, '--', linewidth = 3, color = 'm', label = 'Klypin et al. 2016')


DuMa_me, = plt.plot(np.log10(m_200), c_200_DuMa_me, '--', linewidth = 3, color = 'k', label = 'c-M relation, z = 0.50')
DuMa_cl, = plt.plot(np.log10(m_200), c_200_DuMa_cl, '--', linewidth = 3, color = 'b', label = 'c-M relation, z = 0.35')
DuMa_lo, = plt.plot(np.log10(m_200), c_200_DuMa_lo, '--', linewidth = 3, color = 'r', label = 'c-M relation, z = 0.23')
me1, = plt.plot(14.851, 3.95,  'ko', markersize=10, label = 'This work, $\overline{z} = 0.50$')
plt.errorbar(14.851, 3.95, yerr=np.array([[0.62, 0.74]]).T, xerr=np.array([[0.054, 0.052]]).T, fmt='ko', markersize=10)
cl1, = plt.plot(14.99, 3.79, 'bo', markersize=10, label = 'Stacked CLASH 2016, $\overline{z} = 0.35$')
plt.errorbar(14.99, 3.79, yerr=np.array([[0.28, 0.30]]).T, xerr=np.array([[0.03, 0.03]]).T, fmt='bo', markersize=10)
lo1, = plt.plot(14.80, 3.69, 'ro', markersize=10, label = 'Stacked LoCuSS 2015, $\overline{z} = 0.23$')
plt.errorbar(14.80, 3.69, yerr=np.array([[0.24, 0.26]]).T, xerr=np.array([[0.017, 0.019]]).T, fmt='ro', markersize=10)

first_leg = plt.legend(handles=[DuMa_me, DuMa_cl, DuMa_lo], numpoints = 1,frameon=False, loc=2, prop={'size':14})
ax = plt.gca().add_artist(first_leg)
plt.legend(handles=[me1, cl1, lo1], numpoints = 1, frameon=False, loc=1)

plt.xlabel('$log_{10} M_{\odot}h^{-1}$', fontsize=20)
plt.ylabel('c', fontsize=20)
plt.tick_params(labelsize=14)
plt.xlim(14.4, 15.3)
plt.ylim(1, 8)