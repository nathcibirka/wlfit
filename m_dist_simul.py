from __future__ import division

import numpy as np
from math import *
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.misc import derivative
from scipy.special import gamma
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import RectBivariateSpline as BSpline
from scipy.ndimage.interpolation import map_coordinates
from time import time #
from scipy.integrate import odeint
from astropy import constants as const
from astropy.io import fits
from scipy.stats import poisson, norm
from multiprocessing import Pool
from scipy.optimize import fsolve
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from numpy.linalg import inv
import multiprocessing
import emcee


cosmo = FlatLambdaCDM(H0=100, Om0=0.27)
h = cosmo.h

#################### lambda-M and M-c relations #################################
sigma_c = 0.11  #Dutton e Maccio 2014
sigma_m = 0.25  #Simet+ 2016

def lambda_M(lam):
    return (1.6e14*(lam/30)**1.33)*np.random.lognormal(mean = 0, sigma = sigma_m)

def Mc(m, z):
    b = -0.101 + 0.026*z
    a_ = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z**1.21)

    return 10**(a_ + b*np.log10(m/(1e12*(1/h))))*np.random.lognormal(mean = 0, sigma = sigma_c)

###################### NFW model functions #####################################

rho_c = lambda z: cosmo.critical_density(z)  # g / cm3
rho_c_mpc = lambda z: ((rho_c(z).to('kg/Mpc3')).value)/1.9891e30 # Msun/Mpc3

delta_c = lambda c: (200/3)*((c**3)/(np.log(1+c)-c/(1+c)))

def halo_corr(R_gal):
    Sigma = np.interp(R_gal, data2['R'], (data2['2halo'])/1.9891e30)
    return Sigma

def Sigma_nfw(m, c, r_mpc, z):

    rho = rho_c_mpc(z)
    r = np.power(((m * 3) / (800 * np.pi * rho)), (1 / 3))
    r_s = r/c

    x = (r_mpc /r_s)
    x2 = np.power(x, 2)

    sigma = np.zeros_like(r_mpc)

    # x == 1
    fl = (x == 1)
    sigma[fl] = 2 * r_s * delta_c(c) * rho / 3

    # x < 1
    fl = (x < 1)
    aux_arctanh = np.arctanh(np.sqrt((1 - x[fl]) / (1 + x[fl])))
    sigma[fl] = ((2 * r_s * delta_c(c) * rho) / (x2[fl] - 1)) * (1 - (2 / (np.sqrt(1 - x2[fl]))) * aux_arctanh)

    # x > 1
    fl = (x > 1)
    aux_arctan = np.arctan2(np.sqrt(x[fl] - 1), np.sqrt(1 + x[fl]))
    sigma[fl] = ((2 * r_s * delta_c(c) * rho) / (x2[fl] - 1)) * (1 - (2 / (np.sqrt(x2[fl] - 1))) * aux_arctan)

    return sigma

def Sigma_nfw_bar(m, c, r_mpc, z):

    rho = rho_c_mpc(z)
    r = np.power(((m * 3) / (800 * np.pi * rho)), (1 / 3))
    r_s = r/c

    x = (r_mpc /r_s)
    x2 = np.power(x, 2)

    sigma_bar = np.zeros_like(r_mpc)

    fl = (x == 1)
    sigma_bar[fl] = 4 * r_s * delta_c(c) * rho * (1 + np.log(1/2))

    # x < 1
    fl = (x < 1)
    aux_arctanh = np.arctanh(np.sqrt((1 - x[fl]) / (1 + x[fl])))
    sigma_bar[fl] = (4/x2[fl]) * r_s * delta_c(c) * rho * ((2/(np.sqrt(1 - x2[fl]))) * aux_arctanh + np.log(x[fl]/2))

    # x > 1
    fl = (x > 1)
    aux_arctan = np.arctan2(np.sqrt(x[fl] - 1), np.sqrt(1 + x[fl]))
    sigma_bar[fl] = (4/x2[fl]) * r_s * delta_c(c) * rho * ((2/(np.sqrt(x2[fl] - 1))) * aux_arctan + np.log(x[fl]/2))

    return sigma_bar


def Delta_Sigma_model(m, c, r_mpc, z, L_z):
    return (Sigma_nfw_bar(m, c, r_mpc, z) - Sigma_nfw(m, c, r_mpc, z)) + (Sigma_nfw_bar(m, c, r_mpc, z) - Sigma_nfw(m, c, r_mpc, z)) * Sigma_nfw(m, c, r_mpc, z) * L_z

def Delta_Sigma_simul(m, c, r_mpc, z):
    return (Sigma_nfw_bar(m, c, r_mpc, z) - Sigma_nfw(m, c, r_mpc, z)) #+ halo_corr(r_mpc)

m_cov = np.load('covDS_cl.npy')
m_cov = m_cov/(1.9891e30)**2

inv_m = inv(m_cov)
inv_m = (10000-27-2)/(10000-1) * inv_m

def lnlike(p, r_mpc, data, err, z):
    log10M, c = p #, pcc, s_off
    m = 10**(log10M)
    k = Delta_Sigma_simul(m, c, r_mpc, z) #+ halo_corr(r_mpc)
    dk = data - k

    chi2 = np.dot(dk, (np.dot(err, dk)))

    #chi2 = np.sum(((k - data) ** 2)/(err ** 2))   #(np.diagonal(m_cov) + np.log(2*np.pi*self.**2))  # *self.data['weight_del_sig'])

    return -0.5*chi2


def lnprior(p):
    log10M, c = p
    if 12 < log10M < 16 and\
        0.1 < c < 20:
        #0.5 <  pcc < 1 and\
        #0.1 < s_off < 0.8:

        return 0

    return -np.inf


def lnprob(p, r_mpc, data, err, z):
    if not np.isfinite(lnprior(p)):
        return -np.inf
    return lnlike(p, r_mpc, data, err, z)+lnprior(p)


def get_p0(min, max):
    return min + np.random.random()*(max-min)


#################### clusters info and shear catalogs #############################
dt = np.dtype(
    [('CODEX', np.float), ('X_opt', np.float), ('Y_opt', np.float), ('bcg_spec_z', np.float), ('z_chi2', np.float),
     ('LAMBDA_chisq_opt', np.float), ('P_cen', np.float), ('Q_cen', np.float), ('X_fabrice', np.float),
     ('Y_fabrice', np.float), ('RA_opt', np.float),
     ('DEC_opt', np.float), ('ID', np.float), ('MEM_MATCH_ID', np.float), ('LAMBDA_chisq', np.float)])

clusters_data = np.loadtxt('clusters_z04_l60.dat', dtype=dt)

z_mean_cluster = np.mean(clusters_data['bcg_spec_z'])
Dd_ref = (cosmo.angular_diameter_distance(z_mean_cluster)).value

dt2 = np.dtype([('R', np.float), ('2halo', np.float)])
data2 = np.loadtxt('2halo.cat', dtype=dt2)

clusters_path = '/home/nathalia/Desktop/codex_wl_final/betatree/ugryz'
filenames = clusters_data['CODEX'].tolist()

bins_r = np.logspace(np.log10(0.1), np.log10(2.7), num = 9)
DSs = []
M_true = []
C_true = []
M_fit = []
C_fit = []
for i in range(10):
    Del_Sigs = []
    Ms_true = []
    cs_true = []
    w =[]
    for filename in filenames:

        cluster_id = int(filename)
        np_argwhere = np.argwhere(clusters_data['CODEX'] == cluster_id)
        i_cluster = int(np_argwhere)

        Dd = (cosmo.angular_diameter_distance(clusters_data[i_cluster]['bcg_spec_z'])).value #Mpc

        lf_cat = fits.open('%s/CODEX%s_beta.cat' % (clusters_path,cluster_id))[1].data
        weight = np.sum(lf_cat['beta'] * lf_cat['weight'])/np.sum(lf_cat['weight'])
        m_true = lambda_M(clusters_data[i_cluster]['LAMBDA_chisq'])
        c_true = Mc(m_true, clusters_data[i_cluster]['bcg_spec_z'])

        bins = ([((bins_r[i] + bins_r[i+1])/2) * (Dd/Dd_ref) for i in range(8)])
        DS = ([Delta_Sigma_simul(m_true, c_true, r, clusters_data[i_cluster]['bcg_spec_z']) for r in bins])

        Del_Sigs.append(DS)
        Ms_true.append(m_true)
        cs_true.append(c_true)
        w.append(weight)

    mean_M = np.average(Ms_true, weights=w)
    mean_c = np.average(cs_true)
    mean_DS = np.average(Del_Sigs, axis = 0, weights=w)

    M_true.append(mean_M)
    C_true.append(mean_c)
    DSs.append(mean_DS)

    R_bin = ([((bins_r[i] + bins_r[i+1])/2) for i in range(8)])

    ndim, nwalkers = 2, 128
    pool = multiprocessing.Pool()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (R_bin, mean_DS, inv_m, z_mean_cluster), pool= pool)
    p0 = [[get_p0(12, 16), get_p0(0.1, 20)] for i in range(nwalkers)]#, get_p0(0.1, 0.9), get_p0(0.01, 1)

    print 'Starting emcee'

    sampler.run_mcmc(p0, 5000)

    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    trace = sampler.chain[:, 500:, :].reshape((-1, ndim)).T

    m_mcmc, c_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))) #

    mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print m_mcmc, c_mcmc

    M_fit.append(m_mcmc)
    C_fit.append(c_mcmc)

    del Del_Sigs
    del Ms_true
    del cs_true
    del w

    pool.close()

np.savez('results_simul', DSs = DSs, M_true = M_true, C_true = C_true, M_fit = M_fit, C_fit = C_fit )




