from __future__ import division

import numpy as np
from numpy.linalg import inv
import os
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import RectBivariateSpline as BSpline
from scipy.interpolate import UnivariateSpline as USpline
from scipy import integrate
from scipy.interpolate import interp1d
import multiprocessing
import emcee
import triangle
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

cosmo = FlatLambdaCDM(H0=100, Om0=0.27)  #Okabe 13

pix_val = 0.187  #MegaCam CFHT

path = os.getcwd()
path_cat = ('%s/ugryz' %path)  ########### change ugryz to the name of the directory containing the catalogs

stack = False  ############## change to False to analyse the clusters individually, True for a stacking
two_halo_indiv = True  ############ change to false if you don't want to include the two-halo term in the individual profiles

plots_path = ('%s/plots_indiv' %path)
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

plt.ion()
################ clusters data, dt = column names ######################
dt = np.dtype(
    [('CODEX', np.float), ('X_opt', np.float), ('Y_opt', np.float), ('bcg_spec_z', np.float), ('z_chi2', np.float),
     ('LAMBDA_chisq_opt', np.float), ('P_cen', np.float), ('Q_cen', np.float), ('X_fabrice', np.float),
     ('Y_fabrice', np.float), ('RA_opt', np.float),
     ('DEC_opt', np.float), ('ID', np.float), ('MEM_MATCH_ID', np.float), ('LAMBDA_chisq', np.float)])

clusters_data = np.loadtxt('clusters_z04_l60.dat', dtype=dt)

z_mean_cluster = np.mean(clusters_data['bcg_spec_z'])

rho_c_cl = cosmo.critical_density(z_mean_cluster)  # g / cm3
rho_c_cl_mpc = ((rho_c_cl.to('kg/Mpc3')).value)/1.9891e30 # Msun/Mpc3


if stack is False:
    results_indiv = np.zeros(len(clusters_data), dtype=np.dtype([('ID', np.float), ('M', np.float), ('err1_M', np.float),
                                                              ('err2_M', np.float), ('c', np.float), ('err1_c', np.float),
                                                              ('err2_c', np.float)]))


########## NFW model following Wright & Brainerd 2000 ######################
delta_c = lambda c: (200/3)*((c**3)/(np.log(1+c)-c/(1+c)))

def Sigma_nfw(m, c, r_mpc):

    r = np.power(((m * 3) / (800 * np.pi * rho_c_cl_mpc)), (1 / 3))
    r_s = r/c

    x = (r_mpc /r_s)
    x2 = np.power(x, 2)

    sigma = np.zeros_like(r_mpc)

    # x == 1
    fl = (x == 1)
    sigma[fl] = 2 * r_s * delta_c(c) * rho_c_cl_mpc / 3

    # x < 1
    fl = (x < 1)
    aux_arctanh = np.arctanh(np.sqrt((1 - x[fl]) / (1 + x[fl])))
    sigma[fl] = ((2 * r_s * delta_c(c) * rho_c_cl_mpc) / (x2[fl] - 1)) * (1 - (2 / (np.sqrt(1 - x2[fl]))) * aux_arctanh)

    # x > 1
    fl = (x > 1)
    aux_arctan = np.arctan2(np.sqrt(x[fl] - 1), np.sqrt(1 + x[fl]))
    sigma[fl] = ((2 * r_s * delta_c(c) * rho_c_cl_mpc) / (x2[fl] - 1)) * (1 - (2 / (np.sqrt(x2[fl] - 1))) * aux_arctan)

    return sigma

def Sigma_nfw_bar(m, c, r_mpc):
    r = np.power(((m * 3) / (800 * np.pi * rho_c_cl_mpc)), (1 / 3))
    r_s = r/c

    x = (r_mpc /r_s)
    x2 = np.power(x, 2)

    sigma_bar = np.zeros_like(r_mpc)

    fl = (x == 1)
    sigma_bar[fl] = 4 * r_s * delta_c(c) * rho_c_cl_mpc * (1 + np.log(1/2))

    # x < 1
    fl = (x < 1)
    aux_arctanh = np.arctanh(np.sqrt((1 - x[fl]) / (1 + x[fl])))
    sigma_bar[fl] = (4/x2[fl]) * r_s * delta_c(c) * rho_c_cl_mpc * ((2/(np.sqrt(1 - x2[fl]))) * aux_arctanh + np.log(x[fl]/2))

    # x > 1
    fl = (x > 1)
    aux_arctan = np.arctan2(np.sqrt(x[fl] - 1), np.sqrt(1 + x[fl]))
    sigma_bar[fl] = (4/x2[fl]) * r_s * delta_c(c) * rho_c_cl_mpc * ((2/(np.sqrt(x2[fl] - 1))) * aux_arctan + np.log(x[fl]/2))

    return sigma_bar

def Delta_Sigma_model(m, c, r_mpc, Lz):
    return (Sigma_nfw_bar(m, c, r_mpc) - Sigma_nfw(m, c, r_mpc)) + (Sigma_nfw_bar(m, c, r_mpc) - Sigma_nfw(m, c, r_mpc)) * Sigma_nfw(m, c, r_mpc) * Lz


################################ off-centered term ##################################
perfil_off_sigma = []
Profs = Table.read("Profiles_lookup_table.fits")
xi = np.logspace(-1,1,len(Profs))
radial_x_bins = np.logspace(-2,2,500)
for i in range(len(Profs)):
    perfil_off_sigma.append(list(Profs[i]))
perfil_off_sigma= np.array(perfil_off_sigma)
print("Lookup table loaded!")
Delta_Sigma_NFW_off_x = BSpline(xi,radial_x_bins,perfil_off_sigma,s=0)
del perfil_off_sigma, xi, radial_x_bins


def sigma_off(m, c, s_off, r_mpc):
    r = np.power(((m * 3) / (800 * np.pi * rho_c_cl_mpc)), (1 / 3))
    r_s = r/c
    fact = 2*r_s*delta_c(c)*rho_c_cl_mpc
    xi = s_off/r_s
    X = r_mpc/r_s

    return fact*Delta_Sigma_NFW_off_x(xi, X)[0]


############################### likelihood ################################
if stack == True:

    def lnlike(p, r_mpc, data, err, Lz):
        log10M, c, pcc, s_off, Sm = p
        m = 10**(log10M)
        k = Sm * ((Delta_Sigma_model(m, c, r_mpc, Lz))*pcc + (1 - pcc)*(sigma_off(m, c, s_off, r_mpc))) + halo_corr(r_mpc)
        dk = data - k

        chi2 = np.dot(dk, (np.dot(err, dk)))

        return -0.5*chi2

    mu_pcc = 0.78
    sig_pcc = 0.11
    ln_mu_soff = -1.13
    ln_sig_soff = 0.22
    mu_Sm = 1.016
    sig_Sm = 0.023

    def lnprior(p):
        log10M, c, pcc, s_off, Sm = p
        if 13 < log10M < 16 and\
            0.1 < c < 12 and\
            0.5 <  pcc < 1 and\
            0.1 < s_off < 0.8 and\
            0.0 < Sm < 1.09:

            return -(((pcc) - (mu_pcc))**2/(2*sig_pcc**2)) - (((np.log(s_off)) - (ln_mu_soff))**2/(2*ln_sig_soff**2)) - ((Sm - mu_Sm)**2/(2*sig_Sm**2))

        return -np.inf

else:
    def lnlike(p, r_mpc, data, err, Lz):
        log10M, c = p
        m = 10**(log10M)

        if two_halo_indiv == True:
            k = Delta_Sigma_model(m, c, r_mpc, Lz) + halo_corr(r_mpc)
        else:
            k = Delta_Sigma_model(m, c, r_mpc, Lz)

        chi2 = np.sum(((k - data) ** 2)/(err**2))

        return -0.5*chi2


    def lnprior(p):
        log10M, c = p
        if 13 < log10M < 16 and\
            0.1 < c < 12:

            return 0

        return -np.inf


def lnprob(p, r_mpc, data, err, Lz):
    if not np.isfinite(lnprior(p)):
        return -np.inf
    return lnlike(p, r_mpc, data, err, Lz)+lnprior(p)


def get_p0(min, max):
    return min + np.random.random()*(max-min)

################################# create 2-halo term ###############################
OmegaM = cosmo.Om0
OmegaL = cosmo.Ode0
OmegaK = cosmo.Ok0
H0 = (cosmo.H0).value

pi = np.pi
#c = 299792.458  #km/s
Delta_c = 200
deltac = 1.675
h = cosmo.h
G = 4.302e-9 # kms^2 Mpc /

teste = ascii.read("matterpower.dat")
k, Pk = teste['col1'], teste['col2']
mink = min(k)
maxk = max(k)

P = USpline(k,Pk,s=0,k=5)

matt_corr_table = Table.read("matt_corr.fits")
raios_matt, matt_cor = matt_corr_table['col0'], matt_corr_table['col1']
matter_correlation_norm = USpline(raios_matt, np.array(matt_cor)/0.644809, s=0, k=5)

Window = lambda x: (3/(x*x*x))*(np.sin(x)-x*np.cos(x))
rhoM = OmegaM*(rho_c_cl_mpc*1.9891e30)

H = lambda a: H0*np.sqrt(OmegaM*(a)**(-3) + OmegaK*(a)**(-2) + OmegaL)

def a(z):
    return 1/(1+z)

def D1(a):
    return (H(a))*integrate.quad(lambda x: 1/(x*H(x)/H0)**3 , 0,a)[0]

def D(a):
    return D1(a)/D1(1)

def sigma_squared(z,R):
    sigma2 = (D(a(z))*D(a(z)))*integrate.quad(lambda k: ((k*k)/(19.7392088))*P(k)*(Window(k*R))*(Window(k*R)),
                                                     0,np.inf)[0] #2*pi**2= 19.7392...
    return sigma2

def sigma8(*args):
    return np.sqrt(sigma_squared(0,8))
print(sigma8())

def nu(z, M):
    radius = (3*M/(12.566370614*rhoM))**(1/3) #4*pi = 12.56637...
    niu =deltac/np.sqrt(sigma_squared(z,radius))
    return niu

def bias(nu):
    #Delta = Delta_c(x(z))/wm(z)

    y = np.log10(Delta_c)
    """Tinker et al 2010:"""
    """now, how the hell did he find these functions?"""
    A = 1.0 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
    c = 2.4
    result = 1 - A*(nu**a)/(nu**a+deltac**a) +B*nu**b +C*nu**c

    return result

sig802 = sigma8()**2
print("sigma_8^2 = "+str(sig802))
rhoc0 = cosmo.critical_density(0)
rhoc0 = rhoc0.to('kg/Mpc3')
print("rhoc0 = " + str(rhoc0))

def Bias(z, M):
    B  = rhoc0*bias(nu(z, M))*sig802*OmegaM*D(a(z))**2
    return B

#Implement Johnston 2ht
w_johnston_val = Table.read("W_johnston.fits")
W_Johnston = USpline(w_johnston_val['col0'],w_johnston_val['col1'],s=0,k=5)
Sigma_l = lambda z, R: (1+z)*(1+z)*W_Johnston((1+z)*R)
Delta_Sigma_l = lambda z, R: 2/(R*R)*integrate.romberg(lambda x: x*Sigma_l(z,x), 0, R, vec_func=True, tol=1e-1)-Sigma_l(z,R)
Delta_Sigma_l= np.vectorize(Delta_Sigma_l)

R = np.logspace(-1,1,1000)
signal_2ht = (Delta_Sigma_l(z_mean_cluster,R)*Bias(z_mean_cluster, 6.5e44)).value ## change to the mean mass of the sample in kg

tmp_2halo = np.zeros(len(R) , dtype=np.dtype([('R', np.float), ('2halo', np.float)]))

for i in range(len(R)):
    tmp_2halo['R'] = R
    tmp_2halo['2halo'] = signal_2ht

def halo_corr(R_gal):
    Sigma = np.interp(R_gal, tmp_2halo['R'], (tmp_2halo['2halo'])/1.9891e30)
    return Sigma

###################### stacking the clusters listed in clusters_data if stack == True, otherwise analyses individually ###########################
filenames = clusters_data['CODEX'].tolist()   # use os.listdir(path_cat) to list all the files in a directory instead of using the cluster IDs from the clusters_data file

i_model = 0
for filename in filenames:

    cluster_id = int(filename)
    #print cluster_id

    lf_cat = fits.open('%s/CODEX%s_beta.cat' % (path_cat, cluster_id))[1].data

    np_argwhere = np.argwhere(clusters_data['CODEX'] == cluster_id)

    if len(np_argwhere) != 1:
        print('error')
        raise

    i_cluster = int(np_argwhere)

    Dd = cosmo.angular_diameter_distance(clusters_data[i_cluster]['bcg_spec_z']) #Mpc

    ################ Define a background source flag: ########################
    bg_flag = np.bitwise_and(lf_cat['flag_photoz'] < 4, lf_cat['weight'] > 0)
    bg_flag = np.bitwise_and(bg_flag, lf_cat['fitclass'] == 0)
    bg_flag = np.bitwise_and(bg_flag, lf_cat['MASK'] <= 1)
    bg_flag = np.bitwise_and(bg_flag, lf_cat['deadleaf'] == 0)
    bg_flag = np.bitwise_and(bg_flag, lf_cat['beta'] >= 0.0)
    bg_flag = np.bitwise_and(bg_flag, lf_cat['cprob'] == 0.0)
    #bg_flag = np.bitwise_and(bg_flag, lf_cat['beta_C2015'] >= 0.2)
    #bg_flag = np.bitwise_and(bg_flag, lf_cat['z_phot'] > (1.1*clusters_data[i_cluster]['z_chi2']+0.15))
    #bg_flag = np.bitwise_and(bg_flag, lf_cat['z_phot'] < 2)

    ########### Apply the flag: #####################
    lf_cat = lf_cat[bg_flag]

    tmp_data = np.zeros(len(lf_cat.T), dtype=np.dtype([('e1', np.float64), ('e2', np.float64), ('galaxies_x', np.float64),
         ('galaxies_y', np.float64), ('galaxies_r', np.float64), ('z_phot', np.float64),
         ('weight_LF', np.float64), ('bgprob', np.float64), ('cprob', np.float64), ('weight_del_sig', np.float64), ('e_t', np.float64), ('e_x', np.float64),
         ('del_sig', np.float64), ('del_sig_x', np.float64), ('Sigma_c', np.float64),
         ('factor_gal', np.float64), ('R_gal', np.float64), ('eta', np.float64), ('beta', np.float64)]))

    tmp_data['e1'] = lf_cat['e1']
    tmp_data['e2'] = lf_cat['e2']
    tmp_data['galaxies_x'] = lf_cat['Xpos'] - clusters_data[i_cluster]['X_opt'] #pixel
    tmp_data['galaxies_y'] = lf_cat['Ypos'] - clusters_data[i_cluster]['Y_opt'] #pixel
    tmp_data['galaxies_r'] = np.sqrt(tmp_data['galaxies_x']**2 + tmp_data['galaxies_y']**2) #pixel
    tmp_data['z_phot'] = lf_cat['z_phot']
    tmp_data['weight_LF'] = lf_cat['weight']
    tmp_data['bgprob'] = lf_cat['bgprob']
    tmp_data['cprob'] = lf_cat['cprob']

    rho_c = cosmo.critical_density(clusters_data[i_cluster]['bcg_spec_z']) # g/cm3
    rho_c_mpc = rho_c.to('kg/Mpc3') # kg/Mpc3

    tmp_data['Sigma_c'] = (const.c.to('Mpc/s')**2 / (4 * np.pi * const.G.to('Mpc3 /(kg s2)') * Dd)).value * (1/lf_cat['beta']) # kg / Mpc2
    tmp_data['factor_gal'] = (rho_c_mpc / tmp_data['Sigma_c']).value # 1/Mpc
    tmp_data['R_gal'] = (Dd * (tmp_data['galaxies_r'] * (pix_val/3600) * (np.pi/(180)))).value # Mpc
    tmp_data['eta'] = Dd * lf_cat['beta']
    tmp_data['beta'] = lf_cat['beta']

    phi = np.arctan2(tmp_data['galaxies_y'], tmp_data['galaxies_x'])
    tmp_data['e_t'] = -tmp_data['e1'] * np.cos(2*phi) - tmp_data['e2'] * np.sin(2*phi)
    tmp_data['e_x'] = tmp_data['e1'] * np.sin(2*phi) - tmp_data['e2'] * np.cos(2*phi)
    tmp_data['del_sig'] = (tmp_data['e_t'] * tmp_data['Sigma_c']) # kg
    tmp_data['del_sig_x'] = (tmp_data['e_x'] * tmp_data['Sigma_c']) # kg

    tmp_data['weight_del_sig'] = lf_cat['weight']*(tmp_data['Sigma_c']**(-2))


    ################ creates background catalogs and save to compute the covariance matrix later #######################
    if stack == True:

        if i_model == 0:
            stack_data = tmp_data
        else:
            stack_data = np.append(stack_data, tmp_data)
        i_model += 1

        tmp_data = tmp_data[0.05 <= tmp_data['R_gal']]  # apply a radial cut to make the bg catalog smaller for the bootstrap
        tmp_data = tmp_data[tmp_data['R_gal'] <= 4]

        tmp2_data = np.zeros(len(tmp_data), dtype=np.dtype(
            [('R_gal', np.float), ('del_sig', np.float), ('del_sig_x', np.float), ('weight_del_sig', np.float)]))

        tmp2_data['R_gal'] = tmp_data['R_gal']
        tmp2_data['del_sig'] = tmp_data['del_sig']
        tmp2_data['del_sig_x'] = tmp_data['del_sig_x']
        tmp2_data['weight_del_sig'] = tmp_data['weight_del_sig']

        bg_path = ('%s/bg_cats' %path)
        if not os.path.exists(bg_path):
            os.makedirs(bg_path)

        header = (len(tmp2_data.dtype.names) *'%s '%tmp2_data.dtype.names).rstrip()
        np.savetxt('%s/bg_%s.cat' % (bg_path, cluster_id), tmp2_data, header = header)

        del tmp_data

    else:

        tmp_data['Sigma_c'] = tmp_data['Sigma_c']/1.9891e30
        tmp_data['weight_del_sig'] = tmp_data['weight_del_sig'] * (1.9891e30**2)
        tmp_data['del_sig'] = tmp_data['del_sig']/1.9891e30
        tmp_data['del_sig_x'] = tmp_data['del_sig_x']/1.9891e30

        bins = np.logspace(np.log10(0.1), np.log10(2.5), num = 9)
        i_bin = np.digitize(tmp_data['R_gal'], bins, right = True)

        bin_data = np.zeros(i_bin.max() - 1, dtype=np.dtype(
            [('R_gal', np.float64), ('del_sig', np.float64), ('err_del_sig', np.float64),  ('del_sig_x', np.float64), ('err_del_sig_x', np.float64), ('Sigma_c', np.float64),
             ('z_phot', np.float64)])) # ('del_sig_cal', np.float), ('one_plus_K', np.float), ('del_sig_m', np.float) include these if there is a m correction to be applied on the shear data

        for i in range(i_bin.max()-1):   # weighted average of all quantities

            if len(tmp_data['weight_del_sig'][i_bin == i + 1]-1) == 0:
               bin_data['R_gal'][i] = 0
               bin_data['del_sig'][i] = 0
               bin_data['del_sig_x'][i] = 0
               bin_data['Sigma_c'][i] = 0.000001
               bin_data['z_phot'][i] = 0
               bin_data['err_del_sig'][i] = 0
               bin_data['err_del_sig_x'][i] = 0

            elif len(tmp_data['weight_del_sig'][i_bin == i + 1]-1) == 1:
                bin_data['R_gal'][i] = tmp_data['R_gal'][i_bin == i+1]
                bin_data['del_sig'][i] = tmp_data['del_sig'][i_bin == i+1]
                bin_data['del_sig_x'][i] = tmp_data['del_sig_x'][i_bin == i+1]
                bin_data['Sigma_c'][i] = tmp_data['Sigma_c'][i_bin == i+1]
                bin_data['z_phot'][i] = tmp_data['z_phot'][i_bin == i+1]
                bin_data['err_del_sig'][i] = np.sqrt(bin_data['del_sig'][i])
                bin_data['err_del_sig_x'][i] = np.sqrt(bin_data['del_sig_x'][i])

            else:
                bin_data['R_gal'][i] = np.average(tmp_data['R_gal'][i_bin == i+1], weights=tmp_data['weight_del_sig'][i_bin == i+1])
                bin_data['del_sig'][i] = np.average(tmp_data['del_sig'][i_bin == i+1], weights=tmp_data['weight_del_sig'][i_bin == i+1])
                bin_data['del_sig_x'][i] = np.average(tmp_data['del_sig_x'][i_bin == i+1], weights=tmp_data['weight_del_sig'][i_bin == i+1])
                bin_data['Sigma_c'][i] = np.average(tmp_data['Sigma_c'][i_bin == i+1], weights=tmp_data['weight_del_sig'][i_bin == i+1])
                bin_data['z_phot'][i] = np.average(tmp_data['z_phot'][i_bin == i+1], weights=tmp_data['weight_del_sig'][i_bin == i+1])
                bin_data['err_del_sig'][i] = (np.sqrt(len(tmp_data['weight_del_sig'][i_bin == i + 1]) * np.sum(
                    tmp_data['weight_del_sig'][i_bin == i + 1] * (
                    tmp_data['del_sig'][i_bin == i + 1] - bin_data['del_sig'][i]) ** 2) / (
                                                       (len(tmp_data['weight_del_sig'][i_bin == i + 1]) - 1) * np.sum(
                                                           tmp_data['weight_del_sig'][i_bin == i + 1])))) / (
                                              np.sqrt(len(tmp_data['weight_del_sig'][i_bin == i + 1])))
                bin_data['err_del_sig_x'][i] = np.sqrt(len(tmp_data['weight_del_sig'][i_bin == i + 1]) * np.sum(
                    tmp_data['weight_del_sig'][i_bin == i + 1] * (
                    tmp_data['del_sig_x'][i_bin == i + 1] - tmp_data['del_sig_x'][i]) ** 2) / (
                                                        (len(tmp_data['weight_del_sig'][i_bin == i + 1]) - 1) * np.sum(
                                                            tmp_data['weight_del_sig'][i_bin == i + 1]))) / np.sqrt(
                    len(tmp_data['weight_del_sig'][i_bin == i + 1]))
                #bin_data['del_sig_cal'][i] = np.sum(tmp_data['weight_del_sig'][i_bin == i + 1] *  tmp_data['del_sig'][i_bin == i + 1]) / np.sum(tmp_data['weight_del_sig'][i_bin == i + 1])
                #bin_data['one_plus_K'][i] = np.sum(tmp_data['weight_del_sig'][i_bin == i+1] * (1 + tmp_data['m'][i_bin == i+1])) / np.sum(tmp_data['weight_del_sig'][i_bin == i+1])
                #bin_data['del_sig_m'][i] = bin_data['del_sig_cal'][i]/bin_data['one_plus_K'][i]


        if bin_data['del_sig'][0] == 0:
            bin_data = bin_data[1:]

        del tmp_data


        ########### term for non-weak shear correction ###################
        L_z = np.mean((bin_data['Sigma_c'])**(-3)) / np.mean((bin_data['Sigma_c'])**(-2))

        ndim, nwalkers = 2, 32
        pool = multiprocessing.Pool()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (bin_data['R_gal'], bin_data['del_sig'], bin_data['err_del_sig'], L_z), pool= pool)
        p0 = [[get_p0(13, 16), get_p0(0.1, 12)] for i in range(nwalkers)]#, get_p0(0.1, 0.9), get_p0(0.01, 1)

        print 'Starting emcee'

        sampler.run_mcmc(p0, 10000)

        samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
        trace = sampler.chain[:, 500:, :].reshape((-1, ndim)).T

        m_mcmc, c_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))) #

        mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        print cluster_id, m_mcmc, c_mcmc

        results_indiv['ID'][i_model] = cluster_id
        results_indiv['M'][i_model] = m_mcmc[0]
        results_indiv['err1_M'][i_model] = m_mcmc[1]
        results_indiv['err2_M'][i_model] = m_mcmc[2]
        results_indiv['c'][i_model] = c_mcmc[0]
        results_indiv['err1_c'][i_model] = c_mcmc[1]
        results_indiv['err2_c'][i_model] = c_mcmc[2]

        fig1 = triangle.corner(samples, labels=["$log_{10}M_{\odot}h^{-1}$", "$c$"],
                             truths=[m_mcmc[0], c_mcmc[0]], quantiles=[0.16, 0.5, 0.84])#, levels=(1-np.exp(-0.5),))
        plt.savefig('%s/fit_%s.pdf' %(plots_path, cluster_id))
        fig1.close()

        del bin_data
        pool.close()
        pool.terminate()
        i_model += 1

if stack == False:
    np.savetxt('results_indiv.txt', results_indiv)

######################### binning the data in radial bins [Mpc] ###################################

if stack == True:
    bins = np.logspace(np.log10(0.1), np.log10(2.7), num = 9)
    i_bin = np.digitize(stack_data['R_gal'], bins, right = True)

    bin_data = np.zeros(i_bin.max() - 1, dtype=np.dtype(
        [('R_gal', np.float), ('del_sig', np.float), ('err_del_sig', np.float),  ('del_sig_x', np.float), ('err_del_sig_x', np.float), ('Sigma_c', np.float), ('factor_gal', np.float),
         ('z_phot', np.float)])) # ('del_sig_cal', np.float), ('one_plus_K', np.float), ('del_sig_m', np.float) include these if there is a m correction to be applied on the shear data

    for i in range(i_bin.max()-1):   # weighted average of all quantities
        bin_data['R_gal'][i] = np.sum(stack_data['R_gal'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        bin_data['del_sig'][i] = np.sum(stack_data['del_sig'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        bin_data['err_del_sig'][i] = (np.sqrt(len(stack_data['weight_del_sig'][i_bin == i + 1]) * np.sum(
            stack_data['weight_del_sig'][i_bin == i + 1] * (
            stack_data['del_sig'][i_bin == i + 1] - bin_data['del_sig'][i]) ** 2) / (
                                               (len(stack_data['weight_del_sig'][i_bin == i + 1]) - 1) * np.sum(
                                                   stack_data['weight_del_sig'][i_bin == i + 1])))) / (
                                      np.sqrt(len(stack_data['weight_del_sig'][i_bin == i + 1])))
        bin_data['del_sig_x'][i] = np.sum(stack_data['del_sig_x'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        bin_data['err_del_sig_x'][i] = np.sqrt(len(stack_data['weight_del_sig'][i_bin == i + 1]) * np.sum(
            stack_data['weight_del_sig'][i_bin == i + 1] * (
            stack_data['del_sig_x'][i_bin == i + 1] - stack_data['del_sig_x'][i]) ** 2) / (
                                                (len(stack_data['weight_del_sig'][i_bin == i + 1]) - 1) * np.sum(
                                                    stack_data['weight_del_sig'][i_bin == i + 1]))) / np.sqrt(
            len(stack_data['weight_del_sig'][i_bin == i + 1]))
        bin_data['Sigma_c'][i] = np.sum(stack_data['Sigma_c'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        bin_data['factor_gal'][i] = np.sum(stack_data['factor_gal'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        bin_data['z_phot'][i] = np.sum(stack_data['z_phot'][i_bin == i+1]*stack_data['weight_del_sig'][i_bin == i+1])/np.sum(stack_data['weight_del_sig'][i_bin == i+1])
        #bin_data['del_sig_cal'][i] = np.sum(stack_data['weight_del_sig'][i_bin == i + 1] *  stack_data['del_sig'][i_bin == i + 1]) / np.sum(stack_data['weight_del_sig'][i_bin == i + 1])
        #bin_data['one_plus_K'][i] = np.sum(stack_data['weight_del_sig'][i_bin == i+1] * (1 + stack_data['m'][i_bin == i+1])) / np.sum(tmp_data['weight_del_sig'][i_bin == i+1])
        #bin_data['del_sig_m'][i] = bin_data['del_sig_cal'][i]/bin_data['one_plus_K'][i]

    del stack_data

    L_z = np.mean((bin_data['Sigma_c']/1.9891e30)**(-3)) / np.mean((bin_data['Sigma_c']/1.9891e30)**(-2))

    ########################## create a covariance matrix based on a bootstrap resampling of the cluster catalogs #########

    cov_path = '%s' %bg_path

    lenstack = len(clusters_data)
    dt2 = np.dtype([('R_gal', np.float), ('del_sig', np.float),  ('del_sig_x', np.float), ('weight_del_sig', np.float)])

    bg_pop = []
    for filename in filenames:
        id = int(filename)
        dat = np.loadtxt('%s/bg_%s.cat' % (cov_path, id), dtype = dt2)
        bg_pop.append(dat)

    Delta_Sigmas =[]
    Delta_Xigmas =[]

    for i_ in range(10000): # 10000 realisation bootstrap

        realisation = np.random.choice(range(lenstack),lenstack)

        k = 0
        for j in realisation:
            if k == 0:
                stack_cl = bg_pop[j]
            else:
                stack_cl = np.append(stack_cl, bg_pop[j])
            k += 1

        i_bin = np.digitize(stack_cl['R_gal'], bins, right = False)

        tmp_bins = np.zeros(i_bin.max() - 1, dtype=np.dtype([('del_sig', np.float),  ('del_sig_x', np.float)]))

        for i in range(i_bin.max()-1):
            tmp_bins['del_sig'][i] = np.average(stack_cl['del_sig'][i_bin == i+1], weights = stack_cl['weight_del_sig'][i_bin == i+1])
            tmp_bins['del_sig_x'][i] = np.average(stack_cl['del_sig_x'][i_bin == i+1], weights = stack_cl['weight_del_sig'][i_bin == i+1])

        Delta_Sigmas.append(tmp_bins['del_sig'])
        Delta_Xigmas.append(tmp_bins['del_sig_x'])


    DS = np.mean(np.vstack(Delta_Sigmas),axis=0) #Delta sigma tangential

    DX = np.mean(np.vstack(Delta_Xigmas),axis=0) #Delta sigma crossed

    covDS = np.cov((np.vstack(Delta_Sigmas)).T)  #Delta sigma tangential bin covariance matrix

    covDX = np.cov((np.vstack(Delta_Xigmas)).T)  #Delta sigma crossed bin covariance matrix

    corDS = np.corrcoef((np.vstack(Delta_Sigmas)).T)  #Delta sigma tangential bin correlation matrix

    corDX = np.corrcoef((np.vstack(Delta_Xigmas)).T)  #Delta sigma crossed bin correlation matrix

    m_cov = covDS/(1.9891e30)**2 ### covariance matrix in Msun

    inv_m = inv(m_cov)
    inv_m = (10000-27-2)/(10000-1) * inv_m

    sn2 = np.dot(bin_data['del_sig']/1.9891e30, np.dot(inv_m, bin_data['del_sig']/1.9891e30)) # S/N following Okabe e Smith 15, eq 26



    ######################################### fitting ##################################################
    ndim, nwalkers = 5, 64
    pool = multiprocessing.Pool()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (bin_data['R_gal'], bin_data['del_sig']/1.9891e30, inv_m, L_z), pool= pool)
    p0 = [[get_p0(13, 16), get_p0(0.1, 12), get_p0(0.5, 1), get_p0(0.1, 0.8), get_p0(0.90, 1.09)] for i in range(nwalkers)]#, get_p0(0.1, 0.9), get_p0(0.01, 1)

    print 'Starting emcee'

    sampler.run_mcmc(p0, 7000)

    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    trace = sampler.chain[:, 500:, :].reshape((-1, ndim)).T

    m_mcmc, c_mcmc, pcc_mcmc, soff_mcmc, Sm_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))) #

    mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))


    axes = {'labelsize': 17}

    plt.rc('axes', **axes)

    fig1 = triangle.corner(samples, labels=["$log_{10}M_{\odot}h^{-1}$", "$c$", "$p_{cc}$", "$\sigma_{off}$", "$Sm$"],
                          truths=[m_mcmc[0], c_mcmc[0], pcc_mcmc[0], soff_mcmc[0], Sm_mcmc[0]], quantiles=[0.16, 0.5, 0.84])#, levels=(1-np.exp(-0.5),))


    ################### maximum of the 1D distributions ############################3
    sample = samples.T
    hist_m, m = np.histogram(sample[0], 50)
    m_max = m[np.argmax(hist_m)]
    hist_c, c = np.histogram(sample[1], 50)
    c_max = c[np.argmax(hist_c)]
    hist_p, p = np.histogram(sample[2], 50)
    p_max = p[np.argmax(hist_p)]
    hist_s, s = np.histogram(sample[3], 50)
    s_max = s[np.argmax(hist_s)]


    ############################### Profile plot with all the components and 200 mcmc realizations #########################
    #plt.figure(2)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    gs.update(hspace=0.1)
    ax0 = plt.subplot(gs[0])

    radii2= np.logspace(-1.2,1.2,50)
    yerro = np.sqrt(np.diagonal(m_cov))

    for log10M, c, pcc, s_off, Sm in samples[np.random.randint(len(samples), size=200)]:
        ax0.loglog(radii2, Sm*((Delta_Sigma_model(10**log10M, c, radii2, L_z))*pcc + (1 - pcc)*(sigma_off(10**log10M, c, s_off, radii2))) + halo_corr(radii2), color="k", alpha=0.05)

    ax0.set_ylabel('$\Delta\Sigma_+[M_{\odot}h^{-1}]$', fontsize=17)
    #ax0.set_xlabel('$R \\ [h^{-1} Mpc]$')
    ax0.loglog(radii2, Sm_mcmc[0]*((Delta_Sigma_model(10**m_mcmc[0], c_mcmc[0], radii2, L_z))*pcc_mcmc[0]), label = '$\Delta \Sigma_{NFW}$', color = 'blue', linewidth = 2)
    ax0.loglog(radii2, Sm_mcmc[0]*((1 - pcc_mcmc[0])*(sigma_off(10**m_mcmc[0], c_mcmc[0], soff_mcmc[0], radii2))), label = '$\Delta \Sigma^{off}_{NFW}$', color = 'purple', linewidth = 2)
    ax0.loglog(radii2, halo_corr(radii2), label = '$\Delta \Sigma_{2halo}$', color = 'green', linewidth = 2)
    ax0.loglog(radii2, (Sm_mcmc[0]*(((Delta_Sigma_model(10**m_mcmc[0], c_mcmc[0], radii2, L_z))*pcc_mcmc[0] + (1 - pcc_mcmc[0])*(sigma_off(10**m_mcmc[0], c_mcmc[0], soff_mcmc[0], radii2)))) + halo_corr(radii2)), label = '$\Delta \Sigma_{total}$', color = 'c', linewidth = 2)
    ax0.errorbar(bin_data['R_gal'], bin_data['del_sig']/1.9891e30, yerr = yerro, fmt = 'ro', ecolor = 'r')
    ax0.tick_params(axis='x',labelsize=12)
    for i in range(0, len(ax0.xaxis.get_ticklabels()), 1): ax0.xaxis.get_ticklabels()[i].set_visible(False)
    ax0.set_xlim(0.1, 3.2)
    ax0.set_ylim(1e11, 1e15)
    ax0.legend(loc=4, frameon=False, prop={'size':14})

    ax2 = plt.subplot(gs[1])
    ax2.set_xlim(0.1, 3.2)
    l = np.sqrt(np.max(np.power(ax2.get_ylim(),2)))
    ax2.set_ylim(-3, 3)
    ax2.semilogx(bin_data['R_gal'], bin_data['del_sig_x']/1.9891e44, 'ro')
    ax2.plot([0.1, 5], [0, 0], 'k--')
    ax2.errorbar(bin_data['R_gal'], bin_data['del_sig_x']/1.9891e44, yerr = bin_data['err_del_sig_x']/1.9891e44, fmt = 'ro', ecolor= 'r')
    ax2.set_ylabel('$\Delta\Sigma_{\\times}[10^{14}M_{\odot}h^{-1}]$', fontsize=17)
    ax2.set_xlabel('$R \\ [h^{-1} Mpc]$', fontsize=17)
    #ax2.yaxis.get_ticklabels()[-1].set_visible(False) sumir ultimo label do eixo
    ax2.tick_params(labelsize=12)
    for i in range(0, len(ax2.yaxis.get_ticklabels()), 2): ax2.yaxis.get_ticklabels()[i].set_visible(False)


    ################################## M-c relations ########################################
    ################## Duffy+ 08 m-c ###################
    # m_200, c_200. WMAP5
    c_duffy = 5.71 * ((10**m_mcmc[0])/(2e12*(1/h)))**(-0.084) * (1+z_mean_cluster)**(-0.47)


    ################# Prada+ 12 m-c ###################
    # m_200, c_200, WMAP
    a = (1+z_mean_cluster)**(-1)
    x = ((0.73/0.27)**(1/3)) * a

    def d_a(x):
        return (5 / 2) * ((0.73 / 0.27) ** (1 / 3)) * (np.sqrt(1 + x ** 3) / x ** (3 / 2)) * (
        integrate.quad(lambda x: x ** (3 / 2) / (1 + x ** 3) ** (3 / 2), 0, x)[0])

    c_0 = 3.681
    c_1 = 5.033
    alpha = 6.948
    x_0 = 0.424
    sig_0 = 1.047
    sig_1 = 1.646
    beta = 7.386
    x_1 = 0.526

    def c_prada(m):

        y = ((10**m)/(1e12*(1/h)))**(-1)

        sig_ma = d_a(x)*(16.9*(y**0.41)/(1+1.102*(y**0.2)+6.22*(y**0.333)))

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


    ################# klypin+ 16 m-c #################
    ## parameters for 200*rho_cr, planck cosmology
    z_vec = [0.0, 0.35, 0.50, 1.0, 1.44, 2.15, 2.50, 2.90, 4.10, 5.40]
    Co_ve = [7.40, 6.25, 5.65, 4.30, 3.53, 2.70, 2.42, 2.20, 1.92, 1.65]
    gamma_vec = [0.120, 0.117, 0.115, 0.110, 0.095, 0.085, 0.080, 0.080, 0.080, 0.080]
    Mo_vec = [5.5e5, 1.0e5, 2.0e4, 900, 300, 42, 17, 8.5, 2.0, 0.3]

    Cointerp = interp1d(z_vec, Co_ve, kind='cubic')
    gammainterp = interp1d(z_vec, gamma_vec, kind='cubic')
    Mointerp = interp1d(z_vec, Mo_vec, kind='cubic')

    Co = Cointerp(z_mean_cluster)
    gamma = gammainterp(z_mean_cluster)
    Mo = Mointerp(z_mean_cluster)*1e12*(1/h)

    c_klypin = Co * (((10**m_mcmc[0])/(1e12*(1/h)))**(-gamma))*(1+((10**m_mcmc[0])/(Mo))**0.4)


    ################# Dutton & Maccio 14 m-c ####################
    ## parameters for c_200 and m_200, NFW, planck(?)
    b = -0.101 + 0.026*z_mean_cluster
    a_ = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z_mean_cluster**1.21)

    c_DuMa = 10**(a_ + b*np.log10((10**m_mcmc[0])/(1e12*(1/h))))


############################# histograms and values of mu and sigma############################
# plt.figure(6)
# (mu_m, sigma_m) = norm.fit(sample[0])
# n_m, bins_m, patches_m = plt.hist(sample[0], 50, normed=1, facecolor='olivedrab', alpha=0.75)
# y_m = mlab.normpdf( bins_m, mu_m, sigma_m)
# l_m = plt.plot(bins_m, y_m, 'r--', linewidth=2)
# plt.xlabel('$log_{10} M_{\odot}h^{-1} $')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Mass\ Distribution:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu_m, sigma_m))
#
# plt.figure(7)
# (mu_c, sigma_c) = norm.fit(sample[1])
# n_cc, bins_c, patches_c = plt.hist(sample[1], 50, normed=1, facecolor='cadetblue', alpha=0.75)
# y_c = mlab.normpdf(bins_c, mu_c, sigma_c)
# l_c = plt.plot(bins_c, y_c, 'r--', linewidth=2)
# plt.xlabel('Concentration')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Concentration\ Distribution:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu_c, sigma_c))
#
# plt.figure(8)
# (mu_p, sigma_p) = norm.fit(sample[2])
# n_p, bins_p, patches_p = plt.hist(sample[2], 50, normed=1, facecolor='khaki', alpha=0.75)
# y_p = mlab.normpdf( bins_p, mu_p, sigma_p)
# l_p = plt.plot(bins_p, y_p, 'r--', linewidth=2)
# plt.xlabel('Off-centering distribution')
# plt.ylabel('#')
# plt.title('$p_{cc}$ distribution: $\mu=%.3f$, $\sigma=%.3f$' %(mu_p, sigma_p))
#
# plt.figure(9)
# (mu_s, sigma_s) = norm.fit(sample[3])
# n_s, bins_s, patches_s = plt.hist(sample[3], 50, normed=1, facecolor='khaki', alpha=0.75)
# y_s = mlab.normpdf( bins_s, mu_s, sigma_s)
# l_s = plt.plot(bins_s, y_s, 'r--', linewidth=2)
# plt.xlabel('Percentage')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{\sigma_{off}:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu_p, sigma_p))
#
# plt.figure(10)
# (mu_Sm, sigma_Sm) = norm.fit(sample[4])
# n_Sm, bins_Sm, patches_Sm = plt.hist(sample[4], 50, normed=1, facecolor='khaki', alpha=0.75)
# y_Sm = mlab.normpdf( bins_Sm, mu_Sm, sigma_Sm)
# l_Sm = plt.plot(bins_Sm, y_Sm, 'r--', linewidth=2)
# plt.xlabel('Percentage')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Sm:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu_Sm, sigma_Sm))
