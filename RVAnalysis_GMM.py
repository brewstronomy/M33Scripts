#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import os.path as path, numpy as np, matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import M33.constant as con
from customplot.GridManipulation import add_grid_minor
from M33.utils import deprojected_XYR
from scipy.stats import norm

from sklearn import mixture

# --------- #
# functions #
# --------- #
def make_mask(lo, hi, arr, finite=True):
    mlo = (np.array(arr) >= lo)
    mhi = (np.array(arr) < hi)
    if finite:
        mfi = np.isfinite(np.array(arr))
    else:
        mfi = np.full(arr.size, True)
    return mlo & mhi & mfi

def colorfunc(arr, cmap=plt.cm.viridis):
    return np.array([tuple(x) for x in cmap((arr - arr.min()) / arr.ptp())])

def colorfrom(colarr, arr):
    return tuple(colarr[np.argmin(arr)])

def gauss(x, *pars):
    x = np.array(x)
    amp, mu, sig, off = pars
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2) + off

# -------------- #
# setup & import #
# -------------- #
plot_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
quad_str = ['NorthEast', 'SouthEast', 'SouthWest', 'NorthWest']

full_path = '/Users/luca/Documents/Research/M33/'

# synthesized catalog
table_path = full_path + 'Analysis/Tables/'
table = ascii.read(path.join(table_path, 'M33_AllRVtargets_HIadd_highthreshold.txt'), fill_values=[('99999.9', np.nan)])

OBJID, RA, DEC, RV, eRV, SNR, Hmag, SanR, Est, Age, eAge, MH, eMH, FeH, eFeH, alpha, ealpha,\
AID, ARV1, ARV2, ARV3, ARdis, APA, AVHI, Source, relPA, Xdeproj, Ydeproj, Rdeproj, VHI, eVHI = [table[k] for k in table.keys()]

colors = colorfunc(MH)

# HI Data -- spectral cube & velocity field
hi_path = full_path + 'Data/Keenan+2016/'
mom1, head_mom1 = fits.getdata(path.join(hi_path, 'M33_HI_mom1.fits'), header=True)
cube, head_cube = fits.getdata(path.join(hi_path, 'M33_Arecibo_HI_edited.fits'), header=True)
wcs_mom1 = WCS(head_mom1)
wcs_cube = WCS(head_cube)

# # # compute HI -- should only run once (~6.3k points)
RelPA_hi, Rdis_hi = [], []
VHI_hi = np.array([mom1.T[ira, idec] for ira in range(mom1.T.shape[0]) for idec in range(mom1.T.shape[1])])  # unravel by correct RA & DEC
rr = [wcs_mom1.pixel_to_world_values([[ira, idec]])[0][0] for ira in range(mom1.T.shape[0]) for idec in range(mom1.T.shape[1])]
dd = [wcs_mom1.pixel_to_world_values([[ira, idec]])[0][1] for ira in range(mom1.T.shape[0]) for idec in range(mom1.T.shape[1])]
m_vhi = np.isfinite(VHI_hi)  # most mom1 map is masked out
VHI_hi, rr, dd = VHI_hi[m_vhi], np.array(rr)[m_vhi], np.array(dd)[m_vhi]

skycoord_hi = SkyCoord(rr, dd, unit='deg')
Xdeproj_hi, Ydeproj_hi, Rdeproj_hi = [arr.value for arr in deprojected_XYR(skycoord_hi)]
m_vhi2 = Rdeproj_hi < 50  # kpc
VHI_hi, rr, dd = VHI_hi[m_vhi2], np.array(rr)[m_vhi2], np.array(dd)[m_vhi2]

skycoord_hi = SkyCoord(rr, dd, unit='deg')
RelPA_hi = con.coord_m33.position_angle(skycoord_hi).to('deg').deg
PA_hi = RelPA_hi + con.PA_m33.deg

# masks -- "quadrant" defined in galaxy plane rotated such that major axis is aligned with North
mk1 = make_mask(0, 90, relPA)
mk2 = make_mask(90, 180, relPA)
mk3 = make_mask(180, 270, relPA)
mk4 = make_mask(270, 360, relPA)

# radial bins
bins = 10 ** (np.histogram(np.log10(Rdeproj), bins='auto')[1])  # kpc; binned in log-space
centers = (bins[1:] + bins[:-1]) / 2  # kpc

circ_area = np.pi * centers ** 2  # kpc^2; assuming a circularized disk

r1, rv1, erv1 = Rdeproj[mk1], RV[mk1], eRV[mk1]
r2, rv2, erv2 = Rdeproj[mk2], RV[mk2], eRV[mk2]
r3, rv3, erv3 = Rdeproj[mk3], RV[mk3], eRV[mk3]
r4, rv4, erv4 = Rdeproj[mk4], RV[mk4], eRV[mk4]

# spectral bins
bins_rv = np.arange(-400, 100, 28.866)
rvcenters = (bins_rv[1:] + bins_rv[:-1]) / 2
rv_range = np.linspace(-400, 100, 1000)

cts_all, bins_rv = np.histogram(RV, bins=bins_rv)  #bins_rv)
dens_all, _ = np.histogram(RV[RV<100], density=True, bins=bins_rv)

cts1, _ = np.histogram(rv1, bins=bins_rv)
cts2, _ = np.histogram(rv2, bins=bins_rv)
cts3, _ = np.histogram(rv3, bins=bins_rv)
cts4, _ = np.histogram(rv4, bins=bins_rv)

vts1, _ = np.histogram(VHI[mk1], bins=bins_rv)
vts2, _ = np.histogram(VHI[mk2], bins=bins_rv)
vts3, _ = np.histogram(VHI[mk3], bins=bins_rv)
vts4, _ = np.histogram(VHI[mk4], bins=bins_rv)

vhi1 = VHI_hi[make_mask(0, 90, RelPA_hi)]
vhi2 = VHI_hi[make_mask(90, 180, RelPA_hi)]
vhi3 = VHI_hi[make_mask(180, 270, RelPA_hi)]
vhi4 = VHI_hi[make_mask(270, 360, RelPA_hi)]

hicts_all, _ = np.histogram(VHI_hi, bins=bins_rv)
hicts1, _ = np.histogram(vhi1, bins=bins_rv)
hicts2, _ = np.histogram(vhi2, bins=bins_rv)
hicts3, _ = np.histogram(vhi3, bins=bins_rv)
hicts4, _ = np.histogram(vhi4, bins=bins_rv)
hi_to_sc = (cts_all * rvcenters).sum() / (hicts_all * rvcenters).sum()  # for scaling HI distribution to match cluster distribution

# --- #
# GMM #
# --- #
rv = RV[RV < 100].reshape(-1, 1)  # recasting to a column vector
rvctr = rvcenters.reshape(-1, 1)
rvrange = np.linspace(-400, 100, 500)

IC_fig, IC_ax = plt.subplots()
IC_ax2 = IC_ax.twinx()

### 'full'
Ncomp = range(1, 6)
models = [mixture.GaussianMixture(n_components=n, covariance_type='full').fit(rv) for n in Ncomp]
aic = [m.aic(rv) for m in models]
bic = [m.bic(rv) for m in models]
pdf = [np.exp(m.score_samples(rvrange.reshape(-1, 1))) for m in models]  # weighted log(probability) per data point (RV)
resp = [m.predict_proba(rvrange.reshape(-1, 1)) for m in models]  # predicted posterior probability for each component -- P(c | RV)
ipdf = [r * p[:, np.newaxis] for r, p in zip(resp, pdf)]
means = [models[n-1].means_[:, f-1] for f in range(rv.shape[1]) for n in Ncomp]
sigmas = [np.sqrt(models[n-1].covariances_[:, f-1, f-1]) for f in range(rv.shape[1]) for n in Ncomp]
order = [np.argsort(m) for m in means]

IC_ax2.plot(Ncomp, aic, linestyle='dashed', linewidth=2, color=plot_cols[0], label='AIC, full')
IC_ax.plot(Ncomp, np.exp(0.5 * (min(aic) - aic)), linestyle='solid', linewidth=2, color=plot_cols[0], label='AIC, full')
IC_ax2.plot(Ncomp, bic, linestyle='dashed', linewidth=2, color=plot_cols[1], label='BIC, full')
IC_ax.plot(Ncomp, np.exp(0.5 * (min(bic) - bic)), linestyle='solid', linewidth=2, color=plot_cols[1], label='BIC, full')
#
# plot
#
fig, axes = plt.subplots(nrows=2, ncols=Ncomp[-1], sharex=True, figsize=(15, 15))
for n in Ncomp:
#    axes[0][n-1].fill_between(rvcenters, y1=0, y2=dens_all, step='mid', facecolor=plot_cols[-1], alpha=0.3)
#    axes[0][n-1].step(rvcenters, dens_all, where='mid', color='black', linewidth=2)
    axes[0][n-1].fill(rvrange, (norm.pdf(rvrange, rv, 15) / rv.shape[0]).sum(0), facecolor=plot_cols[-1], alpha=0.3)
    axes[0][n-1].plot(rvrange, (norm.pdf(rvrange, rv, 15) / rv.shape[0]).sum(0), color='black', linewidth=2)
#    axes[0][n-1].fill_between(rvrange, y1=0, y2=pdf[n-1], facecolor=plot_cols[0], alpha=0.3)
    for i, ip in enumerate(ipdf[n-1][:, order[n-1]].T):
        axes[0][n-1].plot(rvrange, ip, linewidth=3, color=plot_cols[i])
        axes[0][n-1].annotate(r'$%.1f$, $%.1f$' % (means[n-1][order[n-1]][i], sigmas[n-1][order[n-1]][i]),
                              (0.9, 0.92-0.06*i), fontsize=8, xycoords='axes fraction', ha='right', va='top', color=plot_cols[i])
    axes[0][n-1].plot(rvrange, pdf[n-1], linewidth=2.5, linestyle='dashed', color=plot_cols[-2])

    rsp = resp[n-1][:, np.argsort(models[n-1].means_.squeeze())]
    rprob = rsp.cumsum(1).T
    rprob = np.concatenate((np.zeros((1, 500)), rprob))
    for i in range(len(rprob)-1):
        axes[1][n-1].fill_between(rvrange, y1=rprob[i], y2=rprob[i+1], color=plot_cols[i], alpha=0.3)

    add_grid_minor(axes[0][n-1])
    add_grid_minor(axes[1][n-1])
    axes[0][n-1].set(xlim=(-400, 100), xticks=[-400, -300, -200, -100, 0], ylim=(0, 0.01), yticks=np.arange(2, 10, 2)/1e3)  # xticks broadcast to all axes
    axes[1][n-1].set(xlim=(-400, 100), ylim=(0, 1))
    axes[1][n-1].set_xticklabels([r'$%i$' % x for x in axes[1][n-1].get_xticks()], rotation=90)
    if n > 1:
        axes[0][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
        axes[1][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
    else:
        axes[0][n-1].set_yticklabels(['2', '4', '6', '8', ''])
    axes[0][n-1].annotate('AIC: %.1f' % aic[n-1], (0.1, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    axes[0][n-1].annotate('BIC: %.1f' % bic[n-1], (0.1, 0.86), fontsize=10, xycoords='axes fraction', ha='left', va='top')


fig.subplots_adjust(hspace=0, wspace=0, bottom=0.16, top=0.9)
fig.text(0.5, 0.04, r'Radial Velocity $\left[\rm km~s^{-1} \right]$', ha='center')  # xlabel
axes[0][0].set_ylabel(r'10$^3 \times p(v_{\rm rad})$')
axes[1][0].set_ylabel(r'$p({\rm component} ~|~ v_{\rm rad})$')


### 'tied'
Ncomp = range(1, 6)
models = [mixture.GaussianMixture(n_components=n, covariance_type='tied').fit(rv) for n in Ncomp]
aic = [m.aic(rv) for m in models]
bic = [m.bic(rv) for m in models]
pdf = [np.exp(m.score_samples(rvrange.reshape(-1, 1))) for m in models]  # weighted log(probability) per data point (RV)
resp = [m.predict_proba(rvrange.reshape(-1, 1)) for m in models]  # predicted posterior probability for each component -- P(c | RV)
ipdf = [r * p[:, np.newaxis] for r, p in zip(resp, pdf)]
means = [models[n-1].means_[:, f-1] for f in range(rv.shape[1]) for n in Ncomp]
sigmas = [np.sqrt(models[n-1].covariances_[:, f-1]) for f in range(rv.shape[1]) for n in Ncomp]
order = [np.argsort(m) for m in means]

IC_ax2.plot(Ncomp, aic, linestyle='dashed', linewidth=3, color=plot_cols[2], label='AIC, tied')
IC_ax.plot(Ncomp, np.exp(0.5 * (min(aic) - aic)), linestyle='solid', linewidth=2, color=plot_cols[2], label='AIC, tied')
IC_ax2.plot(Ncomp, bic, linestyle='dashed', linewidth=3, color=plot_cols[3], label='BIC, tied')
IC_ax.plot(Ncomp, np.exp(0.5 * (min(bic) - bic)), linestyle='solid', linewidth=2, color=plot_cols[3], label='BIC, tied')

#
# plot
#
fig, axes = plt.subplots(nrows=2, ncols=Ncomp[-1], sharex=True, figsize=(15, 15))
for n in Ncomp:
#    axes[0][n-1].fill_between(rvcenters, y1=0, y2=dens_all, step='mid', facecolor=plot_cols[-1], alpha=0.3)
#    axes[0][n-1].step(rvcenters, dens_all, where='mid', color='black', linewidth=2)
    axes[0][n-1].fill(rvrange, (norm.pdf(rvrange, rv, 15) / rv.shape[0]).sum(0), facecolor=plot_cols[-1], alpha=0.3)
    axes[0][n-1].plot(rvrange, (norm.pdf(rvrange, rv, 15) / rv.shape[0]).sum(0), color='black', linewidth=2)
#    axes[0][n-1].fill_between(rvrange, y1=0, y2=pdf[n-1], facecolor=plot_cols[0], alpha=0.3)
    for i, ip in enumerate(ipdf[n-1][:, order[n-1]].T):
        axes[0][n-1].plot(rvrange, ip, linewidth=3, color=plot_cols[i])
        axes[0][n-1].annotate(r'$%.1f$, $%.1f$' % (means[n-1][order[n-1]][i], sigmas[n-1][0]),
                              (0.9, 0.92-0.06*i), fontsize=8, xycoords='axes fraction', ha='right', va='top', color=plot_cols[i])
    axes[0][n-1].plot(rvrange, pdf[n-1], linewidth=2.5, linestyle='dashed', color=plot_cols[-2])

    rsp = resp[n-1][:, np.argsort(models[n-1].means_.squeeze())]
    rprob = rsp.cumsum(1).T
    rprob = np.concatenate((np.zeros((1, 500)), rprob))
    for i in range(len(rprob)-1):
        axes[1][n-1].fill_between(rvrange, y1=rprob[i], y2=rprob[i+1], color=plot_cols[i], alpha=0.3)

    add_grid_minor(axes[0][n-1])
    add_grid_minor(axes[1][n-1])
    axes[0][n-1].set(xlim=(-400, 100), xticks=[-400, -300, -200, -100, 0], ylim=(0, 0.01), yticks=np.arange(2, 10, 2)/1e3)  # xticks broadcast to all axes
    axes[1][n-1].set(xlim=(-400, 100), ylim=(0, 1))
    axes[1][n-1].set_xticklabels([r'$%i$' % x for x in axes[1][n-1].get_xticks()], rotation=90)
    if n > 1:
        axes[0][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
        axes[1][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
    else:
        axes[0][n-1].set_yticklabels(['2', '4', '6', '8', ''])
    axes[0][n-1].annotate('AIC: %.1f' % aic[n-1], (0.1, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    axes[0][n-1].annotate('BIC: %.1f' % bic[n-1], (0.1, 0.86), fontsize=10, xycoords='axes fraction', ha='left', va='top')


fig.subplots_adjust(hspace=0, wspace=0, bottom=0.16, top=0.9)
fig.text(0.5, 0.04, r'Radial Velocity $\left[\rm km~s^{-1} \right]$', ha='center')  # xlabel
axes[0][0].set_ylabel(r'10$^3 \times p(v_{\rm rad})$')
axes[1][0].set_ylabel(r'$p({\rm component} ~|~ v_{\rm rad})$')

IC_ax.set(xticks=[1, 2, 3, 4, 5], xlabel='Number of Gaussian Components', ylabel=r'$\propto p(N_{\rm comp} = n)$')
IC_ax.legend(loc='upper center', ncol=2, fontsize=14)
add_grid_minor(IC_ax)
IC_ax2.set_ylabel('Information Criterion')

"""
### Bayesian 'tied'
Ncomp = range(1, 6)
model = mixture.BayesianGaussianMixture(n_components=max(Ncomp), covariance_type='full', n_init=10).fit(rv)
models = [mixture.BayesianGaussianMixture(n_components=n, covariance_type='tied').fit(rv) for n in Ncomp]
aic = [m.aic(rv) for m in models]
bic = [m.bic(rv) for m in models]
pdf = [np.exp(m.score_samples(rvrange.reshape(-1, 1))) for m in models]  # weighted log(probability) per data point (RV)
resp = [m.predict_proba(rvrange.reshape(-1, 1)) for m in models]  # predicted posterior probability for each component -- P(c | RV)
ipdf = [r * p[:, np.newaxis] for r, p in zip(resp, pdf)]
means = [models[n-1].means_[:, f-1] for f in range(rv.shape[1]) for n in Ncomp]
sigmas = [np.sqrt(models[n-1].covariances_[:, f-1]) for f in range(rv.shape[1]) for n in Ncomp]
order = [np.argsort(m) for m in means]

#
# plot
#
fig, axes = plt.subplots(nrows=2, ncols=Ncomp[-1], sharex=True, figsize=(15, 15))
for n in Ncomp:
    axes[0][n-1].fill_between(rvcenters, y1=0, y2=dens_all, step='mid', facecolor=plot_cols[-1], alpha=0.3)
    axes[0][n-1].step(rvcenters, dens_all, where='mid', color='black', linewidth=2)
#    axes[0][n-1].fill_between(rvrange, y1=0, y2=pdf[n-1], facecolor=plot_cols[0], alpha=0.3)
    for i, ip in enumerate(ipdf[n-1][:, order[n-1]].T):
        axes[0][n-1].plot(rvrange, ip, linewidth=3, color=plot_cols[i])
        axes[0][n-1].annotate(r'$%.1f$, $%.1f$' % (means[n-1][i], sigmas[n-1][0]),  # sigmas[n-1][i]),
                              (0.9, 0.92-0.06*i), fontsize=8, xycoords='axes fraction', ha='right', va='top', color=plot_cols[i])
    axes[0][n-1].plot(rvrange, pdf[n-1], linewidth=2.5, linestyle='dashed', color=plot_cols[-2])

    rsp = resp[n-1][:, np.argsort(models[n-1].means_.squeeze())]
    rprob = rsp.cumsum(1).T
    rprob = np.concatenate((np.zeros((1, 500)), rprob))
    for i in range(len(rprob)-1):
        axes[1][n-1].fill_between(rvrange, y1=rprob[i], y2=rprob[i+1], color=plot_cols[i], alpha=0.3)

    add_grid_minor(axes[0][n-1])
    add_grid_minor(axes[1][n-1])
    axes[0][n-1].set(xlim=(-400, 100), xticks=[-400, -300, -200, -100, 0], ylim=(0, 0.01), yticks=np.arange(2, 10, 2)/1e3)  # xticks broadcast to all axes
    axes[1][n-1].set(xlim=(-400, 100), ylim=(0, 1))
    axes[1][n-1].set_xticklabels([r'$%i$' % x for x in axes[1][n-1].get_xticks()], rotation=90)
    if n > 1:
        axes[0][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
        axes[1][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
    else:
        axes[0][n-1].set_yticklabels(['2', '4', '6', '8', ''])
    axes[0][n-1].annotate('AIC: %.1f' % aic[n-1], (0.1, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    axes[0][n-1].annotate('BIC: %.1f' % bic[n-1], (0.1, 0.86), fontsize=10, xycoords='axes fraction', ha='left', va='top')


fig.subplots_adjust(hspace=0, wspace=0, bottom=0.16, top=0.9)
fig.text(0.5, 0.04, r'Radial Velocity $\left[\rm km~s^{-1} \right]$', ha='center')  # xlabel
axes[0][0].set_ylabel(r'10$^3 \times p(v_{\rm rad})$')
axes[1][0].set_ylabel(r'$p({\rm component} ~|~ v_{\rm rad})$')
"""

# --------- #
# GMM - M/H #
# --------- #
mh = MH[np.isfinite(MH)].reshape(-1, 1)  # recasting to a column vector
mhrange = np.linspace(-2, 1, 500)
mhdens_all, mhbins = np.histogram(mh, bins='auto', density=True)
mhcenters = (mhbins[1:] + mhbins[:-1]) / 2

### 'full'
Ncomp = range(1, 6)
models = [mixture.GaussianMixture(n_components=n, covariance_type='full').fit(mh) for n in Ncomp]
aic = [m.aic(rv) for m in models]
bic = [m.bic(rv) for m in models]
pdf = [np.exp(m.score_samples(mhrange.reshape(-1, 1))) for m in models]  # weighted log(probability) per data point (RV)
resp = [m.predict_proba(mhrange.reshape(-1, 1)) for m in models]  # predicted posterior probability for each component -- P(c | RV)
ipdf = [r * p[:, np.newaxis] for r, p in zip(resp, pdf)]
means = [models[n-1].means_[:, f-1] for f in range(rv.shape[1]) for n in Ncomp]
sigmas = [np.sqrt(models[n-1].covariances_[:, f-1, f-1]) for f in range(rv.shape[1]) for n in Ncomp]
order = [np.argsort(m) for m in means]

#
# plot
#
fig, axes = plt.subplots(nrows=2, ncols=Ncomp[-1], sharex=True, figsize=(15, 15))
for n in Ncomp:
    axes[0][n-1].fill_between(mhcenters, y1=0, y2=mhdens_all, step='mid', facecolor=plot_cols[-1], alpha=0.3)
    axes[0][n-1].step(mhcenters, mhdens_all, where='mid', color='black', linewidth=2)
#    axes[0][n-1].fill_between(rvrange, y1=0, y2=pdf[n-1], facecolor=plot_cols[0], alpha=0.3)
    for i, ip in enumerate(ipdf[n-1][:, order[n-1]].T):
        axes[0][n-1].plot(mhrange, ip, linewidth=3, color=plot_cols[i])
        axes[0][n-1].annotate(r'$%.2f$, $%.2f$' % (means[n-1][order[n-1]][i], sigmas[n-1][order[n-1]][i]),
                              (0.1, 0.75-0.06*i), fontsize=8, xycoords='axes fraction', ha='left', va='top', color=plot_cols[i])
    axes[0][n-1].plot(mhrange, pdf[n-1], linewidth=2.5, linestyle='dashed', color=plot_cols[-2])

    rsp = resp[n-1][:, np.argsort(models[n-1].means_.squeeze())]
    rprob = rsp.cumsum(1).T
    rprob = np.concatenate((np.zeros((1, 500)), rprob))
    for i in range(len(rprob)-1):
        axes[1][n-1].fill_between(mhrange, y1=rprob[i], y2=rprob[i+1], color=plot_cols[i], alpha=0.3)

    add_grid_minor(axes[0][n-1])
    add_grid_minor(axes[1][n-1])
    axes[0][n-1].set(xlim=(-1.8, 0.5), ylim=(0, 1.5))#, yticks=np.arange(2, 10, 2)/1e3)  # xticks broadcast to all axes
    axes[1][n-1].set(xlim=(-1.8, 0.5), ylim=(0, 1))
#    axes[1][n-1].set_xticklabels([r'$%i$' % x for x in axes[1][n-1].get_xticks()], rotation=90)
    if n > 1:
        axes[0][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
        axes[1][n-1].set_yticklabels(['' for y in axes[0][n-1].get_yticks()])
    else:
        #pass
        axes[0][n-1].set_yticklabels(['', 0.5, 1.0, 1.5])
    axes[0][n-1].annotate(r'$10^7$ AIC: %.2f' % (aic[n-1]/1e7), (0.1, 0.92), fontsize=10, xycoords='axes fraction', ha='left', va='top')
    axes[0][n-1].annotate(r'$10^7$ BIC: %.2f' % (bic[n-1]/1e7), (0.1, 0.86), fontsize=10, xycoords='axes fraction', ha='left', va='top')


fig.subplots_adjust(hspace=0, wspace=0, bottom=0.16, top=0.9)
fig.text(0.5, 0.04, 'Metallicity [M/H] [dex]', ha='center')  # xlabel
axes[0][0].set_ylabel(r'$p$([M/H])')
axes[1][0].set_ylabel(r'$p$(component | [M/H])')


