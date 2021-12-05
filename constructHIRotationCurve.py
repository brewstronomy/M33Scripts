#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import os.path as path, numpy as np, matplotlib.pyplot as plt, astropy.units as u, matplotlib.gridspec as gridspec
import M33.constant as con
from M33.rdis import correct_rgc
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from M33.resample_wavelengths import extract_wavelengths
from pvextractor import PathFromCenter, extract_pv_slice
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import ListedColormap

# --------- #
# functions #
# --------- #
class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handles, x0, y0, width, height, fontsize, trans):
        def get_handle_props(orig_handle_tup):
            h1, h2 = orig_handle_tup
            c1, c2 = h1.get_color(), h2.get_color()
            ls1, ls2 = h1.get_linestyle(), h2.get_linestyle()
            lw1, lw2 = h1.get_linewidth(), h2.get_linewidth()
            return c1, ls1, lw1, c2, ls2, lw2

        c1, ls1, lw1, c2, ls2, lw2 = get_handle_props(orig_handles)
        l1 = plt.Line2D([x0, y0 + 1.2 * width], [0.925 * height, 0.925 * height], color=c1, linestyle=ls1, linewidth=lw1)
        l2 = plt.Line2D([x0, y0 + 1.2 * width], [0.075 * height, 0.075 * height], color=c2, linestyle=ls2, linewidth=lw2)
        return [l1, l2]

def gauss(x, amp, mu, sig, c):
    arg = -0.5 * ( (x - mu) / sig ) ** 2
    return amp * np.exp(arg) + c

def v_nonpar(r, pars):
    r = np.array(r)
    v0, r0, d = pars
    return v0 * (r / r0 + d) / (r / r0 + 1)

def MonteCarlo(xdata, ydata, fitfunc, init_pars, bounds, kw_noise, Nsims=1000, init_cov=None):
    # setup fit arrays
    fit_pars, fit_errs = [], []
    fit_residuals_data = []
    fit_residuals_noise = []
    # cycle
    for n in range(Nsims):
        # add noise
        noise = np.random.normal(size=ydata.size, **kw_noise)
        y_noise = ydata + noise
        # fit
        try:
            pars, pars_cov = curve_fit(fitfunc, xdata, y_noise, p0=init_pars, bounds=bounds)
        except:  # if failed, replace with initial best fit
            pars, pars_cov = init_pars, init_cov
        # compute residuals
        res_data = ydata - fitfunc(xdata, *pars)
        res_noise = y_noise - fitfunc(xdata, *pars)
        # parse
        fit_pars.append(pars)
        fit_errs.append(np.sqrt(np.diag(pars_cov)))
        fit_residuals_data.append(res_data)
        fit_residuals_noise.append(res_noise)
    result = [fit_pars, fit_errs, fit_residuals_data, fit_residuals_noise]
    return [np.array(arr) for arr in result]

def MegaToilet():
    # overall
    fig = plt.figure(figsize=(30, 40))
    plot_grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

    # split first row into 1x2
    row1 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=plot_grid[0, :])
    axes1 = [plt.Subplot(fig, r) for r in row1]

    # split last row into 1x1
    row2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=plot_grid[1, :])
    axes2 = [plt.Subplot(fig, row2[0:])]

    # register with fig, then return
    axes = [*axes1, *axes2]
    for ax in axes:
        fig.add_subplot(ax)

    return fig, axes

# -------------- #
# setup & import #
# -------------- #
m33 = SkyCoord(con.ra_m33 * u.deg, con.dec_m33 * u.deg)
m33_kw = {'glx_ctr': m33, 'glx_PA': con.PA_m33, 'glx_incl': con.inc_m33, 'glx_dist': con.dist_m33}

# 3D cube
file_path = '/Users/luca/Documents/Research/M33/Data/Keenan2016_M33_HIData/'
file_name = 'M33_Arecibo_HI_edited.fits'
file = path.join(file_path, file_name)

data, head = fits.getdata(file, header=True)  # mK & km/s
wcs = WCS(head)
pix_scale = wcs.wcs.cdelt[1]  # deg/pix; pixel scale of Arecibo observations (effectively 1 pix ~ 1 arcmin)
beam = np.array([head['BMAJ'], head['BMIN']]) * u.deg  # Arecibo beam
beamwidth_ang = np.sqrt(beam[0] * beam[1])  # deg; FWHM = sqrt(Bmaj * Bmin)
beamwidth_pix = beamwidth_ang.to(u.deg).value / pix_scale  # pix
beamwidth_phys = beamwidth_ang.to(u.rad).value * con.dist_m33.kpc  # kpc

# velocity field (for plotting)
m1_name = 'M33_HI_mom1.fits'
m1_file = path.join(file_path, m1_name)
mom1, m1_head = fits.getdata(m1_file, header=True)
m1_wcs = WCS(m1_head)

# comparison rotation curve from Corbelli et al. (2014)
corbelli_path = '/Users/luca/Documents/Research/M33/Data/Corbelli2014_HI/'
corbelli_file = path.join(corbelli_path, 'Corbelli2014_RotCur.txt')
corbelli = ascii.read(corbelli_file)
corb_rad, corb_vel, corb_sig = [np.array(corbelli[k]) for k in ['R', 'RV', 'sig']]

# -------------------- #
# construct PV diagram #
# -------------------- #
# define PV path [relative to galaxy center] & extract
pv_path = PathFromCenter(center=m33, length=1.7*u.deg, angle=con.PA_m33, width=beamwidth_ang.to(u.arcmin))
pv_slice = extract_pv_slice(file, pv_path, wcs=wcs)
pv_data, pv_head = pv_slice.data, pv_slice.header

# prepare slice coords
slice_x, slice_y = pv_path.sample_points(1, wcs=wcs)  # pix coords of slice sampling (rate of 1 per native sampling)
slice_xyvcoords = np.array([(x, y, 193) for x, y in zip(slice_x, slice_y)])  # 193 ~ argwhere(v_sys) -- unused
slice_xycoords = slice_xyvcoords.T[:-1].T  # shedding the spectral axis

slice_ra, slice_dec = np.array([w[:-1] for w in wcs.pixel_to_world_values([*slice_xyvcoords])]).T  # deg, deg
slice_wcscoords = SkyCoord(slice_ra * u.deg, slice_dec * u.deg)  # deg

slice_head, slice_tail = slice_wcscoords[[-1, 0]] # deg; wcs for endpoints
slice_pa = slice_tail.position_angle(slice_head)  # deg; position angle (E of N)

# pv axes
vels_full = extract_wavelengths(pv_slice.header, spectral_axis=2) / 1e3  # km/s; full velocity range
offs_ang = slice_tail.separation(slice_wcscoords) - m33.separation(slice_tail)  # deg; angular sep from galaxy center
offs_phys = offs_ang.rad * con.dist_m33.kpc  # kpc; projected offset along slice -- not a radius!

# mask only to relevant velocities
vmask = (vels_full >= -325) & (vels_full <= -50)  # km/s; visual inspection
vels = vels_full[vmask]  # km/s
vwidth = np.abs(np.diff(vels)[0])  # km/s; channel width (for uncertainty propagation)
pvdat = pv_data[vmask, :]

# define image boundaries by pv slice extent
extent_orig = [offs_phys.min(), offs_phys.max(), vels_full.max(), vels_full.min()]
extent = [offs_phys.min(), offs_phys.max(), vels.max(), vels.min()]

# ---------------------- #
# extract rotation curve #
# ---------------------- #
fit_mus, fit_sigs, fit_emus, fit_esigs = [], [], [], []
for idx in tqdm(range(offs_phys.size)):
    # parse
    x, y = vels, pvdat[:, idx]
    # automated guess for initial fit
    amp_guess = np.nanmax(y)
    mu_guess = vels[np.argmax(y)]
    c_guess = 0  # assume no background

    # # # estimate initial sigma w/ a full-width extrapolation
    # # # FWnM = sqrt(8 * ln(1 / n)) * sig  for the full-width at n% of the max
    n_pct = 80 / 100  # initial attempt
    vels_fwnm = vels[y >= n_pct * np.nanmax(y)]  # estimate initial sigma by a full-width extrapoloation
    while len(vels_fwnm) < 2:  # if no velocity spread, reduce n% until successful
        n_pct -= 5/100  # 5% increments
        vels_fwnm = vels[y >= n_pct * np.nanmax(y)]
    FWNM = vels_fwnm.max() - vels_fwnm.min()  # full width at 80% of maximum
    sig_guess = FWNM / np.sqrt(8 * np.log(1 / n_pct))  # FWnM = sqrt(8 * ln(100 / n)) * sig
    # initial fit
    init_pars, init_cov = curve_fit(gauss, x, y, p0=[amp_guess, mu_guess, sig_guess, c_guess])
    # use residuals to estimate noise addition
    init_res = y - gauss(x, *init_pars)
    kw_noise = {'loc': 0, 'scale': np.sqrt(np.mean(init_res**2))}  # noise amplitude is 2x RMS
    # perform full simulation
    bounds = [(0, vels.min(), sig_guess / 3, -np.inf),
              (3 * amp_guess, vels.max(), 3 * FWNM, np.inf)]
    fit_pars, fit_errs, res_data, res_noise = MonteCarlo(x, y, gauss, init_pars, bounds, kw_noise, Nsims=500, init_cov=init_cov)
    # parse results & compute statistics
    amps, mus, sigs, cs = fit_pars.T  # len == Nsims
    eamps, emus, esigs, ecs = fit_errs.T  # formal fitting errors

    tup_mu = np.percentile(mus, [16, 50, 84])
    mu_fit = tup_mu[1]  # median of the distribution as the final best fit parameter
    mu_err_syst = np.median(emus)  # representative fitting error
    mu_err_stat = np.nanmax([tup_mu[1] - tup_mu[0], tup_mu[2] - tup_mu[1]])  # scatter in the distribution using largest quartile
    mu_err = np.sqrt(np.sum(np.square([mu_err_stat, mu_err_syst, vwidth])))  # quad. sum of e_stat, e_syst & v_width

    tup_sig = np.percentile(sigs, [16, 50, 84])  # same procedure
    sig_fit = tup_sig[1]
    sig_err_syst = np.median(esigs)
    sig_err_stat = np.nanmax([tup_sig[1] - tup_sig[0], tup_sig[2] - tup_sig[1]])
    sig_err = np.sqrt(np.sum(np.square([sig_err_stat, sig_err_syst, vwidth])))
    # pipe to arrays
    fit_mus.append(mu_fit)
    fit_emus.append(mu_err)
    fit_sigs.append(sig_fit)
    fit_esigs.append(sig_err)

fit_mus = np.array(fit_mus)
fit_emus = np.array(fit_emus)
fit_sigs = np.array(fit_sigs)
fit_esigs = np.array(fit_esigs)

# shift to v_rot & rgc
# # de-projected galactocentric distance
r_deproj = np.array([correct_rgc(coord, **m33_kw).kpc for coord in slice_wcscoords])  # kpc; positive definite
v_rot = np.abs(fit_mus - con.RV_m33) / np.sin(con.inc_m33.rad) / np.cos(con.PA_m33.rad) # km/s
ev_rot = np.sqrt(np.sum(np.square([fit_emus, con.eRV_m33])))

# bin by radius to avoid oversampling
_, r_deproj_binedges = np.histogram(r_deproj, bins=np.arange(r_deproj.min(), r_deproj.max(), beamwidth_phys))
r_deproj_binned = (r_deproj_binedges[1:] + r_deproj_binedges[:-1]) / 2  # kpc

v_rot_binned, ev_rot_binned = [], []
sigs_binned, esigs_binned = [], []
for ctr in r_deproj_binned:
    rmask = (r_deproj >= ctr - beamwidth_phys / 2) & (r_deproj < ctr + beamwidth_phys / 2)

    vbin, evbin = np.nanmedian(v_rot[rmask]), np.nanmean(ev_rot[rmask])
    v_rot_binned.append(vbin); ev_rot_binned.append(evbin)

    sbin, esbin = np.nanmedian(fit_sigs[rmask]), np.nanmean(fit_esigs[rmask])
    sigs_binned.append(sbin); esigs_binned.append(esbin)

v_rot_binned, ev_rot_binned = np.array(v_rot_binned), np.array(ev_rot_binned)
sigs_binned, esigs_binned = np.array(sigs_binned), np.array(esigs_binned)

# same for unfolded offsets & uncorrected velocities
_, offs_phys_binedges = np.histogram(offs_phys, bins=np.arange(offs_phys.min(), offs_phys.max(), beamwidth_phys))
offs_phys_binned = (offs_phys_binedges[1:] + offs_phys_binedges[:-1]) / 2

fit_mus_binned, fit_emus_binned = [], []
fit_sigs_binned, fit_esigs_binned = [], []
slice_x_binned, slice_y_binned = [], []
for ctr in offs_phys_binned:
    pmask = (offs_phys >= ctr - beamwidth_phys / 2) & (offs_phys < ctr + beamwidth_phys / 2)

    mbin, embin = np.nanmedian(fit_mus[pmask]), np.nanmean(fit_emus[pmask])
    fit_mus_binned.append(mbin); fit_emus_binned.append(embin)

    sbin, esbin = np.nanmedian(fit_sigs[pmask]), np.nanmean(fit_esigs[pmask])
    fit_sigs_binned.append(sbin); fit_esigs_binned.append(esbin)

    slice_x_binned.append(np.median(slice_x[pmask]))
    slice_y_binned.append(np.median(slice_y[pmask]))

fit_mus_binned, fit_emus_binned = np.array(fit_mus_binned), np.array(fit_emus_binned)
fit_sigs_binned, fit_esigs_binned = np.array(fit_sigs_binned), np.array(fit_esigs_binned)
slice_x_binned, slice_y_binned = np.array(slice_x_binned), np.array(slice_y_binned)

# split into approaching/receding sides
posmask = lambda x: x > 0
negmask = lambda x: ~posmask(x)
# # unbinned
pmsk, nmsk = posmask(offs_phys), negmask(offs_phys)

offs_phys_pos, offs_phys_neg = offs_phys[pmsk], offs_phys[nmsk]
r_deproj_pos, r_deproj_neg = r_deproj[pmsk], r_deproj[nmsk]

fit_mus_pos, fit_mus_neg = fit_mus[pmsk], fit_mus[nmsk]
fit_emus_pos, fit_emus_neg = fit_emus[pmsk], fit_emus[nmsk]
fit_sigs_pos, fit_sigs_neg = fit_sigs[pmsk], fit_sigs[nmsk]
fit_esigs_pos, fit_esigs_neg = fit_esigs[pmsk], fit_esigs[nmsk]

v_rot_pos, v_rot_neg = v_rot[pmsk], v_rot[nmsk]
ev_rot_pos, ev_rot_neg = ev_rot[pmsk], ev_rot[nmsk]

# # binned
pmsk, nmsk = posmask(offs_phys_binned), negmask(offs_phys_binned)

offs_phys_binned_pos, offs_phys_binned_neg = offs_phys_binned[pmsk], offs_phys_binned[nmsk]
fit_mus_binned_pos, fit_mus_binned_neg = fit_mus_binned[pmsk], fit_mus_binned[nmsk]
fit_emus_binned_pos, fit_emus_binned_neg = fit_emus_binned[pmsk], fit_emus_binned[nmsk]
fit_sigs_binned_pos, fit_sigs_binned_neg = fit_sigs_binned[pmsk], fit_sigs_binned[nmsk]
fit_esigs_binned_pos, fit_esigs_binned_neg = fit_esigs_binned[pmsk], fit_esigs_binned[nmsk]
# # deprojected radii & rotational velocities
pmsk, nmsk = posmask(offs_phys), negmask(offs_phys)
offs_phys_pos, offs_phys_neg = offs_phys[pmsk], offs_phys[nmsk]
fit_mus_pos, fit_mus_neg = fit_mus[pmsk], fit_mus[nmsk]

# --------------- #
# plot PV diagram #
# --------------- #
# colormap shenanigans
cmap = plt.cm.get_cmap('summer_r', 256)
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
alphas = np.linspace(0, 1, cmap.N); bkgd = np.array([1, 1, 1])
cmap_colors[:, :-1] = np.array([col[:-1] * alphas[i] + bkgd * (1 - alphas[i]) for i, col in enumerate(cmap_colors)])
cmap_new = ListedColormap(np.clip(cmap_colors, 0, 1))

fig, (ax1, ax2, ax3) = MegaToilet()

img = ax1.imshow(mom1, origin='lower', cmap='Spectral_r', vmin=-300, vmax=-50)
ax1.contour(mom1, levels=np.arange(-300, -50, vwidth), colors='black', linewidths=0.5, linestyles='solid')
ax1.contour(mom1, levels=[-180], colors='black', linewidths=1.5, linestyles='solid')

plt.rcParams['hatch.linewidth'] = 0.7
beam_patch = plt.Circle((128, 89), beamwidth_pix / 2, facecolor='none', hatch='xxxxxx', edgecolor='black', linewidth=0.6)
slice_patch = plt.Rectangle(slice_xycoords[0] + beamwidth_pix * np.cos(np.pi/2 + slice_pa.rad), width=beamwidth_pix,
                            height=1.01 * np.sqrt(np.diff(slice_x[[0, -1]])**2 + np.diff(slice_y[[0, -1]])**2),
                            angle=con.PA_m33.deg, facecolor='black', edgecolor='none')  # cos term shifts along minor-axis, 1.01 factor is for visuals
ax1.add_artist(beam_patch)
ax1.add_artist(slice_patch)
ax1.arrow(*slice_xycoords[0], *np.diff(slice_xycoords[[0, -1]].T).ravel(), color='black', linewidth=2,
          head_width=7, head_length=7, zorder=1)
#ax1.errorbar(slice_x_binned, slice_y_binned, fmt='^', color='white', markersize=2.5)

lft, tp, wdth, hght = ax1.get_position().bounds  # ; hght = wdth/8
cax = fig.add_axes([lft - 0.04, 0.5 + 0.05, 0.02, hght - 0.05])
cbar = plt.colorbar(img, cax=cax)  #, orientation='horizontal')
cbar.set_label(r'LOS Velocity $\left[{\rm km~s}^{-1}\right]$')
cax.yaxis.set_ticks_position('left')

ax1.axis('off')
ax1.set(xlim=(120, 200), ylim=(80, 205))

img = ax2.imshow(np.log10(pv_data), origin='lower', cmap=cmap_new, aspect='auto', extent=extent_orig, vmin=1, vmax=4)
ax2.contour(pv_data, levels=2 ** (np.linspace(0, 15, 30)) * 30, linewidths=1, colors='black', extent=extent_orig)
ax2.plot(offs_phys, fit_mus, color='black', linewidth=1.5)
ax2.errorbar(offs_phys_binned, fit_mus_binned, fmt='^', color='black', markeredgecolor='black', markeredgewidth=0.9, markersize=8)
ax2.errorbar(-9.5, -280, xerr=beamwidth_phys, yerr=np.median(fit_emus_binned), fmt='', ecolor='black', elinewidth=1.5, capsize=3)

lft, bttm, wdth, hght = ax2.get_position().bounds
cax = fig.add_axes([lft + 0.85 * wdth, bttm + 0.1 * hght, 0.045 * wdth, 0.3 * hght])
cbar = plt.colorbar(img, cax=cax)
cax.tick_params(which='both', length=0)
cax.yaxis.set_ticks_position('left')
cbar.set_ticks([1, 4])
cbar.set_ticklabels([r'$10^{%i}$' % (a - 3) for a in cbar.get_ticks()])
clft, cbttm, cwdth, chght = cax.get_position().bounds
ax2.annotate(r'T$_{\rm B}~\left[{\rm K}\right]$', (clft + 0.5 * cwdth, cbttm - 0.08),  #+ 0.05 * chght),
             xycoords='axes fraction', ha='center', va='center')

ax2.grid()
ax2.set(xlabel=r'Projected Galactocentric Offset $\left[{\rm kpc}\right]$', xlim=(-12, 12), ylim=(-315, -25),
        ylabel=r'LOS Velocity Along Slice $\left[{\rm km~s}^{-1}\right]$')
ax2.set_xticks(np.arange(-12, 13, 4))
ax2.set_yticks(np.arange(-300, 0, 25))
ax2.invert_yaxis()

ax2.plot(ax2.get_xlim(), [con.RV_m33, con.RV_m33], color='black', linestyle='dashed', linewidth=0.7, zorder=1)
ax2.plot([0, 0], ax2.get_ylim(), color='black', linestyle='dashed', linewidth=0.7, zorder=1)

blue, red = '#347fbb', '#f3693b'  # eyedropper-sampled at extrema of moment 1 map

#p_sgp, = ax3.plot(r_deproj_pos, fit_sigs_pos, color=blue, linewidth=3, linestyle='dotted', zorder=2)
#p_sgn, = ax3.plot(r_deproj_neg, fit_sigs_neg, color=red, linewidth=3, linestyle='dotted', zorder=2)
#p_sgb = ax3.errorbar(r_deproj_binned, sigs_binned, fmt='^', color='none',
#                     markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=2, label=r'$\langle \sigma_{\rm disp} \rangle$')
p_vrp, = ax3.plot(r_deproj_pos, v_rot_pos, color=blue, linewidth=3, linestyle='dashed', zorder=2)
p_vrn, = ax3.plot(r_deproj_neg, v_rot_neg, color=red, linewidth=3, linestyle='dashed', zorder=2)
p_vrb = ax3.errorbar(r_deproj_binned, v_rot_binned, fmt='^', color='black',
                     markersize=10, markeredgecolor='black', markeredgewidth=1.2, zorder=2, label=r'Binned')
p_cor, = ax3.plot(corb_rad, corb_vel, color='gray', linewidth=5, linestyle='solid', zorder=1, label=r'Corbelli+14')
#ax3.plot(corb_rad, corb_sig, color='gray', linewidth=4, linestyle='dotted', zorder=1)
rrange = np.linspace(0, r_deproj.max(), 200)
rrange_full = np.linspace(0, 22, 200)
p_npr, = ax3.plot(rrange_full, v_nonpar(rrange_full, [130.2, 1.3, 0.12]), color='gold', linewidth=2.5, linestyle='solid', zorder=1, label=r'L\'{o}pez Fune+17')

ax3.legend([(p_vrn, p_vrp), p_vrb, p_npr, p_cor],
           [r'$V_{\rm rot}$', p_vrb.get_label(), p_npr.get_label(), p_cor.get_label()],
           handler_map={tuple: AnyObjectHandler()}, ncol=2, fontsize='large', loc='lower center')
ax3.set(xlabel=r'De-Projected Galactocentric Distance $\left[{\rm kpc}\right]$', ylabel=r'Velocity $\left[{\rm km~s}^{-1} \right]$',
        xlim=(0, 15), ylim=(0, 150))
ax3.set_xticks(np.arange(0, 20, 2))

ax3.errorbar([0.9 * np.abs(np.diff(ax3.get_xbound()))], [0.2 * np.abs(np.diff(ax3.get_ybound()))],
              xerr=beamwidth_phys, yerr=np.median(ev_rot_binned), fmt='', ecolor='black', elinewidth=1.5, capsize=4)

fig.subplots_adjust(wspace=-0.1, hspace=0.3)
#fig.suptitle('I make-a da plot', y=0.93)

plt.savefig(path.join('/Users/luca/Documents/Research/M33/Analysis/Plots/APOGEE_HIstuff/', 'HIRotCur.pdf'), bbox_inches='tight')