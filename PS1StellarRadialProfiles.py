#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import os.path as path, numpy as np, matplotlib.pyplot as plt, astropy.units as u
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.isophote import EllipseGeometry, Ellipse, EllipseSample, Isophote, IsophoteList
from photutils import EllipticalAperture
from scipy.optimize import curve_fit
from M33 import constant as con
from tqdm import tqdm

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

stars = 'Beale'

# # # ----------------
# # # helper functions
# # # ----------------
def ps1_file_puller(name, file):
    fname = file.format(name)
    if path.isfile(fname):
        data, head = fits.getdata(fname, header=True)
        data /= head['EXPTIME']  # "DU" to flux (in "DU"/s)
        wcs = WCS(head)
    else:
        data = np.ones(shape=(240, 240))
        head = fits.PrimaryHDU(data=data).header
        wcs = 'rectilinear'

    return data, wcs, head

def flux_to_mag(flux, eflux, mag_zp=25):
    mag = -2.5 * np.log10(flux) + mag_zp
    emag = (2.5 / np.log(10)) * eflux / flux
    return mag, emag

def flux_to_surfdens(flux, eflux, area_pc, earea_pc=None):
    surfdens = flux / area_pc
    if earea_pc is None:
        earea = np.zeros_like(flux)
    else:
        earea = earea_pc
    t1 = eflux / flux; t2 = earea / area_pc
    esurfdens = surfdens * np.sqrt(t1**2 + t2**2)
    return surfdens, esurfdens

def mag_to_surfbright(mag, emag, area_as, earea_as=None):
    mu = mag + 2.5 * np.log10(area_as)
    if earea_as is None:
        earea = np.zeros_like(mag)
    else:
        earea = earea_as
    t1 = emag; t2 = (2.5 / np.log(10)) * earea / area_as
    emu = np.sqrt(t1**2 + t2**2)
    return mu, emu

def surfbright_to_astro(surfbright, esurfbright, absmag_sol=4.64, m_offset=21.572):  # default: AB mag_sol in PS1_r
    surfbright_astro = 10 ** ((surfbright - absmag_sol - m_offset) / (-2.5))
    esurfbright_astro = (np.log(10) / 2.5) * surfbright_astro * esurfbright
    return surfbright_astro, esurfbright_astro

def ps1_ellipses(isos, img, hdr, wcs):
    # constants
    sma_ang = (isos.sma * wcs.wcs.cdelt[0] * u.deg).to(u.arcsec).value  # arcsec
    sma_phys = (sma_ang * u.arcsec).to(u.rad).value * con.dist_m33.pc  # pc

    pix_area_ang = (wcs.wcs.cdelt[0] * u.deg).to(u.arcsec).value**2  # arcsec^2 (/ pix)
    pix_area_phys = (pix_area_ang * u.arcsec ** 2).to(u.rad ** 2).value * con.dist_m33.pc  # pc^2 (/ pix)

    area_ang = np.pi * (1 - isos.eps) * sma_ang ** 2  # sq as; from A_ellipse = pi a b
    e_area_ang = np.pi * sma_ang ** 2 * isos.ellip_err  # sq as
    area_phys = (area_ang * u.arcsec ** 2).to(u.rad ** 2).value * con.dist_m33.pc ** 2  # pc^2
    e_area_phys = (e_area_ang * u.arcsec ** 2).to(u.rad ** 2).value * con.dist_m33.pc ** 2  # pc^2

    e_tflux = np.sqrt(isos.npix_e) * isos.pix_stddev  # DU/s

    #   estimate background for masking
    mask_sigclip = np.abs(img) < 3 * np.nanstd(img)  # essentially a 3-sigma-clip
    bkgd = np.nanstd(img[mask_sigclip])  # DU/s
    mag_bkgd = flux_to_mag(bkgd, 0)[0]  # AB mag
    surfdens_bkgd = flux_to_surfdens(bkgd, 0, pix_area_phys)[0] # DU/s/pc^2
    surfbright_bkgd = mag_to_surfbright(mag_bkgd, 0, pix_area_ang)[0]  # AB mag/ss (/pix)
    surfdens_phys_bkgd = surfbright_to_astro(surfbright_bkgd, 0)[0]  # Lsol/pc^2

    # main
    mag_avg, e_mag_avg = flux_to_mag(isos.intens, isos.int_err)  # AB mag
    mag_tot, e_mag_tot = flux_to_mag(isos.tflux_e, e_tflux)  # AB mag

    surfdens_avg, e_surfdens_avg = flux_to_surfdens(isos.intens, isos.int_err, pix_area_phys)  # DU/s/pc^2
    surfdens_tot, e_surfdens_tot = flux_to_surfdens(isos.tflux_e, e_tflux, area_phys, e_area_phys)  # DU/s/pc^2

    surfbright_avg, e_surfbright_avg = mag_to_surfbright(mag_avg, e_mag_avg, pix_area_ang)  # AB mag/ss (/pix)
    surfbright_tot, e_surfbright_tot = mag_to_surfbright(mag_tot, e_mag_tot, area_phys, e_area_phys)  # AB mag/ss (/pix)

    surfdens_phys_avg, e_surfdens_phys_avg = surfbright_to_astro(surfbright_avg, e_surfbright_avg)  # Lsol/pc^2

    outdict = {'sma_pix': isos.sma, 'sma_ang': sma_ang, 'sma_phys': sma_phys,
               'mag_avgs': [mag_avg, e_mag_avg], 'mag_tots': [mag_tot, e_mag_tot],
               'surfdens_avgs': [surfdens_avg, e_surfdens_avg], 'surfdens_tots': [surfdens_tot, e_surfdens_tot],
               'surfbright_avgs': [surfbright_avg, e_surfbright_avg], 'surfbright_tots': [surfbright_tot, e_surfbright_tot],
               'surfdens_phys_avgs': [surfdens_phys_avg, e_surfdens_phys_avg],
               'bkgds': [bkgd, mag_bkgd, surfdens_bkgd, surfbright_bkgd, surfdens_phys_bkgd],
               'isolist': isos}

    return outdict

def king(r, k, rc, rt):
    # units of k depend on units of input y
    d1 = np.sqrt(1 + (r / rc) ** 2)
    d2 = np.sqrt(1 + (rt / rc) ** 2)
    return k * (1 / d1 - 1 / d2) ** 2

def king_calc(rc, rt):
    nom1, nom2 = np.sqrt(0.5), 1 - np.sqrt(0.5)
    den1, den2 = 1, np.sqrt(1 + (rt / rc) ** 2)
    FWHM = 2 * rc * np.sqrt( ( nom1 / den1 + nom2 / den2 ) ** 2 - 1 )
    return FWHM

def gauss(x, A, mu, sig):
    return A * np.exp(- 0.5 * ((x - mu) / sig) ** 2)

def king_fitter(x, y, ey, nX=100, guess=None, bounds=None):
    if len(y[np.isfinite(y)]) == 0:
        guess = [1, 1, 1]
        bounds = [3*(-np.inf,), 3*(np.inf,)]
    else:
        guess = guess if guess is not None else [2 * np.nanmax(y), 1.5 * np.nanmin(x), np.nanmax(x)]
        bounds = bounds if bounds is not None else [(1e-1, 1e-5, 1e-5),
                                                    (2 * np.nanmax(y), np.nanmax(x), 3 * np.nanmax(x))]
    try:
        pars, pars_cov = curve_fit(king, x, y, sigma=ey, p0=guess, bounds=bounds, check_finite=False)
    except:  # handling broken fits
        pars = np.full(len(guess), np.nan)
        pars_cov = np.full((len(guess),) * 2, np.nan)

    errs = np.sqrt(np.diag(pars_cov))
    xfit = np.linspace(np.nanmin(x), np.nanmax(x), nX)
    yfit = king(xfit, *pars)
    return pars, errs, xfit, yfit

def gauss_fitter(x, y, ey, nX=100, guess=None, bounds=None):
    if len(y[np.isfinite(y)]) == 0:
        guess = [1, 1, 1]
        bounds = [3*(-np.inf,), 3*(np.inf,)]
    else:
        guess = guess if guess is not None else [1.1 * np.nanmax(y), 1e-1, np.nanmax(x)]
        bounds = bounds if bounds is not None else [(1e-1, 1e-5, 1e-5),
                                                    (1.5 * np.nanmax(y), 0.5 * np.nanmax(x), 3 * np.nanmax(x))]
    try:
        pars, pars_cov = curve_fit(gauss, x, y, sigma=ey, p0=guess, bounds=bounds, check_finite=False)
    except:  # handling broken fits
        pars = np.full(len(guess), np.nan)
        pars_cov = np.full((len(guess),) * 2, np.nan)

    errs = np.sqrt(np.diag(pars_cov))
    xfit = np.linspace(np.nanmin(x), np.nanmax(x), nX)
    yfit = gauss(xfit, *pars)
    return pars, errs, xfit, yfit

def make_mask(x, y, criterion, test='gtr'):
    mask = y > criterion if test.lower() == 'gtr' else y < criterion
    if len(x[mask]) == 0:
        mask = x.size * [True] if x.size > 0 else True
    return mask

def make_lim(y, lim_type='min', amp=1, offset=0):
    func = np.nanmin if lim_type.lower() == 'min' else np.nanmax
    if len(y[np.isfinite(y)]) == 0:
        return -100 if lim_type.lower() == 'min' else 100
    else:
        return amp * func(y) + offset

def eargs(color='mediumorchid', fmt='.'):
    return dict(fmt=fmt, marker='o', markeredgecolor='black',
                markeredgewidth=1.5, ecolor='black',
                elinewidth=1, capsize=4, color=color)

def annotate_args(pars, errs, color='mediumorchid'):
    rc_str = r'$r_c = %.1f^{\prime\prime} \pm %.1f^{\prime\prime}$' % (pars[1], errs[1])
    rt_str = r'$r_t = %.1f^{\prime\prime} \pm %.1f^{\prime\prime}$' % (pars[2], errs[2])
    #rh =
    label = rc_str + '\n' + rt_str
    kwargs = {'xy': (0.1, 0.12), 'xycoords': 'axes fraction', 'ha': 'left', 'va': 'bottom', 'fontsize': 'small',
              'bbox': {'facecolor': 'white', 'edgecolor': color, 'linewidth': 1.5}}
    return label, kwargs

def annotate_garg(pars, errs, color='limegreen'):
    rc_str = r'$r_c = %.1f^{\prime\prime} \pm %.1f^{\prime\prime}$' % (pars[1], errs[1])
    rt_str = r'$r_t = %.1f^{\prime\prime} \pm %.1f^{\prime\prime}$' % (pars[2], errs[2])
    label = rc_str + '\n' + rt_str
    kwargs = {'xy': (0.1, 0.12), 'xycoords': 'axes fraction', 'ha': 'left', 'va': 'bottom', 'fontsize': 'small',
              'bbox': {'facecolor': 'white', 'edgecolor': color, 'linewidth': 1.5}}
    return label, kwargs

# # # --------------
# # # global imports
# # # --------------

# APOGEE
path_full = '/Users/luca/Documents/Research/M33/'
path_apogee = path_full + f'Data/APOGEE_M33_{stars}/'
file_allStar = path.join(path_apogee, f'APOGEE_M33_{stars}_allStarRVs.txt')

# Pan-STARRS DR1
path_ps1 = path_apogee + 'FITS/PanSTARRSDR1/'
file_ps1 = path.join(path_ps1, '{}_PS1_r.fits')

path_save = path_full + f'Analysis/Plots/APOGEE_M33_{stars}_RadialProfiles/'
# # # ---------
# # # execution
# # # ---------

if __name__ == '__main__':
    # APOGEE: allStar
    table_allStar = ascii.read(file_allStar)

    for name in tqdm(table_allStar['APOGEE_ID']):
        # -------
        # imports
        # -------
        # APOGEE: allVisit & allStar & aspcapStar (& apStar)
        sub_allStar = table_allStar[table_allStar['APOGEE_ID'] == name]
        # Pan-STARRS DR1
        ps1_img, ps1_wcs, ps1_hdr = ps1_file_puller(name, file_ps1)

        # ---------------
        # ellipse fitting
        # ---------------
        crds_pix = ps1_wcs.world_to_pixel_values([[sub_allStar['RA'][0], sub_allStar['DEC'][0]]])[0]
        geometry = EllipseGeometry(x0=crds_pix[0], y0=crds_pix[1], sma=5, eps=0.01, pa=0.01)
        ellipse = Ellipse(ps1_img, geometry)
        for max_grad_error in np.arange(0.3, 0.9, 0.1):
            # try a range of convergence criteria
            # stopping when the first smallest `maxgerr` yields a good fit
            try:
                # first attempt basic fit
                isos = ellipse.fit_image(fix_center=True, linear=True, step=1, minsma=0.5, maxgerr=max_grad_error)
                # account for `successful` fits with 0 ellipses
                if len(isos.intens[np.isfinite(isos.intens)]) == 0:
                    if max_grad_error > 0.8:
                        # if ALL fits fail, replace with nans
                        samples = [EllipseSample(ps1_img, sma=s, geometry=geometry) for s in range(1, 25)]
                        for sample in samples:
                            sample.update(geometry.fix)
                        isos = IsophoteList([Isophote(sample, 0, True, stop_code=4) for sample in samples])
                    else:
                        # otherwise, try again
                        del isos
                        print(max_grad_error)
                        continue
                else:
                    # once successful fit is found, stop loop
                    break
            except:
                # if fit fails, try again
                print('\nfit failed, with max(gerr) == %.1f\n' % max_grad_error)

        # extract computed values
        result = ps1_ellipses(isos, ps1_img, ps1_hdr, ps1_wcs)  # dictionary!!!
        sma, [mag, emag], [fdens, efdens], [mu, emu], [sdens, esdens],\
        [_, mbkgd, fbkgd, mubkgd, sbkgd] = [result[k] for k in ['sma_ang', 'mag_avgs', 'surfdens_avgs',
                                                                'surfbright_avgs', 'surfdens_phys_avgs', 'bkgds']]

        # --------------
        # plot execution
        # --------------
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
        (ax_mag, ax_flux, ax_pans), (ax_magss, ax_phys, ax_ell) = axes
        # sma vs mag
        m = make_mask(sma, mag, mbkgd, 'less')
        pars, errs, xfit, yfit = king_fitter(sma[m], 10**(-(mag[m]-25)/2.5), (np.log(10) / 2.5) * 10**(-(mag[m]-25)/2.5) * emag[m])
        gpar, gerr, gxfi, gyfi = gauss_fitter(sma[m], 10**(-(mag[m]-25)/2.5), (np.log(10) / 2.5) * 10**(-(mag[m]-25)/2.5) * emag[m])
        ax_mag.fill_between([0, 2 * np.nanmax(sma)], [100, 100], [mbkgd, mbkgd],
                            color='black', edgecolor='black', alpha=0.2, label='', zorder=0)
        ax_mag.plot([pars[1], pars[1]], [1, 99], color='black', linestyle='dotted', linewidth=2, zorder=2)
        ax_mag.plot([pars[2], pars[2]], [1, 99], color='black', linestyle='dotted', linewidth=2, zorder=2)
        ax_mag.errorbar(sma, mag, yerr=emag, **eargs('mediumorchid'))
        ax_mag.invert_yaxis()
        ax_mag.set(xscale='log', xlim=(0.8 * np.nanmin(sma), 1.2 * np.nanmax(sma)), xlabel=r'Semi-Major Axis $\left[ ^{\prime\prime} \right]$',
                   ylim=(make_lim(mag, 'max', 1, 0.5), make_lim(mag, 'min', 1, -0.5)), ylabel=r'$\langle r_{\rm PS1} \rangle~\left[\rm AB~mag \right]$')
        ax_mag.plot(gxfi, -2.5 * np.log10(gyfi) + 25, color='limegreen', linestyle='dashdot', linewidth=2, zorder=2)
        ax_mag.plot(xfit, -2.5 * np.log10(yfit) + 25, color='black', linestyle='dashed', linewidth=3, zorder=2)

        ax_mag.annotate(annotate_args(pars, errs)[0], **annotate_args(pars, errs, 'black')[1])

        # sma vs fdens
        m = make_mask(sma, fdens, fbkgd, 'gtr')
        pars, errs, xfit, yfit = king_fitter(sma[m], fdens[m], efdens[m])
        ax_flux.fill_between([0, 2 * np.nanmax(sma)], [0, 0], [fbkgd, fbkgd],
                            color='black', edgecolor='black', alpha=0.2, label='', zorder=0)
        ax_flux.plot([pars[1], pars[1]], [1, 9e10], color='black', linestyle='dotted', linewidth=2, zorder=2)
        ax_flux.plot([pars[2], pars[2]], [1, 9e10], color='black', linestyle='dotted', linewidth=2, zorder=2)

        ax_flux.errorbar(sma, fdens, yerr=efdens, **eargs('mediumorchid'))
        ax_flux.set(xscale='log', xlim=(0.9, 1.2*np.nanmax(sma)), xlabel=r'Semi-Major Axis $\left[ ^{\prime\prime} \right]$',
                    yscale='log', ylim=(make_lim(fdens, 'min', 0.9, 0), make_lim(fdens, 'max', 1.1, 0)),
                    ylabel=r'$\langle f_{r_{\rm PS1}} \rangle\rm~\left[ \`\`counts\" / s / pc^2 \right]$')
        ax_flux.plot(xfit, yfit, color='black', linestyle='dashed', linewidth=3, zorder=2)

        ax_flux.annotate(annotate_args(pars, errs)[0], **annotate_args(pars, errs, 'black')[1])

        # Pan-STARRS image
        ax_pans.imshow(ps1_img, origin='lower', cmap='bone', norm=ImageNormalize(ps1_img, interval=ZScaleInterval()))
        ax_pans.grid()
        ax_pans.set(xlim=(90, 150), ylim=(90, 150), xlabel=r'$\Delta$RA $\left[ ^{\prime\prime} \right]$', ylabel=r'$\Delta$DEC $\left[ ^{\prime\prime} \right]$')
        tks = [95.64757, 105.38854, 115.12951, 124.87048, 134.61145, 144.35242]; ax_pans.set_xticks(tks); ax_pans.set_yticks(tks)
        tls = [r'$-25$', r'$-15$', r'$-5$', r'$5$', r'$15$', r'$25$']; ax_pans.set_xticklabels(tls); ax_pans.set_yticklabels(tls)

        for iso in isos:
            x0, y0, semi, eps, pa = iso.x0, iso.y0, iso.sma, iso.eps, iso.pa
            aper = EllipticalAperture((x0, y0), semi, (1 - eps) * semi, pa)
            aper.plot(axes=ax_pans, color='mediumorchid', linewidth=0.4)

        # sma vs mu
        m = make_mask(sma, mu, mubkgd, 'less')
        pars, errs, xfit, yfit = king_fitter(sma[m], 10**(-(mu[m]-4.64-21.572)/2.5), (np.log(10) / 2.5) * 10**(-(mu[m]-4.64-21.572)/2.5) * emu[m])
        ax_magss.fill_between([0, 2 * np.nanmax(sma)], [100, 100], [mubkgd, mubkgd],
                              color='black', edgecolor='black', alpha=0.2, label='', zorder=0)
        ax_magss.plot([pars[1], pars[1]], [1, 99], color='black', linestyle='dotted', linewidth=2, zorder=2)
        ax_magss.plot([pars[2], pars[2]], [1, 99], color='black', linestyle='dotted', linewidth=2, zorder=2)

        ax_magss.errorbar(sma, mu, yerr=emu, **eargs('mediumorchid'))
        ax_magss.set(xscale='log', xlim=(0.9, 1.2*np.nanmax(sma)), xlabel=r'Semi-Major Axis $\left[ ^{\prime\prime} \right]$',
                     ylim=(make_lim(mu, 'max', 1, 0.5), make_lim(mu, 'min', 1, -0.5)),
                     ylabel=r'$\langle \mu_{r_{\rm PS1}} \rangle~\left[\rm AB~mag / ss \right]$')
        ax_magss.plot(xfit, -2.5*np.log10(yfit) + 4.64 + 21.572, color='black', linestyle='dashed', linewidth=3, zorder=2)

        ax_magss.annotate(annotate_args(pars, errs)[0], **annotate_args(pars, errs, 'black')[1])

        # sma vs sdens
        m = make_mask(sma, sdens, sbkgd, 'gtr')
        pars, errs, xfit, yfit = king_fitter(sma[m], sdens[m], esdens[m])
        ax_phys.fill_between([0, 2 * np.nanmax(sma)], [0, 0], [sbkgd, sbkgd],
                            color='black', edgecolor='black', alpha=0.2, label='', zorder=0)
        ax_phys.plot([pars[1], pars[1]], [1, 9e10], color='black', linestyle='dotted', linewidth=2, zorder=2)
        ax_phys.plot([pars[2], pars[2]], [1, 9e10], color='black', linestyle='dotted', linewidth=2, zorder=2)

        ax_phys.errorbar(sma, sdens, yerr=esdens, **eargs('mediumorchid'))
        ax_phys.set(xscale='log', xlim=(0.9, 1.2*np.nanmax(sma)), xlabel=r'Semi-Major Axis $\left[ ^{\prime\prime} \right]$',
                    yscale='log', ylim=(make_lim(sdens, 'min', 0.9, 0), make_lim(sdens, 'max', 1.1, 0)),
                    ylabel=r'$\langle \Sigma_{r_{\rm PS1}} \rangle\rm~\left[ L_\odot / pc^2 \right]$')
        ax_phys.plot(xfit, yfit, color='black', linestyle='dashed', linewidth=3, zorder=2)

        ax_phys.annotate(annotate_args(pars, errs)[0], **annotate_args(pars, errs, 'black')[1])

        # sma vs ell / sma vs PA
        ax_ell.errorbar(sma, isos.eps, yerr=isos.ellip_err, linewidth=1, **eargs('darkorange', fmt=''))
        ax_ell.set(xlabel=r'Semi-Major Axis $\left[ ^{\prime\prime} \right]$', ylim=(0, 1))
        ax_ell.set_ylabel(r'Ellipticity', bbox=dict(facecolor='white', pad=2, linewidth=1, edgecolor='darkorange'))
        ax_pa = ax_ell.twinx()
        ax_pa.errorbar(sma, (isos.pa * 180 / np.pi) % 90, yerr=isos.pa_err * (180 / np.pi), linewidth=1, **eargs('dodgerblue', fmt=''))
        ax_pa.set_ylim(0, 100)
        ax_pa.set_ylabel(r'Position Angle $\left[ ^\circ \right]$', rotation=270, labelpad=10,
                         bbox=dict(facecolor='white', pad=2, linewidth=1, edgecolor='dodgerblue'))

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        title = r'%s $|$ $v_{\rm rad} = %.2f \pm %.2f \pm %.2f$ km s$^{-1}$' % (name, sub_allStar['VHELIO'],
                                                                                sub_allStar['VERR'],
                                                                                sub_allStar['VSCATTER'])
        title += r'$|$ $N_{\rm vis}$ = %i $|$ SNR $= %.1f$ $|$ $H = %.1f$ mag' % (sub_allStar['NVISITS'], sub_allStar['SNR'], sub_allStar['H'])
        fig.suptitle(title, y=0.93)
        for ax in [ax_mag, ax_magss, ax_flux, ax_phys]:
            ax.set_xticks([1, 3, 6, 12])
            ax.set_xticklabels(['$%i$' % xt for xt in ax.get_xticks()])

        plt.savefig(path.join(path_save, name + '_radialProfiles.pdf'))
        plt.close()