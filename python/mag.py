#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:18 2026

@author: dapec
"""

import numpy
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt 
from astropy.table import Table

path = "/home/daniel/Aplicacións/GALFIT/files/tfm/"
filter = "444"
tel = "jw"
if tel == "jw":
    imfile = f"{path}{tel}st{filter}/mosaic_rxj2129_nircam_f{filter}w_20mas_drz.fits"
elif tel == "h":
    imfile = f"{path}{tel}st{filter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{filter}w_v1_drz.fits"

sum = 43

hduim = fits.open(imfile)
head = hduim[0].header
pixelscale = head['CD2_2']*3600.
corr = ((pixelscale*u.arcsec)**2.).to(u.steradian)
#pixelscale = head["PIXAR_SR"]
if tel == "jw":
    zp = -6.10-2.5*numpy.log10(corr.value)
elif tel == "h":
    with open(f"{path}{tel}st{filter}/galfit.feedme") as f:
        zp = float(f.readlines()[12].split()[1])

flux = numpy.copy(sum)#*pixelscale#/head["TELAPSE"]#*head["PIXAR_SR"]
mag =-2.5*numpy.log10(flux)+zp
print(mag, zp, head["CD2_2"]*3600)


hdumuse = fits.open(f"{path}muse/outcube.fits")
cubo = hdumuse[1].data


hdujwst = fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_drz.fits")
hduhst = fits.open(f"{path}hst105/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f105w_v1_drz.fits")
hdujwstrms = fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_wht_rms.fits")
#print(hduhst[0].header["*GAIN"])
tab_cont = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/muse_contours.fits")  
print(hdumuse[0].header["*GAIN*"])
contours = [
    numpy.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i], tab_cont["Y"][tab_cont["Num_cont"] == i] ))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

plt.imshow(
    cubo[1500],
    cmap="gray",
    origin="lower"
    )

for cont in contours:
    plt.plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)
"""index = "3"
hdugal = fits.open(f"{path}jwst{filter}/jwst{filter}_galindex{index}.fits")
for i in range(len(hdugal)):
    hdugal[i].writeto(
        f"{path}jwst{filter}/jwst{filter}_galindex{index}_{i}c.fits",
        overwrite=True,
        output_verify="fix"
        )"""

"""index = "3"
hdugal = fits.open(f"{path}jwst{filter}/jwst{filter}_galindex{index}.fits")
hdugal[3].writeto(
        f"{path}jwst{filter}/jwst{filter}_sinbcg.fits",
        overwrite=True,
        output_verify="fix"
        )"""
"""path2 = "/home/daniel/Aplicacións/GALFIT/files/EXAMPLE/"
hduex = fits.open(f"{path2}imgblock.fits")
for i in range(len(hduex)):
    hduex[i].writeto(
        f"{path2}imgblock{i}.fits",
        overwrite=True,
        output_verify="fix"
        )"""

# %%
