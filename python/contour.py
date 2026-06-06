#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 02:36 2026

@author: dapec
"""

import numpy as np
import glob
from astropy.io import fits
#import cv2 as cv
import skimage as ski
import astropy.units as u
import matplotlib.pyplot as plt


def contour(telfilts):
    telescope, ref = telfilts
    ref = str(ref)
    telescope = telescope.lower()
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{telescope}st"
    imfilenames = sorted(glob.glob(f'{path}*/*_f*w_*', recursive=True))
    filter = [imfilenames[i][imfilenames[i].index("_f")+2:imfilenames[i].index("w_")] for i in range(len(imfilenames))]
    imfiles = [fits.open(imfilenames[i]) for i in range(len(imfilenames))]

    """imref = imfiles[filter.index(ref)][0].data
    with open(f"{path}{ref}/galfit.feedme", "r") as f:
        line = f.readlines()[10]
        line = line[line.index(")")+2:line.index("#")-3]
        maskvalues = line.split("    ")
        maskvalues = [int(maskvalues[i]) for i in range(len(maskvalues))]

    mask = np.full(imref.shape, False)
    mask[maskvalues[0]:maskvalues[1], maskvalues[2]:maskvalues[3]] = True"""
    index = 3
    imref = fits.open(f"{path}{ref}/{telescope}st{ref}_galindex{index}.fits")[3].data
    #imref[imref < 0.01] = 0
    #imref[imref > 10] = 10
    #imref = cv.convertScaleAbs(imref, alpha=2*16-1)
    #edges = cv.Canny(imref, 0.03, 0.08, apertureSize=7, L2gradient=True)
    #edges = ski.feature.canny(imref, sigma=1, low_threshold=0, high_threshold=0.06, mode="nearest")

    imref_contours = ski.measure.find_contours(imref, 0.0295, fully_connected="high")
    large_contours = [cont for cont in imref_contours if len(cont) > 1000]
    large_contours = large_contours[1:3]
    

    def contour_masking(contours):
        contour_mask = ski.measure.grid_points_in_poly(
                                    np.shape(imref),
                                    contours[2],
                                    binarize=True
                                    )
        hdumask = fits.PrimaryHDU(data=np.invert(contour_mask).astype(int))
        hdumask.writeto(
            f"{path}{ref}/{telescope}st{ref}_contourmask.fits",
            overwrite=True,
            output_verify="fix"
        )
    
    #contour_masking(large_contours)

    np.savetxt("/home/daniel/Documentos/UGR/TFM/imágenes/jwst_contours.txt", large_contours)

    np.savetxt("/home/daniel/Documentos/UGR/TFM/imágenes/jwst_smallcontours", imref_contours)

#def 

    """contour_mask = np.invert(np.loadtxt("contourmask.txt").astype(bool))
    #print(contour_mask.astype(int))
    #contour_mask = np.invert(contour_mask)
    
    #imref_masked = np.where(contour_mask, imref, 0)
    imref_masked = np.ma.masked_array(imref, mask=contour_mask)
    #print(imref_masked)
    contour_median = np.ma.median(imref_masked)
    print("MEDIANA", contour_median)
    imref_masked = imref_masked - contour_median

    hducontour = fits.PrimaryHDU(data=imref_masked.data)
    hducontour.writeto(
        f"{path}{ref}/{telescope}st{ref}_shellless.fits",
        overwrite=True,
        output_verify="fix"
        )"""
    
    
    """galaxias = fits.open(f"{path}{ref}/{telescope}st{ref}_galshellless.fits")[2].data

    print(galaxias)

    imref_resta = imref - galaxias

    fig, ax = plt.subplots(2, 1, figsize=(14, 14))
    maximo = 0.35
    minimo = -0.045
    ax[0].imshow(
        imref,
        vmin=minimo,
        vmax=maximo,
        cmap='gray',
        origin="lower"
        )
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].imshow(
        imref_resta,
        vmin=minimo,
        vmax=maximo,
        cmap='gray',
        origin="lower"
        )
    for cont in imref_contours:
        ax[1].plot(cont[:, 1], cont[:, 0], 'r-', linewidth=0.3)

    #for cont in large_contours:
    #    ax[1].plot(cont[:, 1], cont[:, 0], 'g-', linewidth=0.5)

    for i in [1, 2]:
        ax[1].plot(large_contours[i][:, 1], large_contours[i][:, 0], 'g-', linewidth=1)
    #ax[1].set_title('Original Image')
    ax[1].set_xticks([])
    ax[1].set_yticks([])"""

    """ax[1].imshow(edges, cmap='gray', origin="lower")
    ax[1].set_title('Edge Image')
    ax[1].set_xticks([])
    ax[1].set_yticks([])"""

    #plt.show()


contour(("jw", 277))
#contour(("h", "110"))

# %%
