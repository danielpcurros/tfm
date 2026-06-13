#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 02:36 2026

@author: dapec
"""

import numpy as np
import glob
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column
#import cv2 as cv
import skimage as ski
import astropy.units as u
import matplotlib.pyplot as plt
import pyregion

def saveconts(cont):
    x = []
    y = []
    contnum = []

    for j in range(len(cont)):
        x.extend(cont[j][:,0])
        y.extend(cont[j][:,1])
        contnum.extend([j]*len(cont[j]))
    
    tabla = Table()
    tabla["X"] = x
    tabla["Y"] = y
    tabla['Num_cont'] = contnum

    return tabla

def contour(telfilts):
    telescope, ref = telfilts
    ref = str(ref)
    telescope = telescope.lower()
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{telescope}st"

    index = 3
    imref = fits.open(f"{path}{ref}/{telescope}st{ref}_galindex{index}.fits")[3].data

    imref_contours = ski.measure.find_contours(imref, 0.0295, fully_connected="high")
    imref_contours = [np.flip(cont, axis=1) for cont in imref_contours]
    large_contours = [cont for cont in imref_contours if len(cont) > 1000]
    large_contours = large_contours[1:3]

    
    tab_largecontours = saveconts(large_contours)
    tab_contours = saveconts(imref_contours)

    tab_largecontours.write(
        "/home/daniel/Documentos/UGR/TFM/imágenes/jwst_contours.fits",
        format="fits",
        overwrite=True
        )
    
    tab_contours.write(
        "/home/daniel/Documentos/UGR/TFM/imágenes/jwst_smallcontours.fits",
        format="fits",
        overwrite=True
        )
    
def insertmask(tel, maskinic):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if tel == "jw":
        filter = 277
        N = fits.open(f"{path}{tel}st{filter}/mosaic_rxj2129_nircam_f{filter}w_20mas_drz.fits")[0].header["NAXIS1"]
    elif tel == "h":
        filter = 160
        N = fits.open(f"{path}{tel}st{filter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{filter}w_v1_drz.fits")[0].header["NAXIS1"]

    mask = np.ones((N, N))
    with open(f"{path}{tel}st{filter}/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = int(line[1]) - 1
        xmax = int(line[2])
        ymin = int(line[3]) - 1
        ymax = int(line[4])

    mask[ymin:ymax, xmin:xmax] = np.copy(maskinic)
    return mask

def contour_masking(tel, windowed=True):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if tel == "jw":
        filter = 277
    elif tel == "h":
        filter = 160

    if windowed:
        with open(f"{path}{tel}st{filter}/galfit.feedme") as f:
            line = f.readlines()[10].split()
            N = int(line[2]) - int(line[1]) + 1
            M = int(line[4]) - int(line[3]) + 1
        
        tab_cont = Table.read(f"{path}{tel}st_contours.fits")  
        contours = [
        np.column_stack((tab_cont["Y"][tab_cont["Num_cont"] == i], tab_cont["X"][tab_cont["Num_cont"] == i]))
        for i in range(tab_cont["Num_cont"].max() + 1)
        ]

        conts_para_mask = np.copy(contours[1])
        contour_mask = ski.measure.grid_points_in_poly(
                                    (M, N),
                                    conts_para_mask,
                                    binarize=True
                                    )
        hdumask = fits.PrimaryHDU(data=np.invert(contour_mask).astype(int))
        hdumask.writeto(
            f"{path}{tel}st{filter}/contourmask.fits",
            overwrite=True,
            output_verify="fix"
        )
    else:
        if tel == "jw":
            N = 16384
        elif tel == "h":
            N = 10000

        smallmask = fits.open(f"{path}{tel}st{filter}/contourmask.fits")[0].data
        contour_mask = np.ones((N, N))
        with open(f"{path}hst160/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1]) - 1
            xmax = int(line[2])
            ymin = int(line[3]) - 1
            ymax = int(line[4])

        contour_mask[ymin:ymax, xmin:xmax] = np.copy(smallmask)

        """tab_cont = Table.read(f"{path}{tel}st_contours_full.fits")  
        contours = [
        np.column_stack((tab_cont["Y"][tab_cont["Num_cont"] == i], tab_cont["X"][tab_cont["Num_cont"] == i]))
        for i in range(tab_cont["Num_cont"].max() + 1)
        ] 

        conts_para_mask = np.copy(contours[1])
        contour_mask = ski.measure.grid_points_in_poly(
                                    (N, N),
                                    conts_para_mask,
                                    binarize=True
                                    )"""
        hdumask = fits.PrimaryHDU(contour_mask)
        hdumask.writeto(
            f"{path}{tel}st{filter}/contourmask_full2.fits",
            overwrite=True,
            output_verify="fix"
        )  


def region_masking(filename, tel, filter):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}st{filter}/"
    region = pyregion.open(f"{path}{filename}.reg")
    mascara = region.get_mask(shape=(10000, 10000))
    hdumask = fits.PrimaryHDU(mascara.astype(int))
    hdumask.writeto(
            f"{path}shellmask.fits",
            overwrite=True,
            output_verify="fix"
        )  

region_masking("shellmask3", "h", 160)

def conversor(jwst, hfilter=160, rounding=False):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    wjw = wcs.WCS(fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_drz.fits")[0].header)
    wh = wcs.WCS(fits.open(f"{path}hst{hfilter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{hfilter}w_v1_drz.fits")[0].header)

    world = wjw.wcs_pix2world(jwst, 1)
    hst = wh.wcs_world2pix(world, 1)
    if rounding:
        hst = np.around(hst, 0).astype(int)

    return hst

def conversor_elipses(elip, jwfilter=444):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst{jwfilter}/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = float(line[1])
        ymin = float(line[3])
    
    print("elipse", elip)
    aux = conversor([(elip[0] + xmin, elip[1] + ymin)])
    cdjwst = fits.open(f"{path}jwst{jwfilter}/mosaic_rxj2129_nircam_f{jwfilter}w_20mas_drz.fits")[0].header["CD2_2"]
    cdhst = fits.open(f"{path}hst160/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f160w_v1_drz.fits")[0].header["CD2_2"]

    eje = cdjwst/cdhst*elip[2]

    return np.array([aux[0][0], aux[0][1], eje, elip[3], elip[4]])

def cont_to_hst(filename, windowed=True):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    tab_cont = Table.read(f"{path}jwst_{filename}.fits")

    #print(tab_cont)

    with open(f"{path}jwst277/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = float(line[1])
        ymin = float(line[3])

    cont = [
    np.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i] + xmin, tab_cont["Y"][tab_cont["Num_cont"] == i] + ymin))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

    #print(cont[1])

    with open(f"{path}hst160/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin_hst = float(line[1])
        ymin_hst = float(line[3])
        ref =  np.array([[xmin_hst, ymin_hst]])

    if windowed:
        cont_hst = [conversor(cnt) - np.repeat(ref, repeats=np.shape(cnt)[0], axis=0) for cnt in cont]
        tab_hst = saveconts(cont_hst)
        tab_hst.write(
            f"/home/daniel/Documentos/UGR/TFM/imágenes/hst_{filename}.fits",
            format="fits",
            overwrite=True
            )
    else:
        cont_hst = [conversor(cnt) for cnt in cont]
        tab_hst = saveconts(cont_hst)
        tab_hst.write(
            f"/home/daniel/Documentos/UGR/TFM/imágenes/hst_{filename}_full.fits",
            format="fits",
            overwrite=True
            )




#contour(("jw", 277))
#cont_to_hst("contours", windowed=False)
#contour_masking("h")
#contour_masking("h", False)


def resta(telfilts):
    telescope, filter = telfilts
    filter = str(filter)
    telescope = telescope.lower()
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{telescope}st"
    index = 3
    imbase = fits.open(f"{path}{filter}/{telescope}st{filter}_galindex{index}.fits")[3].data

    imgalaxias = fits.open(f"{path}{filter}/{telescope}st{filter}_galmask.fits")[2].data
    M, N = np.shape(imgalaxias)
    lim_galfit = {"xmin": 380, "xmax": 800, "ymin": 480, "ymax": 980}
    imgalaxias_expand = np.zeros(np.shape(imbase))
    for i in range(M):
        for j in range(N):
            imgalaxias_expand[
                i + lim_galfit["ymin"] - 1,
                j + lim_galfit["xmin"] - 1
                ] = imgalaxias[i, j]
    
    imresta = imbase - imgalaxias_expand
    return imbase, imresta


def flux(telfilts, mascara="simple", chefs=False, bcg=True):
    telescope, filter = telfilts
    filter = str(filter)
    telescope = telescope.lower()
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm"
        
    if bcg and chefs:
        print("No se puede escoger la imagen de chefs con BCG")
        return
    elif chefs:
        imagen = fits.open(f"{path}/{telescope}st{filter}/{telescope}st{filter}_sinbcg_residual.fits")[0].data
    elif bcg:
        imagen = fits.open(f"{path}/{telescope}st{filter}/{telescope}st{filter}_galindex3.fits")[1].data
    else:
        imagen = resta(telfilts)[1]

    def fluxmask(modo):
        mask = fits.open(f"{path}/jwst277/jwst277_contourmask.fits")[0].data
        
        if modo == "simple":
            lims = (680, 740)
            mask[:lims[0], :] = 1
            mask[lims[1]:, :] = 1
        elif modo == "none":
            mask = np.copy(mask)
        else:
            elip = np.loadtxt(f"{path}/{mascara}.txt")
            for i in elip:
                #print("VECTOR", i)
                mask[ski.draw.ellipse(i[1], i[0], i[2]*i[3], i[2], rotation=i[4])] = 1

        return mask
    
    mask = fluxmask(mascara)
    pixelscale = fits.open(f"{path}/{telescope}st{filter}/mosaic_rxj2129_nircam_f{filter}w_20mas_drz.fits")[0].header["CD2_2"]*3600
    pixarsr = ((pixelscale*u.arcsec)**2.).to(u.steradian).value
    imagen_mask = np.ma.masked_array(imagen, mask=mask)
    imagen_mask = imagen_mask*pixarsr
    flujo = imagen_mask.sum()
    
    return imagen, mask, flujo

def hstflux(e, photflam): return e*photflam 

def colorindex(f1, f2, mascara="elipses3", cam="acs", ymin=4700, ymax=5300, xmin=4650, xmax=5400):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if cam == "acs":
        tel = "h"
        file1 = fits.open(f"{path}{cam}{f1}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f1}w_v1_drz.fits")[0]
        file2 = fits.open(f"{path}{cam}{f2}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f2}w_v1_drz.fits")[0]

    photflam1 = file1.header["PHOTFLAM"]
    photflam2 = file2.header["PHOTFLAM"]

    img1 = file1.data[ymin:ymax, xmin:xmax]
    img2 = file2.data[ymin:ymax, xmin:xmax]

    print(img1[300, 360], img2[300, 360])
    print(photflam1, photflam2)
    print(hstflux(img1[300, 360], photflam1), hstflux(img2[300, 360], photflam2))

    img1 = np.ma.masked_less_equal(img1, 0)
    img2 = np.ma.masked_less_equal(img2, 0)
    img1[img2 <= 0] = np.ma.masked
    img2[img1 <= 0] = np.ma.masked

    elipse = np.loadtxt(f"{path}{mascara}.txt")
    for elip in elipse:
        elipconv = conversor_elipses(elip)
        elipconv = elipconv - np.array([xmin, ymin, 0, 0, 0])
        img1[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = np.ma.masked
        img2[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = np.ma.masked

    mag1 = -2.5*np.ma.log10(hstflux(img1, photflam1))
    mag2 = -2.5*np.ma.log10(hstflux(img2, photflam2))

    """fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    maximo = 0.08
    minimo = 0
    ax.imshow(
        img2,
        vmin=minimo,
        vmax=maximo,
        cmap='gray',
        origin="lower"
        )"""

    return mag1 - mag2

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


#contour(("jw", 277))
#contour(("h", "110"))


"""jwfiltros = [115, 150, 200, 277, 356, 444]
jwfiltros_repr = []
flujo_parcial = np.array([])
flujo_full = np.array([])
flujo_full2 = np.array([])
flujo_full3 = np.array([])
for i in jwfiltros:
    flujo_parcial = np.append(flujo_parcial, flux(("jw", i))[2])
    flujo_full = np.append(flujo_full, flux(("jw", i), "elipses")[2])
    flujo_full2 = np.append(flujo_full2, flux(("jw", i), "elipses2")[2])
    flujo_full3 = np.append(flujo_full3, flux(("jw", i), "elipses3")[2])
    jwfiltros_repr.append(f"F{i}W")

#flujo_parcial_chefs = flux(("jw", 277), chefs=True)[2]
#flujo_full_chefs = flux(("jw", 277), "elipses", chefs=True)[2]
#flujo_full2_chefs = flux(("jw", 277), "elipses2", chefs=True)[2]
#flujo_fulltotal_chefs = flux(("jw", 277), "none", chefs=True)[2]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(jwfiltros_repr, flujo_full/flujo_parcial, ".-", color="blue", label="conservador")
ax.plot(jwfiltros_repr, flujo_full2/flujo_parcial, ".-", color="red", label="agresivo")
ax.plot(jwfiltros_repr, flujo_full3/flujo_parcial, ".-", color="green", label="conservador 2")

#ax.plot(jwfiltros_repr[-3], flujo_full_chefs/flujo_parcial_chefs, ".-", color="green", label="conservador chefs")
#ax.plot(jwfiltros_repr[-3], flujo_full2_chefs/flujo_parcial_chefs, ".-", color="pink", label="agresivo chefs")
#ax.plot(jwfiltros_repr[-3], flujo_fulltotal_chefs/flujo_parcial_chefs, ".-", color="orange", label="total chefs")

ax.legend()
ax.set_title("Flujo total/flujo pequeño")
#ax.set_ylim(0.9,1.15)"""

"""fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

ax2.plot(jwfiltros_repr, flujo_parcial, ".-", color="black", label="parcial")
ax2.plot(jwfiltros_repr, flujo_full, ".-", color="blue", label="conservador")
ax2.plot(jwfiltros_repr, flujo_full2, ".-", color="red", label="agresivo")
ax2.plot(jwfiltros_repr, flujo_full3, ".-", color="green", label="conservador 2")
#ax2.plot(jwfiltros_repr[-3], flujo_parcial_chefs, ".-", color="cyan", label="parcial chefs")
#ax2.plot(jwfiltros_repr[-3], flujo_full_chefs, ".-", color="green", label="conservador chefs")
#ax2.plot(jwfiltros_repr[-3], flujo_full2_chefs, ".-", color="pink", label="agresivo chefs")
#ax2.plot(jwfiltros_repr[-3], flujo_fulltotal_chefs, ".-", color="orange", label="total chefs")
ax2.legend()"""

#ax[1].plot(jwfiltros_repr, flujo_parcial/np.max(flujo_parcial), ".-", color="red", label="parcial")
#ax[1].plot(jwfiltros_repr, flujo_full/flujo_parcial, ".-", color="blue", label="total/parcial")
#ax[1].legend()


"""fig, ax = plt.subplots(2, 1, figsize=(8, 8))
""ax[0].plot(jwfiltros_repr, flujo_parcial, ".-", color="red", label="parcial")
ax[0].plot(jwfiltros_repr, flujo_full, ".-", color="blue", label="total")
ax[0].legend()""

#ax[1].plot(jwfiltros_repr, flujo_parcial/np.max(flujo_parcial), ".-", color="red", label="parcial")
ax[1].plot(jwfiltros_repr, flujo_full/flujo_parcial, ".-", color="blue", label="total/parcial")
ax[1].legend()"""

"""filt = 444
path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/jwst{filt}/"
imbase, mask, a = flux(("jw", filt), "elipses3")
img = fits.open(f"{path}jwst{filt}_galindex3.fits")[3].data"""

tel = "h"
filter = 160
#path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}st{filter}/"
#imbase = fits.open(f"{path}{tel}st{filter}_galindex3.fits")[3].data
#mask = fits.open(f"{path}contourmask.fits")[0].data
"""
tab_cont = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}st_contours.fits")  
large_contours = [
    np.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i], tab_cont["Y"][tab_cont["Num_cont"] == i]))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]"""

"""fig2, ax2 = plt.subplots(2, 1, figsize=(12, 12))
maximo = 0.1
minimo = 0
ax2[0].imshow(
    img,
    vmin=minimo,
    vmax=maximo,
    cmap='gray',
    origin="lower"
    )

#for cont in large_contours:
#    ax2[0].plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)

ax2[0].set_xticks([])
ax2[0].set_yticks([])

ax2[0].imshow(
    mask,
    #vmin=minimo,
    #vmax=maximo,
    cmap='gray_r',
    origin="lower", 
    alpha=0.3
    )

ax2[1].imshow(
    mask,
    #vmin=minimo,
    #vmax=maximo,
    cmap='gray_r',
    origin="lower",
    )

ax2[1].set_xticks([])
ax2[1].set_yticks([])
ax2[0].set_ylim(400, 1000)
ax2[0].set_xlim(350, 800)"""

"""x1 = 4450
x2 = 5500
y1 = 4200
y2 = 5700
imbase3 = fits.open(f"{path}hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{filter}w_v1_drz.fits")[0].data[x1:x2, y1:y2]
#mask3 = fits.open(f"{path}contourmask_full2.fits")[0].data[x1:x2, y1:y2]
mask3 = fits.open(f"{path}shellmask.fits")[0].data[x1:x2, y1:y2]

tab_contfull = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}st_contours_full.fits")  
large_contoursfull = [
    np.column_stack((tab_contfull["X"][tab_contfull["Num_cont"] == i] - y1, tab_contfull["Y"][tab_contfull["Num_cont"] == i] - x1))
    for i in range(tab_contfull["Num_cont"].max() + 1)
    ]

#print(tab_cont)
#print(large_contoursfull)

fig3, ax3 = plt.subplots(2, 1, figsize=(8, 8))
maximo = 0.3
minimo = 0

"""#ax3[0].set_xlim(4200, 5500)
#ax3[0].set_ylim(4650, 5600)
#ax3[1].set_xlim(4200, 5500)
#ax3[1].set_ylim(4650, 5600)

"""ax3[0].imshow(
    imbase3,
    vmin=minimo,
    vmax=maximo,
    cmap='gray',
    origin="lower"
    )

for cont in large_contoursfull:
    ax3[0].plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)

ax3[0].set_xticks([])
ax3[0].set_yticks([])

ax3[0].imshow(
    mask3,
    #vmin=minimo,
    #vmax=maximo,
    cmap='gray',
    origin="lower", 
    alpha=0.2
    )

ax3[1].imshow(
    mask3,
    #vmin=minimo,
    #vmax=maximo,
    cmap='gray',
    origin="lower",
    )

ax3[1].set_xticks([])
ax3[1].set_yticks([])"""

metalsimple = colorindex(435, 606)
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
maximo = 1
minimo = 0.7
current_cmap = plt.colormaps['rainbow'].copy()
current_cmap.set_bad(color='black')
im4 = ax4.imshow(
    metalsimple,
    vmin=minimo,
    vmax=maximo,
    cmap=current_cmap,
    origin="lower"
    )

cbar = plt.colorbar(im4, ax=ax4)
#fig4.colorbar(figura)
#for cont in large_contours:
#    ax2[0].plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)

ax4.set_xticks([])
ax4.set_yticks([])

plt.show()

#plt.show()
#ax2[0].set_ylim(400, 1000)
#ax2[0].set_xlim(350, 800)
# %%
