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
from astropy.table import Table
#import cv2 as cv
import skimage as ski
import astropy.units as u
import matplotlib.pyplot as plt
import pyregion
from scipy.ndimage import gaussian_filter
import fotometria_HST_JWST as fot

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

def contour_masking(tel, windowed=True, numcont=1):
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

        conts_para_mask = np.copy(contours[numcont])
        contour_mask = ski.measure.grid_points_in_poly(
                                    (M, N),
                                    conts_para_mask,
                                    binarize=True
                                    )
        hdumask = fits.PrimaryHDU(data=np.invert(contour_mask).astype(int))
        hdumask.writeto(
            f"{path}{tel}st{filter}/contourmask_{numcont}.fits",
            overwrite=True,
            output_verify="fix"
        )
    else:
        if tel == "jw":
            N = 16384
        elif tel == "h":
            N = 10000

        smallmask = fits.open(f"{path}{tel}st{filter}/contourmask_{numcont}.fits")[0].data
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
            f"{path}{tel}st{filter}/contourmask_{numcont}_full2.fits",
            overwrite=True,
            output_verify="fix"
        )

#contour_masking("h", numcont=0, windowed=True)  

def contour_maskingmuse(tel="muse", numcont=1):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
        
    tab_cont = Table.read(f"{path}{tel}_contours.fits")  
    contours = [
    np.column_stack((tab_cont["Y"][tab_cont["Num_cont"] == i], tab_cont["X"][tab_cont["Num_cont"] == i]))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

    #headmuse = fits.open(f"{path}muse/ADP.2017-12-14T12_30_03.217.fits")[1].header
    headmuse = fits.open(f"{path}muse/outcube.fits")[1].header
    M = headmuse["NAXIS2"]
    N = headmuse["NAXIS1"]

    conts_para_mask = np.copy(contours[numcont])
    contour_mask = ski.measure.grid_points_in_poly(
                                (M, N),
                                conts_para_mask,
                                binarize=True
                                )
    hdumask = fits.PrimaryHDU(data=np.invert(contour_mask).astype(int))
    hdumask.writeto(
        f"{path}{tel}/contourmask_{numcont}.fits",
        overwrite=True,
        output_verify="fix"
    )

def rms_masking():
    path =  f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    mask = fits.open(f"{path}hst160/contourmask_1.fits")[0].data
    maskcopy = np.zeros(np.shape(mask))

    with open(f"{path}hst160/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1]) - 1
            xmax = int(line[2])
            ymin = int(line[3]) - 1
            ymax = int(line[4])
    filtros = [435, 475, 606, 775]
    for filt in filtros:
        bg_rms = fits.open(f"{path}acs{filt}/hlsp_clash_hst_acs-30mas_rxj2129_f{filt}w_v1_wht_rms.fits")[0].data[ymin:ymax,xmin:xmax]
        mask[bg_rms > 1e2] = 1
        maskcopy[bg_rms > 1e2] = 1
    
    maskcopy[600:,:] = 0
    maskcopy[:200,:] = 0
    maskcopy[:,:500] = 0
    maskcopy[:,850:] = 0
    
    hdumask = fits.PrimaryHDU(mask)
    hdumaskcopy = fits.PrimaryHDU(maskcopy)
    hdumask.writeto(
        f"{path}hst160/contourmask_1.fits",
        overwrite=True
    )
    hdumaskcopy.writeto(
        f"{path}hst160/contourmask_1_rms.fits",
        overwrite=True
    )
    

#rms_masking()   

def conversorhstmuse(hst, rounding=False):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    wh = wcs.WCS(fits.open(f"{path}hst160/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f160w_v1_drz.fits")[0].header)
    wmuse = wcs.WCS(fits.open(f"{path}muse/outcube.fits")[1].header)
    
    world = wh.wcs_pix2world(hst, 1)
    muse = wmuse.wcs_world2pix([[world[i][0], world[i][1], 0] for i in range(len(world))], 1)

    if rounding:
        muse = np.around(muse, 0).astype(int)
    return np.array([[muse[i][0], muse[i][1]] for i in range(len(muse))])

def rms_maskingmuse():
    path =  f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    mask = fits.open(f"{path}hst160/contourmask_1_rms.fits")[0].data

    with open(f"{path}hst160/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xminhst = int(line[1])
            yminhst = int(line[3])

    coords = np.where(mask == 1)
    coordsy = coords[0] + yminhst
    coordsx = coords[1] + xminhst

    coords = np.array([[coordsx[i], coordsy[i]] for i in range(len(coordsy))])

    print(coords)

    coords_muse = conversorhstmuse(coords, rounding=True)

    print(coords_muse)
    
    #coords_jwst2 = np.array([[coords_muse[i,1], coords_muse[i,0]] for i in range(np.shape(coords_muse)[0])])

    print(coords_muse)
    mask_muse = fits.open(f"{path}muse/contourmask_1.fits")[0].data
    #mask_jwst = np.ones(np.shape(mask_jwst_inic))
    for i in range(np.shape(coords_muse)[0]):
        mask_muse[coords_muse[i,0], coords_muse[i,1]] = 1

    hdumask = fits.PrimaryHDU(mask_muse)
    hdumask.writeto(
        f"{path}muse/contourmask_1prueba.fits",
        overwrite=True
    )

#rms_maskingmuse()

def rms_maskingjwst():
    path =  f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    mask = fits.open(f"{path}hst160/contourmask_1_rms.fits")[0].data

    with open(f"{path}hst160/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xminhst = int(line[1])
            yminhst = int(line[3])

    coords = np.where(mask == 1)
    coordsy = coords[0] + yminhst
    coordsx = coords[1] + xminhst

    coords = np.array([[coordsx[i], coordsy[i]] for i in range(len(coordsy))])

    print(coords)

    coords_jwst = conversor_rev(coords, rounding=True)
    with open(f"{path}jwst277/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xminjwst = int(line[1])
            yminjwst = int(line[3])
    
    print(coords_jwst)
    
    coords_jwst2 = np.array([[coords_jwst[i,1]-yminjwst, coords_jwst[i,0]-xminjwst] for i in range(np.shape(coords_jwst)[0])])

    print(coords_jwst2)
    mask_jwst = fits.open(f"{path}jwst277/contourmask_1.fits")[0].data
    #mask_jwst = np.ones(np.shape(mask_jwst_inic))
    for i in range(np.shape(coords_jwst2)[0]):
        mask_jwst[coords_jwst2[i,0], coords_jwst2[i,1]] = 1

    hdumask = fits.PrimaryHDU(mask_jwst)
    hdumask.writeto(
        f"{path}jwst277/contourmask_1.fits",
        overwrite=True
    )

#rms_maskingjwst()
#contour_maskingmuse()

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

#region_masking("shellmask3", "h", 160)

def conversor(jwst, hfilter=160, rounding=False):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    wjw = wcs.WCS(fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_drz.fits")[0].header)
    wh = wcs.WCS(fits.open(f"{path}hst{hfilter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{hfilter}w_v1_drz.fits")[0].header)

    world = wjw.wcs_pix2world(jwst, 1)
    hst = wh.wcs_world2pix(world, 1)
    if rounding:
        hst = np.around(hst, 0).astype(int)

    return hst

def conversor_rev(hst, hfilter=160, rounding=False):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    wjw = wcs.WCS(fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_drz.fits")[0].header)
    wh = wcs.WCS(fits.open(f"{path}hst{hfilter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{hfilter}w_v1_drz.fits")[0].header)

    world = wh.wcs_pix2world(hst, 1)
    jwst = wjw.wcs_world2pix(world, 1)
    if rounding:
        jwst = np.around(jwst, 0).astype(int)

    return jwst

def conversor_elipses(elip, jwfilter=444):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst{jwfilter}/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = float(line[1])
        ymin = float(line[3])
    
    aux = conversor([(elip[0] + xmin, elip[1] + ymin)])
    cdjwst = fits.open(f"{path}jwst{jwfilter}/mosaic_rxj2129_nircam_f{jwfilter}w_20mas_drz.fits")[0].header["CD2_2"]
    cdhst = fits.open(f"{path}hst160/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f160w_v1_drz.fits")[0].header["CD2_2"]

    eje = cdjwst/cdhst*elip[2]

    return np.array([aux[0][0], aux[0][1], eje, elip[3], elip[4]])

def conversormuse_elipses(elip, jwfilter=444):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst{jwfilter}/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = float(line[1])
        ymin = float(line[3])
    
    aux = conversormuse([(elip[0] + xmin, elip[1] + ymin)])
    cdjwst = fits.open(f"{path}jwst{jwfilter}/mosaic_rxj2129_nircam_f{jwfilter}w_20mas_drz.fits")[0].header["CD2_2"]
    #cdmuse = fits.open(f"{path}muse/ADP.2017-12-14T12_30_03.217.fits")[1].header["CD2_2"]
    cdmuse = fits.open(f"{path}muse/outcube.fits")[1].header["CD2_2"]

    eje = cdjwst/cdmuse*elip[2]

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
        
def conversormuse(jwst):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    wjw = wcs.WCS(fits.open(f"{path}jwst277/mosaic_rxj2129_nircam_f277w_20mas_drz.fits")[0].header)
    #wmuse = wcs.WCS(fits.open(f"{path}muse/ADP.2017-12-14T12_30_03.217.fits")[1].header)
    headmuse = fits.open(f"{path}muse/outcube.fits")[1].header
    wmuse = wcs.WCS(headmuse)

    M = headmuse["NAXIS2"]/headmuse["NAXIS1"]
    
    world = wjw.wcs_pix2world(jwst, 1)
    muse = wmuse.wcs_world2pix([[world[i][0], world[i][1], 0] for i in range(len(world))], 1)
    return np.array([[muse[i][0] + 4, muse[i][1] - 6] for i in range(len(muse))])

def cont_to_muse(filename):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    tab_cont = Table.read(f"{path}jwst_{filename}.fits")

    with open(f"{path}jwst277/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = float(line[1])
        ymin = float(line[3])

    cont = [
    np.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i] + xmin, tab_cont["Y"][tab_cont["Num_cont"] == i] + ymin))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

    cont_muse = [conversormuse(cnt) for cnt in cont]
    tab_muse = saveconts(cont_muse)
    tab_muse.write(
        f"/home/daniel/Documentos/UGR/TFM/imágenes/muse_{filename}.fits",
        format="fits",
        overwrite=True
        )


#contour(("jw", 277))
#cont_to_hst("contours", windowed=False)
#contour_masking("h")
#contour_masking("h", False)
#cont_to_muse("contours")
#print("CONVERSIÓN", conversormuse([[8000, 8000]]))

def shell_properties(numcont, mascara="simple"):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst277/fit.log") as f:
        n = -1
        lines = f.readlines()
        while "galfit.feedme" not in lines[n]:
            n-=1
        line = lines[n+4].split()
        x0 = float(line[3][:-1])
        y0 = float(line[4][:-1])
    tel = "jwst"
    filt = 277
    with open(f"{path}{tel}{filt}/galfit.feedme") as f:
        line = f.readlines()[10].split()
        xmin = int(line[1])
        ymin = int(line[3])
    x0-=xmin
    y0-=ymin
    path = f"{path}{tel}{filt}/"
    mask = fits.open(f"{path}contourmask_{numcont}.fits")[0].data

    if mascara == "simple":
        lims = (680, 740)
        mask[:lims[0], :] = 1
        mask[lims[1]:, :] = 1
    #print(mask[mask == 0])
    area = mask[mask == 0].size
    print(area)

    hduim = fits.open(f"{path}mosaic_rxj2129_nircam_f277w_20mas_drz.fits")
    head = hduim[0].header
    pixelscale = head['CD2_2']*3600.
    #corr = ((pixelscale*u.arcsec)**2.).to(u.steradian)
    areasec = area*pixelscale**2

    maskcoords = np.where(mask == 0)

    rholist = np.array([])
    for i in range(len(mask[0])):
        xinic = maskcoords[1][i] - x0
        yinic = maskcoords[0][i] - y0
        rho = np.sqrt(xinic**2 + yinic**2)
        rholist = np.append(rholist, rho)
    
    rhomedian = np.median(rholist)*pixelscale
    print(areasec, rhomedian)

#shell_properties(1)

    

def rotatemask(tel, numcont=1, mascara="_elipses3", rotacion=3/5):
    if tel == "hst":
        filt = 160
    elif tel == "jwst":
        filt = 277
    elif tel == "muse":
        filt = ""
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst277/fit.log") as f:
        n = -1
        lines = f.readlines()
        while "galfit.feedme" not in lines[n]:
            n-=1
        line = lines[n+4].split()
        x0 = float(line[3][:-1])
        y0 = float(line[4][:-1])

    if tel == "hst":
        [[x0, y0]] = conversor([[x0, y0]])
    elif tel == "muse":
        [[x0, y0]] = conversormuse([[x0, y0]])
        print("MUSE", x0,y0)

    if tel != "muse":
        with open(f"{path}{tel}{filt}/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1])
            ymin = int(line[3])
        
        x0-=xmin
        y0-=ymin
    path = f"{path}{tel}{filt}/"
    mask_inic = fits.open(f"{path}contourmask_{numcont}{mascara}.fits")[0].data

    """fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.imshow(
        fits.open(f"{path}outcube.fits")[1].data[3600],
        vmin=0,
        vmax=30,
        cmap='gray',
        origin="lower",
    )
    ax.imshow(
        mask_inic,
        #vmin=minimo,
        #vmax=maximo,
        cmap='gray',
        origin="lower",
        alpha=0.3
        )
    
    tab_contfull = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}_contours.fits") 
    large_contours = [
    np.column_stack((tab_contfull["X"][tab_contfull["Num_cont"] == i], tab_contfull["Y"][tab_contfull["Num_cont"] == i] ))
    for i in range(tab_contfull["Num_cont"].max() + 1)
    ]
    
    for cont in large_contours:
        ax.plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)"""
    
    mask_rot = np.ones(np.shape(mask_inic))
    mask_inic = np.where(mask_inic == 0)

    #print(x0, y0)
    #print(mask_inic)

    for i in range(len(mask_inic[0])):
        xinic = mask_inic[1][i] - x0
        yinic = mask_inic[0][i] - y0
        rho = np.sqrt(xinic**2 + yinic**2)
        alfa = np.arctan(yinic/xinic)
        if numcont == 1:
            alfa += np.pi
        #rotacion = 0

        xrot = int(np.around(x0 + rho*np.cos(alfa + rotacion*np.pi)))
        yrot = int(np.around(y0 + rho*np.sin(alfa + rotacion*np.pi)))

        mask_rot[yrot, xrot] = 0
    
    """fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.imshow(
        fits.open(f"{path}{tel}{filt}_galindex3.fits")[1].data,
        vmin=0,
        vmax=1,
        cmap='gray',
        origin="lower",
    )
    ax.imshow(
        mask_rot,
        #vmin=minimo,
        #vmax=maximo,
        cmap='gray',
        origin="lower",
        alpha=0.2
        )"""
    
    hdumask = fits.PrimaryHDU(mask_rot)
    hdumask.writeto(f"{path}contourmask_{numcont}{mascara}_vacio.fits", overwrite=True)

#rotatemask("jwst")
#rotatemask("hst", numcont=0, rotacion=1/3, mascara="")
#rotatemask("muse")

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

def rotatemask2(tel, numcont=1, mascara="elipses3", rotacion=3/5):
    if tel == "hst":
        filt = 160
    elif tel == "jwst":
        filt = 277
    elif tel == "muse":
        filt = ""
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"

    with open(f"{path}jwst277/fit.log") as f:
        n = -1
        lines = f.readlines()
        while "galfit.feedme" not in lines[n]:
            n-=1
        line = lines[n+4].split()
        x0 = float(line[3][:-1])
        y0 = float(line[4][:-1])

    if tel == "hst":
        [[x0, y0]] = conversor([[x0, y0]])
    elif tel == "muse":
        [[x0, y0]] = conversormuse([[x0, y0]])

    if tel != "muse":
        with open(f"{path}{tel}{filt}/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1])
            ymin = int(line[3])

            N = int(line[2]) - xmin + 1
            M = int(line[4]) - ymin + 1

        x0 = x0 - xmin
        y0 = y0 - ymin
    else:
        [M, N] = np.shape(fits.open(f"{path}{tel}/outcube.fits")[1].data)

    tab_cont = Table.read(f"{path}{tel}_contours.fits")  
    contour_inic = [
        np.column_stack(((tab_cont["X"][tab_cont["Num_cont"] == i] - xmin), (tab_cont["Y"][tab_cont["Num_cont"] == i] - ymin)))
        for i in range(tab_cont["Num_cont"].max() + 1)
        ][numcont]
    
    
    contour_rot = np.zeros(np.shape(contour_inic))
    for i in range(len(contour_inic)):
        xinic = contour_inic[i, 0] - x0
        yinic = contour_inic[i, 1] - y0
        rho = np.sqrt(xinic**2 + yinic**2)
        alfa = np.arctan(yinic/xinic)
        if numcont == 1:
            alfa += np.pi

        xrot = int(np.around(x0 + rho*np.cos(alfa + rotacion*np.pi)))
        yrot = int(np.around(y0 + rho*np.sin(alfa + rotacion*np.pi)))

        contour_rot[i] = [xrot, yrot]

    print(contour_rot)
    contour_mask = ski.measure.grid_points_in_poly(
                                    (M, N),
                                    contour_rot,
                                    binarize=True
                                    )
    hdumask = fits.PrimaryHDU(data=np.invert(contour_mask).astype(int))
    hdumask.writeto(
            f"{path}{tel}{filt}/contourmask_{numcont}_vacioprueba.fits",
            overwrite=True,
            output_verify="fix"
        )

#rotatemask2("jwst")

def rotatecont(tel="hst", filt=160, ymin=4600, ymax=5400, xmin=4650, xmax=5450, npixbin=4, numcont=1, rotacion=3/5):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst277/fit.log") as f:
        n = -1
        lines = f.readlines()
        while "galfit.feedme" not in lines[n]:
            n-=1
        line = lines[n+4].split()
        x0 = float(line[3][:-1])
        y0 = float(line[4][:-1])

    if tel == "hst":
        [[x0, y0]] = conversor([[x0, y0]])

    """with open(f"{path}{tel}{filt}/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1])
            ymin = int(line[3])"""
        
    x0 = (x0 - xmin)/npixbin
    y0 = (y0 - ymin)/npixbin

    tab_cont = Table.read(f"{path}hst_contours_full.fits")  
    contour_inic = [
        np.column_stack(((tab_cont["X"][tab_cont["Num_cont"] == i] - xmin)/npixbin, (tab_cont["Y"][tab_cont["Num_cont"] == i] - ymin)/npixbin))
        for i in range(tab_cont["Num_cont"].max() + 1)
        ][numcont]
    
    print(np.shape(contour_inic))
    
    contour_rot = np.zeros(np.shape(contour_inic))
    for i in range(len(contour_inic)):
        xinic = contour_inic[i, 0] - x0
        yinic = contour_inic[i, 1] - y0
        rho = np.sqrt(xinic**2 + yinic**2)
        alfa = np.arctan(yinic/xinic)
        if numcont == 1:
            alfa += np.pi

        xrot = int(np.around(x0 + rho*np.cos(alfa + rotacion*np.pi)))
        yrot = int(np.around(y0 + rho*np.sin(alfa + rotacion*np.pi)))

        contour_rot[i] = [xrot, yrot]
    
    return contour_rot

def flux(telfilts, mascara="simple", chefs=False, bcg=True, numcont=1, vacio=False):
    tel, filter = telfilts
    filter = str(filter)
    tel = tel.lower()
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm"
        
    if bcg and chefs:
        print("No se puede escoger la imagen de chefs con BCG")
        return
    elif chefs:
        imagen = fits.open(f"{path}/{tel}{filter}/{tel}{filter}_sinbcg_residual.fits")[0].data
    elif bcg and tel != "muse":
        imagen = fits.open(f"{path}/{tel}{filter}/{tel}{filter}_galindex3.fits")[1].data
    elif bcg and tel == "muse":
        filter = int(filter)
        #imagen = fits.open(f"{path}/{tel}/ADP.2017-12-14T12_30_03.217.fits")[1].data[filter]
        imagen = fits.open(f"{path}/{tel}/outcube.fits")[1].data[filter]
    else:
        imagen = resta(telfilts)[1]
    

    def fluxmask(modo):
        if tel == "muse":
            mask = fits.open(f"{path}/{tel}/contourmask_{numcont}.fits")[0].data
        else:
            mask = fits.open(f"{path}/{tel}{filter}/contourmask_{numcont}.fits")[0].data
        
        if modo == "simple":
            lims = (680, 740)
            if tel == "hst" or tel == "acs":
                with open(f"{path}/jwst277/galfit.feedme") as f:
                    line = f.readlines()[10].split()
                    yminjw = float(line[3])
                limconv = conversor([[8000, lims[0]+yminjw], [8000, lims[1]+yminjw]])
                with open(f"{path}/{tel}{filter}/galfit.feedme") as f:
                    line = f.readlines()[10].split()
                    xminhst = int(line[1])
                    yminhst = int(line[3])
                lims = (int(limconv[0][1]) - yminhst, int(limconv[1][1]) - yminhst)
            elif tel == "muse":
                with open(f"{path}/jwst277/galfit.feedme") as f:
                    line = f.readlines()[10].split()
                    yminjw = float(line[3])
                limconv = conversormuse([[8000, lims[0]+yminjw], [8000, lims[1]+yminjw]])
                lims = (int(limconv[0][1]), int(limconv[1][1]))
            mask[:lims[0], :] = 1
            mask[lims[1]:, :] = 1
        elif modo == "none":
            mask = np.copy(mask)
        else:
            elipse = np.loadtxt(f"{path}/{mascara}.txt")
            if tel == "jwst":
                for i in elipse:
                    mask[ski.draw.ellipse(i[1], i[0], i[2]*i[3], i[2], rotation=i[4])] = 1
            elif tel == "hst" or tel == "acs":
                for elip in elipse:
                    with open(f"{path}/{tel}{filter}/galfit.feedme") as f:
                        line = f.readlines()[10].split()
                        xmin = float(line[1])
                        ymin = float(line[3])

                    elipconv = conversor_elipses(elip)
                    elipconv = elipconv - np.array([xmin, ymin, 0, 0, 0])
                    elipconv[0:3] = elipconv[0:3]
                    mask[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = 1
            elif tel == "muse":
                for elip in elipse:
                    elipconv = conversormuse_elipses(elip)
                    elipconv[0:3] = elipconv[0:3]
                    mask[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = 1
        return mask
    
    if vacio and tel != "muse":
        mask = fits.open(f"{path}/{tel}{filter}/contourmask_{numcont}_{mascara}_vacio.fits")[0].data
    elif vacio and tel == "muse":
        mask = fits.open(f"{path}/{tel}/contourmask_{numcont}_{mascara}_vacio.fits")[0].data
    else:
        mask = fluxmask(mascara)
    """if (tel == "jwst" and filter == "277") or (tel == "hst" and filter == "160"):
        hdumascara = fits.PrimaryHDU(mask)
        hdumascara.writeto(f"{path}/{tel}{filter}/contourmask_{numcont}_{mascara}.fits", overwrite=True)
    elif tel == "muse":
        hdumascara = fits.PrimaryHDU(mask)
        hdumascara.writeto(f"{path}/{tel}/contourmask_{numcont}_{mascara}.fits", overwrite=True)"""
    if tel != "muse":
        with open(f"{path}/{tel}{filter}/galfit.feedme") as f:
            line = f.readlines()[10].split()
            xmin = int(line[1]) - 1
            xmax = int(line[2])
            ymin = int(line[3]) - 1
            ymax = int(line[4])

    print(filter)
    if tel == "jwst":
        bg_rms = fits.open(f"{path}/{tel}{filter}/mosaic_rxj2129_nircam_f{filter}w_20mas_wht_rms.fits")[0].data[ymin:ymax, xmin:xmax]
        head_jwst = fits.open(f"{path}/{tel}{filter}/mosaic_rxj2129_nircam_f{filter}w_20mas_drz.fits")[0].header
        #mask[bg_rms > 1e2] = 1
        filtrosjwst = np.loadtxt(f"{path}/jwstfilters.txt")
        lbda = filtrosjwst[np.where(filtrosjwst == float(filter))[0], 1]
        if lbda < 200:
            flujo, error_flujo, mag, error_mag = fot.main2(
                                                imagen,
                                                head_jwst,
                                                bg_rms,
                                                mask,
                                                tel
                                            )
        else:
            flujo, error_flujo, mag, error_mag = fot.main2(
                                                imagen,
                                                head_jwst,
                                                bg_rms,
                                                mask,
                                                tel,
                                                jwsw=False,
                                            )
    elif tel == "hst":
        bg_rms = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{filter}w_v1_wht_rms.fits")[0].data[ymin:ymax, xmin:xmax]
        head_hst = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_wfc3ir-30mas_rxj2129_f{filter}w_v1_drz.fits")[0].header
        #mask[bg_rms > 1e2] = 1
        flujo, error_flujo, mag, error_mag = fot.main2(
                                                imagen,
                                                head_hst,
                                                bg_rms,
                                                mask,
                                                tel
                                            )
        #photfnu = head_hst["PHOTFNU"]*1e-6
        lbda = head_hst["PHOTPLAM"]
    elif tel == "acs":
        if filter == "850":
            bg_rms = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_{tel}-30mas_rxj2129_f{filter}lp_v1_wht_rms.fits")[0].data[ymin:ymax, xmin:xmax]
            head_acs = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_{tel}-30mas_rxj2129_f{filter}lp_v1_drz.fits")[0].header
        else:
            bg_rms = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_{tel}-30mas_rxj2129_f{filter}w_v1_wht_rms.fits")[0].data[ymin:ymax, xmin:xmax]
            head_acs = fits.open(f"{path}/{tel}{filter}/hlsp_clash_hst_{tel}-30mas_rxj2129_f{filter}w_v1_drz.fits")[0].header
        #mask[bg_rms > 1e2] = 1
        print("BG RMS", bg_rms[bg_rms > 1])
        flujo, error_flujo, mag, error_mag = fot.main2(
                                                imagen,
                                                head_acs,
                                                bg_rms,
                                                mask,
                                                tel
                                            )
        lbda = head_acs["PHOTPLAM"]
        #photfnu = photflam*1e17*(lbda)**2/3e18
    elif tel == "muse":
        #head_muse = fits.open(f"{path}/{tel}/ADP.2017-12-14T12_30_03.217.fits")[1].header
        head_muse = fits.open(f"{path}/{tel}/outcube.fits")[1].header
        pixelscale = head_muse["CD2_2"]*3600
        lbda = head_muse["CRVAL3"] + head_muse["CD3_3"]*filter
        #fluxconv = 1e-3*lbda**2/3e18
        fluxconv = 1e-20*lbda**2/3e18
        imagen = imagen*fluxconv
        mag = fot.main2(
                        imagen,
                        head_muse,
                        region=mask,
                        telescope=tel
                    )
        mag = 10**(-0.4*mag)
        return mag, lbda
        #fluxconv = 1e-20

    #pixarsr = ((pixelscale*u.arcsec)**2.).to(u.steradian).value
    #area = imagen_mask.count()*pixarsr  

    error_mag = 10**(-0.4*mag)*np.log(10)*(0.4)*error_mag#/area
    mag = 10**(-0.4*mag)#/area

    return imagen, mask, mag, lbda, error_mag

#print("CONVERSIÓN", conversor([[680 + 7000, 8000], [740 + 7000, 8000]]))

def hstflux(e, photflam): return e*photflam 

def circulos_color(rho, theta, r, xmin=4650, ymin=4600, flux=False):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    with open(f"{path}jwst277/fit.log") as f:
        n = -1
        lines = f.readlines()
        while "galfit.feedme" not in lines[n]:
            n-=1
        line = lines[n+4].split()
        x0 = float(line[3][:-1])
        y0 = float(line[4][:-1])

    [[x0h, y0h]] = conversor([[x0, y0]])
    x0h-=xmin
    y0h-=ymin
    
    x0circ = x0h + rho*np.cos((theta+90)*np.pi/180)
    y0circ = y0h + rho*np.sin((theta+90)*np.pi/180)

    if flux:
        return ski.draw.ellipse(int(y0circ), int(x0circ), int(r), int(r))
    else:
        return ski.draw.circle_perimeter(int(y0circ), int(x0circ), int(r))

radio = 15
parametros_circ = [
        [200, 70],
        [200, -20],
        [200, 70+180],
        [0, 0],
        [315, -105],
        [315, 160],
        [315, -105+180],
    ]

def colorindex(f1, f2, mascara="elipses3", cam="acs", ymin=4600, ymax=5400, xmin=4650, xmax=5450, write=False, npixbin=4, sigma=0.5, circparam=parametros_circ, circr=radio):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if cam == "acs":
        tel = "h"
        file1 = fits.open(f"{path}{cam}{f1}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f1}w_v1_drz.fits")[0]
        file2 = fits.open(f"{path}{cam}{f2}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f2}w_v1_drz.fits")[0]

    photflam1 = file1.header["PHOTFLAM"]
    photflam2 = file2.header["PHOTFLAM"]

    photplam1 = file1.header["PHOTPLAM"]
    photplam2 = file2.header["PHOTPLAM"]

    zp1 = -5*np.log10(photplam1)-2.408
    zp2 = -5*np.log10(photplam2)-2.408

    img1 = file1.data[ymin:ymax, xmin:xmax]
    img2 = file2.data[ymin:ymax, xmin:xmax]

    nbinsx = (xmax-xmin)//npixbin
    nbinsy = (ymax-ymin)//npixbin

    img1 = img1.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)
    img2 = img2.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)

    img1 = np.ma.masked_less_equal(img1, 0)
    img2 = np.ma.masked_less_equal(img2, 0)
    img1[img2 <= 0] = np.ma.masked
    img2[img1 <= 0] = np.ma.masked

    """elipse = np.loadtxt(f"{path}{mascara}.txt")
    for elip in elipse:
        elipconv = conversor_elipses(elip)
        elipconv = elipconv - np.array([xmin, ymin, 0, 0, 0])
        elipconv[0:3] = elipconv[0:3]/npixbin
        img1[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = np.ma.masked
        img2[ski.draw.ellipse(elipconv[1], elipconv[0], elipconv[2]*elipconv[3], elipconv[2], rotation=elipconv[4])] = np.ma.masked"""

    mag1 = -2.5*np.ma.log10(hstflux(img1, photflam1)) + zp1
    mag2 = -2.5*np.ma.log10(hstflux(img2, photflam2)) + zp2

    tab_cont = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/hst_contours_full.fits")  
    contours = [
        np.column_stack(((tab_cont["X"][tab_cont["Num_cont"] == i] - xmin)/npixbin, (tab_cont["Y"][tab_cont["Num_cont"] == i] - ymin)/npixbin))
        for i in range(tab_cont["Num_cont"].max() + 1)
        ]
    
    contours_rot = [rotatecont(numcont=0, npixbin=npixbin, rotacion=1/3), rotatecont(numcont=1, npixbin=npixbin, rotacion=3/5)]
    circles = []
    for p in circparam:
        circles.append(circulos_color(p[0], p[1], circr))
    
    circlesbin = []
    for circ in circles:
        circlesbini = []
        for coord in circ:
            circlesbini.append(coord/npixbin)
        circlesbin.append(circlesbini)


    resta = mag1 - mag2

    resta = gaussian_filter(resta, sigma=sigma)
    if write:
        restafits = resta.filled(fill_value=np.nan)
        hduresta = fits.PrimaryHDU(restafits)
        hduresta.writeto(f"{path}color{cam}_{f1}-{f2}.fits", overwrite=True)
    return resta, contours, contours_rot, circlesbin

def colorflux(f1, f2, cam="acs", ymin=4600, ymax=5400, xmin=4650, xmax=5450, circparam=parametros_circ, circr=radio, npixbin=1):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if cam == "acs":
        tel = "h"
        file1 = fits.open(f"{path}{cam}{f1}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f1}w_v1_drz.fits")[0]
        file2 = fits.open(f"{path}{cam}{f2}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f2}w_v1_drz.fits")[0]

    photflam1 = file1.header["PHOTFLAM"]
    photflam2 = file2.header["PHOTFLAM"]

    photplam1 = file1.header["PHOTPLAM"]
    photplam2 = file2.header["PHOTPLAM"]

    zp1 = -5*np.log10(photplam1)-2.408
    zp2 = -5*np.log10(photplam2)-2.408

    img1 = file1.data[ymin:ymax, xmin:xmax]
    img2 = file2.data[ymin:ymax, xmin:xmax]

    nbinsx = (xmax-xmin)//npixbin
    nbinsy = (ymax-ymin)//npixbin

    img1 = img1.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)
    img2 = img2.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)

    mask = []
    for p in circparam:
        c = (circulos_color(p[0], p[1], circr, flux=True)[0]//npixbin, circulos_color(p[0], p[1], circr, flux=True)[1]//npixbin)
        mask.append(np.ones(np.shape(img1)))
        mask[-1][c] = 0
    
    mag1 = -2.5*np.log10(hstflux(img1, photflam1)) + zp1
    mag2 = -2.5*np.log10(hstflux(img2, photflam2)) + zp2

    mag = mag1 - mag2
    #mag = -2.5*np.log10(hstflux(img1, photflam1)/hstflux(img2, photflam2))
    #fluxmag = [np.sum(img1[np.where(circ == 0)])*photflam1 for circ in mask]
    #flux2 = [np.sum(img2[np.where(circ == 0)])*photflam2 for circ in mask]

    return [np.nanmedian(mag[np.where(circ == 0)]) for circ in mask]

def colorflux_cont(f1, f2, cam="acs", ymin=4600, ymax=5400, xmin=4650, xmax=5450, npixbin=1, numcont=1, mascara="_elipses3"):
    path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
    if cam == "acs":
        tel = "h"
        file1 = fits.open(f"{path}{cam}{f1}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f1}w_v1_drz.fits")[0]
        file2 = fits.open(f"{path}{cam}{f2}/hlsp_clash_hst_{cam}-30mas_rxj2129_f{f2}w_v1_drz.fits")[0]

    photflam1 = file1.header["PHOTFLAM"]
    photflam2 = file2.header["PHOTFLAM"]

    photplam1 = file1.header["PHOTPLAM"]
    photplam2 = file2.header["PHOTPLAM"]

    zp1 = -5*np.log10(photplam1)-2.408
    zp2 = -5*np.log10(photplam2)-2.408

    img1 = file1.data[ymin:ymax, xmin:xmax]
    img2 = file2.data[ymin:ymax, xmin:xmax]

    nbinsx = (xmax-xmin)//npixbin
    nbinsy = (ymax-ymin)//npixbin

    img1 = img1.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)
    img2 = img2.reshape(nbinsy, npixbin, nbinsx, npixbin).sum(axis=3).sum(axis=1)

    mask = fits.open(f"{path}{cam}{f1}/contourmask_{numcont}{mascara}.fits")[0].data
    mask_rot = fits.open(f"{path}{cam}{f1}/contourmask_{numcont}{mascara}_vacio.fits")[0].data

    mag1 = -2.5*np.log10(hstflux(img1, photflam1)) + zp1
    mag2 = -2.5*np.log10(hstflux(img2, photflam2)) + zp2

    mag = mag1 - mag2
    print(mag[np.isnan(mag)])

    mag[np.isnan(mag)] = 0

    xgalfit = 4200
    ygalfit = 4650
    coordsmask = (np.where(mask == 0)[0] + ygalfit - ymin, np.where(mask == 0)[1] + xgalfit - xmin)
    coordsrot = (np.where(mask_rot == 0)[0] + ygalfit - ymin, np.where(mask_rot == 0)[1] + xgalfit - xmin)
    print(coordsrot)
    color_mask = np.nanmedian(mag[coordsmask])
    color_rot = np.nanmedian(mag[coordsrot])

    mask1 = np.ones(np.shape(mag))
    mask1[coordsmask] = 0
    mask2 = np.ones(np.shape(mag))
    mask2[coordsrot] = 0

    return color_mask, color_rot, mask1, mask2

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

"""vac = True
jwfiltros = np.array([115, 150, 200, 277, 356, 444])
flujo_parcialjw = np.array([])
flujo_fulljw = np.array([])
flujo_fulljw2 = np.array([])
flujo_fulljw3 = np.array([])
lbda_jw = np.array([])
err_jw = np.array([])
for i in jwfiltros:
    #flujo_parcialjw = np.append(flujo_parcialjw, flux(("jwst", i))[2])
    #flujo_fulljw = np.append(flujo_fulljw, flux(("jwst", i), "elipses")[2])
    #flujo_fulljw2 = np.append(flujo_fulljw2, flux(("jwst", i), "elipses2")[2])
    flujolbda_jw = flux(("jwst", i), "elipses3", vacio=vac)
    flujo_fulljw3 = np.append(flujo_fulljw3, flujolbda_jw[2])
    lbda_jw = np.append(lbda_jw, flujolbda_jw[3])
    err_jw = np.append(err_jw, flujolbda_jw[4])

hfiltros = np.array([105, 110, 125, 140, 160])
flujo_parcialh = np.array([])
flujo_fullh = np.array([])
flujo_fullh2 = np.array([])
flujo_fullh3 = np.array([])
lbda_h = np.array([])
err_h = np.array([])
for i in hfiltros:
    #flujo_parcialh = np.append(flujo_parcialh, flux(("hst", i))[2])
    #flujo_fullh = np.append(flujo_fullh, flux(("hst", i), "elipses")[2])
    #flujo_fullh2 = np.append(flujo_fullh2, flux(("hst", i), "elipses2")[2])
    flujolbda_h = flux(("hst", i), "elipses3", vacio=vac)
    flujo_fullh3 = np.append(flujo_fullh3, flujolbda_h[2])
    lbda_h = np.append(lbda_h, flujolbda_h[3])
    err_h = np.append(err_h, flujolbda_h[4])

acsfiltros = np.array([435, 475, 606, 625, 775, 814, 850])
flujo_parcialacs = np.array([])
flujo_fullacs = np.array([])
flujo_fullacs2 = np.array([])
flujo_fullacs3 = np.array([])
lbda_acs = np.array([])
err_acs = np.array([])
for i in acsfiltros:
    #flujo_parcialacs = np.append(flujo_parcialacs, flux(("acs", i))[2])
    #flujo_fullacs = np.append(flujo_fullacs, flux(("acs", i), "elipses")[2])
    #flujo_fullacs2 = np.append(flujo_fullacs2, flux(("acs", i), "elipses2")[2])
    flujolbda_acs = flux(("acs", i), "elipses3", vacio=vac)
    flujo_fullacs3 = np.append(flujo_fullacs3, flujolbda_acs[2])
    lbda_acs = np.append(lbda_acs, flujolbda_acs[3])
    err_acs = np.append(err_acs, flujolbda_acs[4])

filtrosfot = "F"+np.append(np.append(acsfiltros, hfiltros), jwfiltros).astype(str)+"W"
filtrosfot[6] = "F850LP"
#filtrosfot = np.append(np.append(acsfiltros, hfiltros), jwfiltros)
flujofot = np.append(np.append(flujo_fullacs3, flujo_fullh3), flujo_fulljw3)
errfot = np.append(np.append(err_acs, err_h), err_jw)

print(flujofot.dtype)
path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
fotarray = np.zeros(filtrosfot.size, dtype=[('Filtros', 'U6'), ('Flujo', float), ('Error', float)])
#fotarray = np.column_stack((filtrosfot, flujofot, errfot))
fotarray['Filtros'] = filtrosfot
fotarray["Flujo"] = flujofot
fotarray["Error"] = errfot
head = "Filtro (HST y JWST)    Flujo (maggies)       Error flujo (maggies)"
np.savetxt(f"{path}fotometria_ref.txt", fotarray, header=head, fmt="%10s %10.16f  %10.16f") 

musefiltros = np.arange(0, 3682, 1)
flujo_parcialmuse = np.array([])
flujo_fullmuse = np.array([])
flujo_fullmuse2 = np.array([])
flujo_fullmuse3 = np.array([])
lbda_muse = np.array([])
for i in musefiltros:
    #flujo_parcialmuse = np.append(flujo_parcialmuse, flux(("muse", i))[2])
    #flujo_fullmuse = np.append(flujo_fullmuse, flux(("muse", i), "elipses")[2])
    #flujo_fullmuse2 = np.append(flujo_fullmuse2, flux(("muse", i), "elipses2")[2])
    flujolbda_muse = flux(("muse", i), "elipses3", vacio=vac)
    flujo_fullmuse3 = np.append(flujo_fullmuse3, flujolbda_muse[0])
    lbda_muse = np.append(lbda_muse, flujolbda_muse[1])

specarray = np.column_stack((lbda_muse, flujo_fullmuse3))
headmuse = "Longitud de onda (ang)      Flujo (maggies)"
np.savetxt(f"{path}espectrometria_ref.txt", specarray, header=headmuse) 

#flujo_parcial_chefs = flux(("jw", 277), chefs=True)[2]
#flujo_full_chefs = flux(("jw", 277), "elipses", chefs=True)[2]
#flujo_full2_chefs = flux(("jw", 277), "elipses2", chefs=True)[2]
#flujo_fulltotal_chefs = flux(("jw", 277), "none", chefs=True)[2]

#fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#ax.plot(jwfiltros, flujo_full/flujo_parcial, ".-", color="blue", label="conservador")
#ax.plot(jwfiltros, flujo_full2/flujo_parcial, ".-", color="red", label="agresivo")
#ax.plot(jwfiltros, flujo_full3/flujo_parcial, ".-", color="green", label="conservador 2")

#ax.plot(jwfiltros[-3], flujo_full_chefs/flujo_parcial_chefs, ".-", color="green", label="conservador chefs")
#ax.plot(jwfiltros[-3], flujo_full2_chefs/flujo_parcial_chefs, ".-", color="pink", label="agresivo chefs")
#ax.plot(jwfiltros[-3], flujo_fulltotal_chefs/flujo_parcial_chefs, ".-", color="orange", label="total chefs")

#ax.legend()
#ax.set_title("Flujo total/flujo pequeño")
#ax.set_ylim(0.9,1.15)

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 4))
#fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

estilo = "s"
tamaño = 4
areas = False
lbda_jw = lbda_jw/1e2
lbda_h = lbda_h/1e4
lbda_acs = lbda_acs/1e4
lbda_muse = lbda_muse/1e4
print(err_jw)
print(err_h)
print(err_acs)
if areas:
    ax2.plot(lbda_jw, flujo_parcialjw, f"{estilo}", markersize=tamaño, color="black", label="parcial")
    ax2.plot(lbda_jw, flujo_fulljw, f"{estilo}", markersize=tamaño, color="green", label="conservador")
    #ax2.plot(lbda_jw, flujo_fulljw2, f"{estilo}", markersize=tamaño, color="red", label="agresivo")
    ax2.plot(lbda_jw, flujo_fulljw3, f"{estilo}", markersize=tamaño, color="red", label="conservador 2")

    ax2.plot(lbda_h, flujo_parcialh, f"{estilo}", markersize=tamaño, color="black", label="parcial")
    ax2.plot(lbda_h, flujo_fullh, f"{estilo}", markersize=tamaño, color="green", label="conservador")
    #ax2.plot(lbda_h, flujo_fullh2, f"{estilo}", markersize=tamaño, color="red", label="agresivo")
    ax2.plot(lbda_h, flujo_fullh3, f"{estilo}", markersize=tamaño, color="red", label="conservador 2")

    ax2.plot(lbda_acs, flujo_parcialacs, f"{estilo}", markersize=tamaño, color="black", label="parcial")
    ax2.plot(lbda_acs, flujo_fullacs, f"{estilo}", markersize=tamaño, color="green", label="conservador")
    #ax2.plot(lbda_acs, flujo_fullacs2, f"{estilo}", markersize=tamaño, color="red", label="agresivo")
    ax2.plot(lbda_acs, flujo_fullacs3, f"{estilo}", markersize=tamaño, color="red", label="conservador 2")

    ax2.plot(lbda_muse, flujo_parcialmuse, f"{estilo}", markersize=tamaño, color="black", label="parcial")
    ax2.plot(lbda_muse, flujo_fullmuse, f"{estilo}", markersize=tamaño, color="green", label="conservador")
    #ax2.plot(lbda, flujo_fullmuse2, f"{estilo}", markersize=tamaño, color="red", label="agresivo")
    ax2.plot(lbda_muse, flujo_fullmuse3, f"{estilo}", markersize=tamaño, color="red", label="conservador 2")
else:
    ax2.plot(lbda_muse, flujo_fullmuse3, f"-", markersize=tamaño/3, color="black", label="MUSE")
    ax2.plot(lbda_jw, flujo_fulljw3, f"{estilo}", markersize=tamaño, color="red", label="JWST/NIRCam")
    ax2.errorbar(lbda_jw, flujo_fulljw3, yerr=err_jw, ecolor="red", fmt="none")
    ax2.plot(lbda_h, flujo_fullh3, f"{estilo}", markersize=tamaño, color="blue", label="HST/WFC3")
    ax2.errorbar(lbda_h, flujo_fullh3, yerr=err_h, ecolor="blue", fmt="none")
    ax2.plot(lbda_acs, flujo_fullacs3, f"{estilo}", markersize=tamaño, color="green", label="HST/ACS")
    ax2.errorbar(lbda_acs, flujo_fullacs3, yerr=err_acs, ecolor="green", fmt="none")
#ax2.plot(lbda_jw[-3], flujo_parcial_chefs, f"{estilo}", color="cyan", label="parcial chefs")
#ax2.plot(lbda_jw[-3], flujo_full_chefs, f"{estilo}", color="green", label="conservador chefs")
#ax2.plot(lbda_jw[-3], flujo_full2_chefs, f"{estilo}", color="pink", label="agresivo chefs")
#ax2.plot(lbda_jw[-3], flujo_fulltotal_chefs, f"{estilo}", color="orange", label="total chefs")
ax2.set_xlabel("$\\lambda$ ($\\mu$m)")
#ax2.set_ylabel("Flujo (erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$ sr$^{-1}$)")
#ax2.set_ylabel("Flujo (MJy/sr)")
ax2.set_ylabel("Flujo (maggies)")
#ax2.set_yscale("log")
#ax2.set_xlim(400,900)
#ax2.set_ylim(17.5,22)
ax2.legend()"""

#ax[1].plot(lbda_jw, flujo_parcial/np.max(flujo_parcial), f"{estilo}", color="red", label="parcial")
#ax[1].plot(lbda_jw, flujo_full/flujo_parcial, ".-", color="blue", label="total/parcial")
#ax[1].legend()


"""fig, ax = plt.subplots(2, 1, figsize=(8, 8))
""ax[0].plot(lbda_jw, flujo_parcial, ".-", color="red", label="parcial")
ax[0].plot(lbda_jw, flujo_full, ".-", color="blue", label="total")
ax[0].legend()""

#ax[1].plot(lbda_jw, flujo_parcial/np.max(flujo_parcial), ".-", color="red", label="parcial")
ax[1].plot(lbda_jw, flujo_full/flujo_parcial, ".-", color="blue", label="total/parcial")
ax[1].legend()"""

"""filt = 444
path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/jwst{filt}/"
imbase, mask, a = flux(("jw", filt), "elipses3")
img = fits.open(f"{path}jwst{filt}_galindex3.fits")[3].data"""

tel = "acs"
filter = 814
path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}{filter}/"
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

"""#x1 = 4450
#x2 = 5500
#y1 = 4200
#y2 = 5700
#imbase3 = fits.open(f"{path}hlsp_clash_hst_acs-30mas_rxj2129_f{filter}w_v1_drz.fits")[0].data[x1:x2, y1:y2]
imbase3 = fits.open(f"{path}{tel}{filter}_galindex3.fits")[1].data
#mask3 = fits.open(f"{path}contourmask_full2.fits")[0].data[x1:x2, y1:y2]
mask3 = flux(("acs", i), "simple")[1]

#tab_contfull = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/{tel}st_contours_full.fits")  
#large_contoursfull = [
#    np.column_stack((tab_contfull["X"][tab_contfull["Num_cont"] == i], tab_contfull["Y"][tab_contfull["Num_cont"] == i] ))
#    for i in range(tab_contfull["Num_cont"].max() + 1)
#    ]

#print(tab_cont)
#print(large_contoursfull)

fig3, ax3 = plt.subplots(2, 1, figsize=(12, 12))
maximo = 0.3
minimo = 0

#ax3[0].set_xlim(4200, 5500)
#ax3[0].set_ylim(4650, 5600)
#ax3[1].set_xlim(4200, 5500)
#ax3[1].set_ylim(4650, 5600)

ax3[0].imshow(
    imbase3,
    vmin=minimo,
    vmax=maximo,
    cmap='gray',
    origin="lower"
    )

#for cont in large_contoursfull:
#    ax3[0].plot(cont[:, 0], cont[:, 1], 'g-', linewidth=0.5)

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

f1 = 435
f2 = 625

metalsimple, large_contours, large_contours_rot, circles = colorindex(f1, f2, write=False, sigma=0.6, npixbin=1)
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 16))
maximo = 2
minimo = 0
current_cmap = plt.colormaps['rainbow'].copy()
current_cmap.set_bad(color='black')
im4 = ax4.imshow(
    metalsimple,
    vmin=minimo,
    vmax=maximo,
    cmap=current_cmap,
    origin="lower"
    )

cbar = plt.colorbar(im4, ax=ax4, location="bottom", pad=0.008, shrink=0.95, aspect=30)
cbar.set_label("Índice de color F435W - F625W ($m_{AB}$)", size=12) 
#fig4.colorbar(figura)
for cont in large_contours:
    ax4.plot(cont[:, 0], cont[:, 1], '-', color="white", linewidth=4)

for cont in large_contours_rot:
    ax4.plot(cont[:, 0], cont[:, 1], '-', color="orange", linewidth=4)

"""for i in range(len(circles)):
    ax4.plot(circles[i][1], circles[i][0], ".", color="black", markersize=1)
    size = 16
    ax4.text(np.mean(circles[i][1]), 2*np.max(circles[i][0]) - np.mean(circles[i][0]), f"{i+1}", fontsize=size, fontweight="bold", horizontalalignment='center', verticalalignment='center')"""

ax4.set_xticks([])
ax4.set_yticks([])

"""metalcirculos = colorflux(f1, f2)
for i in range(len(metalcirculos)):
    print(f"{i+1}   {metalcirculos[i]}")
    ax4.plot(circfilled[i][1], circfilled[i][0], ".", color="black")"""

metalmasks = colorflux_cont(f1, f2, numcont=1)
metalmasks0 = colorflux_cont(f1, f2, numcont=0, mascara="")
print(f"CONTORNO 1 {metalmasks[0]} {metalmasks[1]}")
print(f"CONTORNO 0 {metalmasks0[0]} {metalmasks0[1]}")

"""ax4.imshow(
    metalmasks[2],
    vmin=0,
    vmax=1,
    cmap="gray",
    origin="lower",
    alpha=0.2
    )"""

plt.show()


#plt.show()
#ax2[0].set_ylim(400, 1000)
#ax2[0].set_xlim(350, 800)
# %%
