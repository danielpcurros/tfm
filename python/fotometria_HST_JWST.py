import scipy
from astropy.io import fits
import astropy.units as u
import numpy as np

# To measure the magnitude of a HST or JWST image using a region map

def main(imfile,bgrmsfile,regionfile,telescope,gain,bgfile='',maskfile='',identifier=1):
    
    # imfile =  archivo con la imagen donde se quiere medir la fotometria
    # regionfile = archivo con una mascara senhalando region donde se quiere medir la fotometria
    # telescope = 'HST' o 'JWST'
    # gain = ganancia de la camara
    # bgfile =  archivo con el mapa de background
    # bgrmsfile =  archivo con el mapa de la rms del background
    # maskfile = archivo con una mascara para no medir la fotometria en esos pixeles
    # identifier = numero que identifica en mascara del regionfile, el objeto al cual se le quiere medir la fotometria
    
    hduim = fits.open(imfile)
    im = hduim[0].data
    head = hduim[0].header
    pixelscale = head['CD2_2']*3600.
    if telescope=='JWST': 
        exptime = head['XPOSURE']
        photmjsr = head['PHOTMJSR']
        corr = ((pixelscale*u.arcsec)**2.).to(u.steradian)
    else:
        photflam = head['PHOTFLAM']
        photplam = head['PHOTPLAM']
        exptime = head['EXPTIME']
    hduim = fits.open(bgrmsfile)
    bg_rms = hduim[0].data
    if len(bgfile):
        hduim = fits.open(bgfile)
        bg = hduim[0].data
        im = im-bg
    if len(maskfile):
        hduim = fits.open(maskfile)
        mask = hduim[0].data
        im[mask==1] = 0.
    
    hduim = fits.open(regionfile)
    region = hduim[0].data
    
    flux = im[region==identifier].sum()
    if telescope=='HST':
        zp = -5*scipy.log10(photplam)-2.408
        mag = -2.5*scipy.log10(flux*photflam)+zp
    elif telescope=='JWST':
        zp = -6.10-2.5*scipy.log10(corr.value)
        mag = -2.5*scipy.log10(flux)+zp
    else:
        print('Invalid telescope')
        return
    
    good = scipy.equal(region,identifier)*scipy.not_equal(im,0.)
    area = scipy.size(region[good])
    
    '''
    flujo = flux*exptime
    bg_rms = scipy.mean(bg_rms[scipy.isfinite(bg_rms)])*exptime
    print bg_rms
    fluxerrorskynoise = (bg_rms**2.)*area    # Error due to sky noise in the aperture (squared)
    fluxerroraperturesum = flujo/gain       # Error in the aperture sum (squared)
    error_mag = 1.0857*scipy.sqrt(fluxerrorskynoise+fluxerroraperturesum)/flujo
    #error_flujo = scipy.log(10)*flux*error_mag/2.5
    '''
    flujo = flux*exptime
    if telescope=='JWST':
        flujo = flujo/photmjsr
        im = im/photmjsr
        bg_rms = bg_rms/photmjsr
    error_bg = 1.0857*scipy.sqrt((bg_rms**2.)*area+flujo/gain)/flujo
    error_mag_total = scipy.sqrt(error_bg**2.+(im/exptime)*((error_bg/scipy.median(bg_rms[scipy.isfinite(bg_rms)]))**2.)) #el ultimo termino incluye el ruido de Poisson de la imagen, calculado como en photutils
    #error_mag_total = scipy.sqrt(error_bg**2.+(im/exptime))
    error_mag_total[im<0] = error_mag_total[im<0]
    error_mag_total = error_mag_total.real
    good = scipy.equal(region,identifier)*scipy.isfinite(error_mag_total)*scipy.not_equal(im,0.)
    error_mag = scipy.sqrt((error_mag_total[good]**2.).sum())
    error_flujo = scipy.log(10)*flux*error_mag/2.5
    
    
    print(flujo,error_flujo,mag,error_mag)
    return flujo,error_flujo,mag,error_mag


def main2(im,head="",bg_rms="",region="",telescope="",jwsw=True,bgfile='',maskfile='',identifier=0):
    
    # imfile =  archivo con la imagen donde se quiere medir la fotometria
    # regionfile = archivo con una mascara senhalando region donde se quiere medir la fotometria
    # telescope = 'HST' o 'JWST'
    # gain = ganancia de la camara
    # bgfile =  archivo con el mapa de background
    # bgrmsfile =  archivo con el mapa de la rms del background
    # maskfile = archivo con una mascara para no medir la fotometria en esos pixeles
    # identifier = numero que identifica en mascara del regionfile, el objeto al cual se le quiere medir la fotometria
    

    pixelscale = head['CD2_2']*3600.
    if telescope=='jwst': 
        exptime = head['XPOSURE']
        photmjsr = head['PHOTMJSR']
        corr = ((pixelscale*u.arcsec)**2.).to(u.steradian)
        if jwsw:
            gain = 2.05
        else:
            gain = 1.84
    elif telescope=="hst" or telescope=="acs":
        photflam = head['PHOTFLAM']
        photplam = head['PHOTPLAM']
        exptime = head['EXPTIME']
        gain = head["CCDGAIN"]

  
    if len(bgfile):
        hduim = fits.open(bgfile)
        bg = hduim[0].data
        im = im-bg
    if len(maskfile):
        hduim = fits.open(maskfile)
        mask = hduim[0].data
        im[mask==1] = 0.
    
    
    flux = im[region==identifier].sum()
    if telescope=='hst' or telescope=="acs":
        zp = -5*np.log10(photplam)-2.408
        mag = -2.5*np.log10(flux*photflam)+zp
    elif telescope=='jwst':
        zp = -6.10-2.5*np.log10(corr.value)
        mag = -2.5*np.log10(flux)+zp
    elif telescope=="muse":
        mag = -2.5*np.log10(flux)-48.6
        return mag
    else:
        print('Invalid telescope')
        return
    
    good = np.equal(region,identifier)*np.not_equal(im,0.)
    area = np.size(region[good])
    
    '''
    flujo = flux*exptime
    bg_rms = np.mean(bg_rms[np.isfinite(bg_rms)])*exptime
    print bg_rms
    fluxerrorskynoise = (bg_rms**2.)*area    # Error due to sky noise in the aperture (squared)
    fluxerroraperturesum = flujo/gain       # Error in the aperture sum (squared)
    error_mag = 1.0857*np.sqrt(fluxerrorskynoise+fluxerroraperturesum)/flujo
    #error_flujo = np.log(10)*flux*error_mag/2.5
    '''
    flujo = flux*exptime
    if telescope=='jwst':
        flujo = flujo/photmjsr
        im = im/photmjsr
        bg_rms = bg_rms/photmjsr
    error_bg = 1.0857*np.sqrt((bg_rms**2.)*area+flujo/gain)/flujo
    error_mag_total = np.sqrt(error_bg**2.+(im/exptime)*((error_bg/np.median(bg_rms[np.isfinite(bg_rms)]))**2.)) #el ultimo termino incluye el ruido de Poisson de la imagen, calculado como en photutils
    #error_mag_total = np.sqrt(error_bg**2.+(im/exptime))
    error_mag_total[im<0] = error_mag_total[im<0]
    error_mag_total = error_mag_total.real
    good = np.equal(region,identifier)*np.isfinite(error_mag_total)*np.not_equal(im,0.)
    error_mag = np.sqrt((error_mag_total[good]**2.).sum())
    error_flujo = np.log(10)*flux*error_mag/2.5
    
    
    print(flujo,error_flujo,mag,error_mag)
    return flujo,error_flujo,mag,error_mag

