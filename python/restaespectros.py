#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 18:06 2026

@author: dapec
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits

path = f"/home/daniel/Aplicacións/GALFIT/files/tfm/"
muse_ref = np.loadtxt(f"{path}espectrometria_ref.txt")
lbda_ref = muse_ref[:,0]
spec_ref = muse_ref[:,1]

muse_shell = np.loadtxt(f"{path}espectrometria_shell.txt")
lbda_shell = muse_shell[:,0]
spec_shell = muse_shell[:,1]

tel_ref = np.loadtxt(f"{path}fotometria_ref.txt")
filt_ref = tel_ref[:,0]
fot_ref = tel_ref[:,1]
errfot_ref = tel_ref[:,2]

for i in range(7, len(filt_ref)):
    filt_ref[i] = filt_ref[i]*10

tel_shell = np.loadtxt(f"{path}fotometria_shell.txt")
filt_shell = tel_shell[:,0]
fot_shell = tel_shell[:,1]
errfot_shell = tel_shell[:,2]

for i in range(7, len(filt_shell)):
    filt_shell[i] = filt_shell[i]*10

spec_resta = spec_shell - spec_ref
fot_resta = fot_shell - fot_ref
errfot_resta = np.sqrt(errfot_ref**2 + errfot_shell**2)
errfot_ratio = np.sqrt((errfot_shell/fot_ref)**2 + (errfot_ref*fot_shell/fot_ref**2)**2) 

fig, ax = plt.subplots(1, 1, figsize=(10,5))
ax.plot(lbda_shell/10, spec_resta, "k-", label="Espectroscopía")
ax.plot(filt_shell, fot_resta, "rs", label="Fotometría")
#ax.errorbar(filt_shell, fot_resta, yerr=errfot_resta, ecolor="green", fmt="none")
ax.set_title("shell - ref")

fig2, ax2 = plt.subplots(1, 1, figsize=(10,5))
ax2.plot(lbda_shell/10, spec_shell/spec_ref, "k-", label="Espectroscopía")
ax2.plot(filt_shell, fot_shell/fot_ref, "rs", label="Fotometría")
ax2.errorbar(filt_shell, fot_shell/fot_ref, yerr=errfot_ratio, ecolor="red", fmt="none")
ax2.set_title("shell/ref")

tab_cont = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/jwst_smallcontours.fits")  
contours = [
    np.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i], tab_cont["Y"][tab_cont["Num_cont"] == i]))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

tab_cont = Table.read(f"/home/daniel/Aplicacións/GALFIT/files/tfm/jwst_contours.fits")  
large_contours = [
    np.column_stack((tab_cont["X"][tab_cont["Num_cont"] == i], tab_cont["Y"][tab_cont["Num_cont"] == i]))
    for i in range(tab_cont["Num_cont"].max() + 1)
    ]

img = fits.open(f"{path}jwst277/jwst277_galindex3.fits")[3].data
M, N = np.shape(img)

escala = 6
fig3, ax3 = plt.subplots(1, 1, figsize=(N/M*escala,escala))

ax3.imshow(
    img,
    vmin=0,
    vmax=0.2,
    cmap="gray",
    origin="lower"
)

for cont in contours:
    ax3.plot(cont[:, 0], cont[:, 1], 'r-', linewidth=0.3)
for cont in large_contours:
    ax3.plot(cont[:, 0], cont[:, 1], 'g-', linewidth=1.5)

ax3.set_xticks([])
ax3.set_yticks([])
