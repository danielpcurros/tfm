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

"""fig, ax = plt.subplots(1, 1, figsize=(10,5))
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

med_contours = [cont for cont in contours if len(cont) > 1000]
med_contours = med_contours[4:]

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
for cont in med_contours:
    ax3.plot(cont[:, 0], cont[:, 1], '-', color="orange", linewidth=0.3)
for cont in large_contours:
    ax3.plot(cont[:, 0], cont[:, 1], 'g-', linewidth=1.5)

ax3.set_xticks([])
ax3.set_yticks([])"""

#líneas: halfa, hgamma, hdelta, ca absorción, OTi, CaT en 8500 aa
z = 0.234
unit = 10*(1+z)
halfa = [656.279*unit, "H$\\alpha$"]
hbeta = [486.135*unit, "H$\\beta$"]
hgamma = [434.0472*unit, "H$\\gamma$"]
hdelta = [410.1734*unit, "H$\\delta$"]
cata = [849.8*unit, "CaT a"]
catb = [854.2*unit, "CaT b"]
catc = [866.2*unit, "CaT c"]
d4000 = [400*unit, "D4000"]
bandag = [430.4*unit, "G"]
cah = [393.4*unit, "Ca H"]
cak = [396.9*unit, "Ca K"]
mg = [517.5*unit, "Mg b"]
na = [589.4*unit, "Na"]
nii = [658.4*unit, "NII"]
sii = [671.7*unit, "SIIa"]
sii2 = [673.1*unit, "SIIb"]
l1 = [510*unit, "A"]
l2 = [515*unit, "B"]


print(cata, catb, catc)

lineas = []
lineastop = [hgamma, hdelta, cah, hbeta, mg, halfa]
lineasbottom = [bandag, cak, na]
#print(lineasbottom)
lineas.extend(lineastop)
lineas.extend(lineasbottom)

fig4, ax4 = plt.subplots(1, 1, figsize=(15,7.5))
ax4.plot(lbda_shell, spec_shell*1e8, "k-", label="Espectroscopía")
ylimite = 5
offset = 0.12
size = 12
length = 0.12
maxlambda = 7500
minlambda = 5000
anchura = 0.7
for l in lineastop:
    xaux = np.argmin(np.abs(lbda_shell-l[0]))
    yaux = spec_shell[xaux]*1e8
    print(l[1], yaux)
    ymax = yaux + 4*length
    ymin = yaux + length
    if l[0] > maxlambda:
        ymax+=2*length
        ymin+=2*length
    elif l[0] > 5900 and l[0] < 6600:
        ymax+=0.5*length
        ymin+=0.5*length
    print(ymax)
    ax4.axvline(l[0], color="red" ,linewidth=anchura, ymax=ymax/ylimite, ymin=ymin/ylimite)
    """offset = 15
    xpos = l[0]-offset
    if l[0] < 550:
        ypos = 2
    else:
        ypos = 1"""
    ypos = ymax + offset
    ax4.text(l[0], ypos, f"{l[1]}", horizontalalignment='center', verticalalignment='center', fontsize=size)

for l in lineasbottom:
    xaux = np.argmin(np.abs(lbda_shell-l[0]))
    yaux = spec_shell[xaux]*1e8
    print(l[1], yaux)
    ymax = yaux - length
    ymin = yaux - 4*length
    """if l[0] > maxlambda:
        ymax-=length
        ymin-=length"""
    if l[0] < minlambda:
        ymin+=length
    print(ymax)
    ax4.axvline(l[0], color="red" ,linewidth=anchura, ymax=ymax/ylimite, ymin=ymin/ylimite)
    """offset = 15
    xpos = l[0]-offset
    if l[0] < 550:
        ypos = 2
    else:
        ypos = 1"""
    ypos = ymin - offset
    ax4.text(l[0], ypos, f"{l[1]}", horizontalalignment='center', verticalalignment='center', fontsize=size)
"""ax4_t = ax4.secondary_xaxis('top')
ax4_b = ax4.secondary_xaxis('bottom')
ax4_t.set_ticks([l[0] for l in lineastop])
ax4_t.set_xticklabels([l[1] for l in lineastop], fontsize=10)
ax4_b.set_ticks([l[0] for l in lineasbottom])
ax4_b.set_xticklabels([l[1] for l in lineasbottom], fontsize=10)"""
ax4.set_xlabel("$\\lambda$ (Å)", fontsize=12)
ax4.set_ylabel("Flujo de la shell ($10^{-8} \\: \\text{maggies}$)", fontsize=12)
ax4.set_xlim(4500, 9600)
ax4.set_ylim(0,ylimite)
ax4.tick_params(labelsize=12)
#ax.plot(filt_shell, fot_resta, "rs", label="Fotometría")
