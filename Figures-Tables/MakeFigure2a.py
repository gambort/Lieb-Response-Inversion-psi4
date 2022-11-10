import numpy as np

import matplotlib.pyplot as plt
from NiceColours import *

from LibHelper import *

from mplChemView.mplChemView import *

Mol = "Purple"
Basis = "def2-msvp"

fig, (ax_En, ax_Fs) = plt.subplots(2,1, figsize=(6,3), sharex=True)

NOcc, T, En_Iter, Fs_Iter, Fs_Inf, Misc \
    = GetProps(Mol+".xyz_SCF", Basis, DFA="svwn")

eps = Misc['epsilon_k']
Gap = (eps[NOcc]-eps[NOcc-1]) / kcal * 27.211 # Gap in eV

NIter = len(En_Iter)

FL_Err = Fs_Inf - Fs_Iter

K = "Navy"
ax_En.semilogy(T, En_Iter,
               color=NiceColour(K),
               label=Basis,
               )

ax_Fs.semilogy(T, FL_Err,
               color=NiceColour(K),
               label=Basis,
               )

ax_En.set_yticks([10,1,0.1,0.01,0.001])
ax_En.set_yticklabels(['', '1', '', '', '$\\frac{1}{1000}$'])
ax_En.axis([0,NIter, 2e-4, 2])

ax_Fs.set_yticks([100,10,1])
ax_Fs.set_yticklabels(['100','10', '1'])
ax_Fs.axis([0,NIter, 8e-1, 100])

AddBorder(
    ax_En.text(0.02, 0.03, "Hartree error",
               fontsize=14,
               transform=ax_En.transAxes,
               )
)
AddBorder(
    ax_Fs.text(0.02, 0.06, "Lieb error @ "+Basis,
               fontsize=14,
               transform=ax_Fs.transAxes,
            )
)

fig.supylabel("Error [kcal/mol]", x=0.001, fontsize=14)

# Add the HOMO-LUMO gap
ax_En.text(
    0.48, 0.03, "HOMO/LUMO gap = %.1f eV"%(Gap),
    fontsize=14,
    transform=ax_En.transAxes,
)

# Add the molecule
Mol = MoleculeDrawer()
Mol.ReadXYZFile("Purple.xyz")

ax_Mol = ax_Fs.inset_axes([0.48,0.2,0.5,0.8],
                          transform=ax_Fs.transAxes)
ax_Mol.axis('off')
Mol.DrawToAxes(ax_Mol, ViewAng=[45,0,0])

fig.text(0.0,1.0,"${\\bf a}$", fontsize=18,
         verticalalignment="top",
         )

fig.tight_layout(pad=0.1)
plt.savefig("Error-Purple.pdf")
plt.show()
