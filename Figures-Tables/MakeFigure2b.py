import numpy as np

import matplotlib.pyplot as plt
from NiceColours import *

from LibHelper import *

Mol = "hydrogen_chloride"
BasisList = ['cc-pvdz', 'aug-cc-pvdz',
             'cc-pvtz', 'aug-cc-pvtz',
             'cc-pvqz', 'aug-cc-pvqz',
             'def2-tzvp', 'def2-qzvp', 'def2-qzvppd']

NoShow = ['cc-pvdz', 'aug-cc-pvdz', 'aug-cc-pvtz']

fig, (ax_En, ax_Fs) = plt.subplots(2,1, figsize=(6,3), sharex=True)


KList = {}
NIter = 0

NShow = 0

TsList = {}
epsList = {}
for K, Basis in enumerate(BasisList):
    NOcc, T, En_Iter, Fs_Iter, Fs_Inf, Misc \
        = GetProps(Mol+".xyz", Basis, DFA="svwn")
    _, T_H2 , En_Iter_H2 , Fs_Iter_H2 , Fs_Inf_H2 , Misc_H2  \
        = GetProps("H2.mol", Basis, DFA="pbe")
    _, T_Cl2, En_Iter_Cl2, Fs_Iter_Cl2, Fs_Inf_Cl2, Misc_Cl2 \
        = GetProps("Cl2.mol", Basis, DFA="pbe")

    
    if (T is None) or (T_H2 is None) or (T_Cl2 is None): continue

    NIter = max(len(En_Iter), NIter)        

    FL_Err = Fs_Inf - Fs_Iter
    FL_Err_Cl = (Fs_Inf_Cl2 - Fs_Iter_Cl2)/2

    TsList[Basis] = [Fs_Inf, Fs_Inf_H2/2., Fs_Inf_Cl2/2.,
                     Fs_Inf - Fs_Inf_H2/2. - Fs_Inf_Cl2/2.]
    epsList[Basis] = Misc['epsilon_k']

    if Basis in NoShow:
        print("Skipping")
        continue

    NShow += 1

    #########################
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

    ax_Fs.set_yticks([10,1,0.1])
    ax_Fs.set_yticklabels(['10', '1', '0.1'])
    ax_Fs.axis([0,NIter, 1e-1, 5])

    AddBorder(
        ax_En.text(0.02, 0.03, "Hartree error",
                   fontsize=14,
                   transform=ax_En.transAxes,
                   )
    )
    AddBorder(
        ax_Fs.text(0.02, 0.03, "Lieb error",
                   fontsize=14,
                   transform=ax_Fs.transAxes,
                   )
        )

    ax_En.legend(loc="upper right", ncol=min(3,int(np.ceil(NShow/2))))


print("="*72)    
print("%14s %35s %26s"%("", "Ts", "epsilon"))
print("%-14s & %8s & %8s & %8s & %8s & %8s \\\\"\
      %("Basis", "HCl", "Diff",
        "HOMO", "LUMO", "Gap"))
for K, Basis in enumerate(BasisList):
    Fs = TsList[Basis]
    h = epsList[Basis][8]
    l = epsList[Basis][9]
    g = l-h
    print("%-14s & %8.1f & %8.1f & %8.1f & %8.1f & %8.1f \\\\"\
          %(Basis, Fs[0], Fs[3],
            h, l, g))
    
fig.supylabel("Error [kcal/mol]", x=0.001, fontsize=14)

fig.text(0.0,1.0,"${\\bf b}$", fontsize=18,
         verticalalignment="top",
         )

fig.tight_layout(pad=0.1)

plt.savefig("Error-Basis.pdf")
plt.show()
