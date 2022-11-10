import numpy as np

import matplotlib.pyplot as plt
from NiceColours import *

from LibHelper import *

NInf = 30 # use last this many points to extrapolate Fs
NFit = 60 # use to fit Fs

Rm = ['I', 'II', 'III', 'IV', 'V',
      'VI', 'VII', 'VIII', 'IX', 'X']

SysList = ('hydrogen_chloride',
           'water', 'ketene', 'ethylene', "glyoxal",
           'butadiene', 'benzene', 'benzoquinone')

LongID = {'cc-pvdz':'CCSD@cc-pvdz', 'cc-pvtz':'HF@cc-pvtz'}

fig_En, (ax1, ax2) = plt.subplots(2,1, figsize=(6,3), sharex=True)
fig_Fs, (ax3, ax4) = plt.subplots(2,1, figsize=(6,3), sharex=True)

KList = {}
NIter = 0
for Basis, ax_En, ax_Fs in \
    zip(('cc-pvdz', 'cc-pvtz'), (ax1, ax2), (ax3, ax4)):
    for K, Mol in enumerate(SysList):
        ID = Mol+".xyz"
        if Basis == 'cc-pvtz': ID += "_HF"
        
        NOcc, T, En_Iter, Fs_Iter, Fs_Inf, vHxc_k \
            = GetProps(ID, Basis, DFA='svwn',
                       NWork = 300, # Only use 300 entries
                       )

        if T is None:
            continue

        KList[K] = True
        

        #T = T[:300]
        #Fs_Iter = Fs_Iter[:300]
        #En_Iter = En_Iter[:300]

        NIter = max(len(En_Iter), NIter)

        # Filter to non-zero (i.e. actually evaluated) value
        ii = En_Iter>0.
        En_Iter = En_Iter[ii]
        Fs_Iter = Fs_Iter[ii]
        T = np.arange(len(En_Iter))

        # Evaluate the FL error
        p = np.polyfit(1./T[-NInf:]**1.5, Fs_Iter[-NInf:], 1)
        Fs_Inf = np.polyval(p, 0.)
        FL_Err = Fs_Inf - Fs_Iter

        #Fs_Inf, FL_Fit = FitFs(T[-NInf:], Fs_Iter[-NInf:], TAll=T)

        #########################
        ax_En.semilogy(T, En_Iter,
                    color=NiceColour(K),
                    label=Rm[K],
                    )
        print("%20s %.2f%% @ %d"%(Mol, En_Iter[:50].min()/En_Iter[0]*100., 50))
        k0 = np.min(np.hstack((np.argwhere(En_Iter<1e-5).reshape(-1,),
                               1000)))
        if k0<0:
            ax_En.scatter(T[k0], En_Iter[k0],
                          marker='v',
                          color=NiceColour(K),
                          zorder=100,
                          )

        
        ax_Fs.semilogy(T, FL_Err,
                       color=NiceColour(K),
                       label=Rm[K],
                       )

    for ax in (ax_En, ax_Fs):
        AddBorder(
            ax.text(0.02, 0.03, LongID[Basis],
                    fontsize=14,
                    verticalalignment="bottom",
                    transform = ax.transAxes,
                    )
        )

    ax_En.set_yticks([10,1,0.1,0.01,0.001])
    ax_En.set_yticklabels(['', '1', '', '', '$\\frac{1}{1000}$'])
    ax_En.axis([0,NIter, 1e-4, 60])

    ax_Fs.set_yticks([200,100,50,20,5,2,0.5,0.2], minor=True)
    ax_Fs.set_yticks([100,10,1,0.1])
    ax_Fs.set_yticklabels(['100', '', '1', ''])
    ax_Fs.axis([0,NIter, 3e-2, 200])

    ax_En.legend(loc="upper right", ncol=4)
    ax_Fs.legend(loc="upper right", ncol=4)

        
fig_En.supylabel("Hartree error [kcal/mol]", fontsize=14, x=0.001)
ax2.set_xlabel("Number iterations, $T$", fontsize=14)
fig_En.tight_layout(pad=0.1)

fig_Fs.supylabel("Lieb error [kcal/mol]", fontsize=14, x=0.001)
ax4.set_xlabel("Number iterations, $T$", fontsize=14)
fig_Fs.tight_layout(pad=0.1)

fig_En.savefig("Error-En.pdf")
fig_Fs.savefig("Error-Lieb.pdf")

print(", ".join(["%s) %s"%(Rm[K], SysList[K].replace("_", " "))
                 for K in list(KList)]))

plt.show()
