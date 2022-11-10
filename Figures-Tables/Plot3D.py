import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from NiceColours import *

from LibPlot3D import *

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('-M', type="string", default="ketene",
                  help="One of a list")

parser.add_option('--h', type="float", default=0.0,
                  help="Step size")
parser.add_option('--sigma', type="float", default=0.08,
                  help="Integration region")
parser.add_option('--YR', type="float", default=1.5,
                  help="Step size")

parser.add_option('--DFA', type="string", default="svwn",
                  help="Step size")
parser.add_option('--Suff', type="string", default="xyz",
                  help="Step size")

parser.add_option('--Pad', type="float", default=2.0,
                  help="Padding on line")

parser.add_option('--Ratio', action="store_true", default=False,
                  help="Show ratio of xc/H potentials")

parser.add_option('--Basis', type="string", default="cc-pvdz",
                  help="Specify the basis set")

parser.add_option('--Save', action="store_true", default=False,
                  help="Save the figures properly")

(Opts, args) = parser.parse_args()


np.set_printoptions(precision=3, suppress=True)

Basis = Opts.Basis # "cc-pvdz"
DoSave = Opts.Save

# Force a save if a fine grid is selected
if (-0.2<Opts.h) and (Opts.h<0.): DoSave = True

h = Opts.h
sigma = Opts.sigma
All = True

MolStr = Opts.M.lower()

Suff = Opts.Suff # Default for xyz files
DFA = Opts.DFA

if MolStr in ("h2"," h2_0.75"):
    Mol, P0, P1 = "h2_0.75", (0,2), None
    Suff = 'mol'
elif MolStr in ("h2x2"," h2_1.50"):
    Mol, P0, P1 = "h2_1.50", (0,2), None
    Suff = 'mol'
elif MolStr in ("h2d"," h2_3.50"):
    Mol, P0, P1 = "h2_3.50", (0,2), None
    Suff = 'mol'
elif MolStr in ("hydrogen_chloride", "hcl"):
    Mol, P0, P1 = "hydrogen_chloride", (0,1), None
elif MolStr in ("hydrogen_sulfide", "sh2"):
    Mol, P0, P1 = "hydrogen_sulfide", (0,1), None
elif MolStr[:2] == "wa":
    Mol, P0, P1 = "water", (0,1), (0,2)
elif MolStr[:2] == "bu":
    Mol, P0, P1 = "butadiene", (0,1), (0,2)
elif MolStr == "h2ox3":
    Mol, P0, P1 = "H2Ox3", (0,3), (0,6)
elif MolStr[:2] == "ke":
    Mol, P0, P1 = "ketene", (0,2), (0,4)
elif MolStr[:2] == "et":
    Mol, P0, P1 = "ethylene", (0,1), (0,2)
elif MolStr[:2] == "gl":
    Mol, P0, P1 = "glyoxal", (0,1), (0,2)
elif MolStr[:5] == "benze":
    Mol, P0, P1, All = "benzene", (0,3), (0,4), True
elif MolStr[:5] == "benzo":
    Mol, P0, P1, All = "benzoquinone", (0,1), (0,2), True
else: quit()

MolName = Mol
if len(Suff)>3:
    MolName += Suff[3:]
if not(DFA.lower()=="svwn"):
    MolName += "_%s"%(DFA.lower())
    
X = np.load("Densities/rho_%s.%s_%s_%s.npz"%(Mol, Suff, Basis, DFA),
            allow_pickle=True)
xyz = X['xyz']
xyz_Nuc = X['xyz_Nuc']
w = X['w']
rho = X['rho']
rhoxc = X['rhoxc']
rhox = None # Eventually might be able to use this

Nel = np.dot(w, rho)
if Nel <= 2.001: rhox = -rho*(Nel-1.)/Nel

print("<rho> = %8.4f, <rho_xc> = %8.4f"\
      %(np.dot(w, rho), np.dot(w, rhoxc)))

print("Original nuclear positions")
print(xyz_Nuc)

U, Range = GetPlane(xyz_Nuc, P0, P1)

PHelp = Projector(xyz, w, xyz_Nuc, U=U, P0=P0)
print("Rotated nuclear positions")
print(PHelp.xyz_Nuc)

print("="*72)

x,y, Range = PHelp.GetPlane(Range, h=h, All=All)
xp, yp = PHelp.GetLine(Pad=Opts.Pad)
z0 = 0.


if True:
    PHelp.Prescreen(sigma, z0)
    
    rho_L = np.zeros((len(xp),))
    rhoxc_L = np.zeros((len(xp),))

    VH_L = np.zeros((len(xp),))
    Vxc_L = np.zeros((len(xp),))
    Vx_L = np.zeros((len(xp),))

    for i in range(len(xp)):
        rho_L[i], rhoxc_L[i] \
            = PHelp.Project(xp[i], PHelp.xyz_Nuc[0,1], z0,
                            (rho, rhoxc))
        if rhox is None:
            VH_L[i], Vxc_L[i] \
                = PHelp.Potential(xp[i], PHelp.xyz_Nuc[0,1], z0,
                                  (rho, rhoxc),
                                  )
        else:
            VH_L[i], Vx_L[i], Vxc_L[i] \
                = PHelp.Potential(xp[i], PHelp.xyz_Nuc[0,1], z0,
                                  (rho, rhox, rhoxc),
                                  )

    fig_L, (axL, axV) = plt.subplots(2,1, figsize=(6,3), sharex=True)

    Norm, YR = 1., Opts.YR # np.abs(rhoc_L).max()+0.1

    axL.plot(xp, rhoxc_L * Norm,
             color=NiceColour("Navy"), linewidth=3, 
             label='xc',
             )
    axL.plot(xp, (rho_L + rhoxc_L) * Norm,
             color=NiceColour("Magenta"), dashes=(1,1),
             label='Hxc',
             )
    axL.plot(xp, rho_L * Norm,
             color=NiceColour("Orange"), dashes=(3,3),
             label='H',
             )

    if Opts.Ratio:
        axV.plot(xp, Vxc_L/VH_L, color=NiceColour("Navy"))
    else:
        Vx_LDA_L = -(3*rho_L/np.pi)**(1/3)
        axV.plot(xp, Vxc_L, color=NiceColour("Navy"),
                 label="xc",)
        if not(rhox is None):
            Vc_L = Vxc_L - Vx_L
            axV.plot(xp, Vc_L, color=NiceColour("Teal"),
                     label="c",)
            
        axV.plot(xp, Vx_LDA_L, color=NiceColour("Blue"), dashes=(3,1),
                 label="LDA x",)



    # Add the nuclear positions
    X_Nuc = PHelp.xyz_Nuc[np.abs(PHelp.xyz_Nuc[:,1]
                                 -PHelp.xyz_Nuc[0,1])<1e-2,0]
    axL.scatter(X_Nuc, 0.*X_Nuc, color='k')

    axV.plot([-100,100],[0,0],":k", linewidth=1)
    axV.scatter(X_Nuc, 0.*X_Nuc, color='k')

    axL.axis([xp.min(), xp.max(), -YR, YR])
    if Opts.Ratio:
        axV.axis([xp.min(), xp.max(), -0.7, 0.4])
        axV.set_yticks([-0.6,-0.3,0.0,0.3])
    else:
        YMin = max(-np.ceil(-2. * Vxc_L.min())*0.5, -4.0)
        YMax = -0.25*YMin
        
        axV.axis([xp.min(), xp.max(), YMin, YMax])
        axV.legend(loc="lower left")
        
    axL.legend(loc="lower left")

    axL.set_ylabel("$n^S$ [Bohr$^{-3}$]", fontsize=14)

    if Opts.Ratio:
        axV.set_ylabel("$v_{\\rm xc}/v_{\\rm H}$", fontsize=14)
    else:
        axV.set_ylabel("$v$ [Ha]", fontsize=14)

    axV.set_xlabel("Distance, $D$ [Bohr]", fontsize=14)

    fig_L.tight_layout(pad=0.1)
    if DoSave:
        fig_L.savefig("LDens_%s_%s.pdf"%(MolName, Basis))
    else:
        fig_L.savefig("Images/LDens_%s_%s.png"%(MolName, Basis))

if np.abs(h)>0.:
    PHelp.Prescreen(sigma, z0)

    rho_P = np.zeros((len(x),len(y)))
    rhoxc_P = np.zeros((len(x),len(y)))
    
    for i,j in itertools.product(range(len(x)), range(len(y))):
        rho_P[i,j], rhoxc_P[i,j] \
            = PHelp.Project(x[i], y[j], z0,
                            (rho, rhoxc))

    fig_P, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(6,2.4))
    
    im1 = ax1.imshow(np.log10(rho_P+1e-4).T, vmin=-4.3, vmax=0.3,
                     origin='upper', extent=Range,
                     )
    
    rat_xc = (rhoxc_P/(rho_P + 1e-7))
    im2 = ax2.imshow( rat_xc.T, vmin = -1.1, vmax = 1.1,
                      cmap='seismic',
                      origin='upper', extent=Range,
                     )
    if False:
        ax2.contour( rat_xc.T, levels=[-0.5,0.,0.5],
                     colors=('k','k','k'),
                     linestyles=('dashed', 'solid', 'dashdot'),
                     origin='upper', extent=Range,
                    )

    ax1.set_xlabel("CCSD density", fontsize=14)
    ax2.set_xlabel("Ratio, $n^S_{\\rm xc}/n_{\\rm H}$", fontsize=14)

    for ax, im in zip((ax1, ax2), (im1, im2)):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig_P.colorbar(im, cax=cax)
        if ax==ax1:
            cb.set_ticks([-4,-3,-2,-1,0])
            cb.set_ticklabels(['$10^{-4}$', '$10^{-3}$',
                               '$10^{-2}$', '$10^{-1}$', '1'],
                              rotation=45,)
        else:
            cb.set_ticks([-1,-0.5,0.,0.5,1])
            cb.set_ticklabels([-1,-0.5,0.,0.5,1],
                              rotation=45,)

    for ax in (ax1, ax2):
        ax.scatter(PHelp.xyz_Nuc_S[:,0], PHelp.xyz_Nuc_S[:,1],
                   color=NiceColour("Black"),
                   )
    
        ax.set_xticks([])
        ax.set_yticks([])

    fig_P.tight_layout(pad=0.1)

    if DoSave:
        fig_P.savefig("Dens_%s_%s.pdf"%(MolName, Basis))
    else:
        fig_P.savefig("Images/Dens_%s_%s.png"%(MolName, Basis))


plt.show()
