#!/home/timgould/psi4conda/bin/python3

import numpy as np
import psi4

from psi4Invert.LibInvert import *
from psi4Invert.LibReference import *

import itertools
import os

import time

from  optparse import OptionParser

psi4.set_num_threads(4)
psi4.set_memory('8 GB')

psi4.set_output_file("__Invert-Lieb.out")

parser = OptionParser()
parser.add_option('-M', type="string", default='./SysDB/hydrogen_chloride.xyz',
                  help="A molecule file in psi4 or xyz format")
parser.add_option('--DFA', type="string", default="svwn",
                  help="Specify the DFA")
parser.add_option('--Basis', type="string", default="cc-pvdz",
                  help="Specify the basis set")

parser.add_option('--NoDF', action="store_true", default=False,
                  help="Run initial calculations without density fitting")

parser.add_option('--NoSym', dest="Sym", default=True, action="store_false",
                  help="Force symmetry to be C1")
parser.add_option('--EThermal', type="float", default=0.,
                  help="Use thermal smearing")

parser.add_option('--Safe', dest="Safe", default=False, action="store_true",
                  help="Use safer convergence parameters")

parser.add_option('--CCSD', default=False, action="store_true",
                  help="Use a CCSD density (old option)")
parser.add_option('--Reference', type="string", default="scf",
                  help="Reference density to use (e.g. scf, mp2. ccsd)")

parser.add_option('--SOnly', default=False, action="store_true",
                  help="Restrict the potential to S only (not implemented)")


parser.add_option('--Freeze', default=False, action="store_true",
                  help="Freeze the core")

parser.add_option('--ForceCCSD', default=False, action="store_true",
                  help="Force compuitation of a CCSD density")

parser.add_option('--NoNormPot', dest="NormPot", default=True, action="store_false",
                  help="Don't force potential to be normalized")

parser.add_option('--CalcPot', default=False, action="store_true",
                  help="Calculate the potential (should use a pure DFA)")
parser.add_option('--CalcPotc', default=False, action="store_true",
                  help="Calculate the correlation potential (should use DFA=pbe0_100)")
parser.add_option('--Calcdv', default=False, action="store_true",
                  help="Calculate dv only")
parser.add_option('--PotReg', type="float", default=1e-5,
                  help="Potential regularization (-ve for a scan)")

parser.add_option('--NIter', type="int", default=300,
                  help="Maximum inversion iterations")
parser.add_option('--En_Exit', type="float", default=1e-7,
                  help="Terminate when Hartree energy is this small")
parser.add_option('--a_Max', type="float", default=3.0,
                  help="Maximum allowed step")

parser.add_option('--eps_Cut', type="float", default=3.0,
                  help="Ignore eps>eps_Cut in response")
parser.add_option('--W_Cut', type="float", default=0.5,
                  help="Cut response after this fraction")
parser.add_option('--ResponseEvery', type="int", default=1,
                  help="Update response every this steps")

(Opts, args) = parser.parse_args()

if Opts.CCSD: Opts.Reference = "ccsd"

psi4.set_options({
    "basis": Opts.Basis,
    "reference": "rhf",
    "freeze_core": Opts.Freeze,
    "scf_type": "df",
    "e_convergence": 1e-6,
    "r_convergence": 1e-6,
    #"cc_type": "df",
    #"df_ints": "save",
    #"df_basis_scf": "aug-cc-pv5z-decon",
})

if Opts.Safe: # Safer calculations for tricky problems
    psi4.set_options({
        #"mom_start": 2,
        "level_shift": 1.5,
        "damping_percentage": 80.,
        #"SCF_INITIAL_ACCELERATOR": None,
        "DIIS_START_CONVERGENCE": 1e-9,
        'diis': False,
        'maxiter': 300,
    })    

if Opts.NoDF: # No density fitting
    psi4.set_options({
        "scf_type": "pk",
        "cc_type": "cd",
    })
    
GeomStr = """
0 1
Li
H 1 3.0
"""

print("="*72)
print("Running %s @ %s starting from %s"%(Opts.M, Opts.Basis, Opts.DFA))
print("="*72)

if not(Opts.M is None):
    GeomStr = ReadGeom(Opts.M)

if not(Opts.Sym):
    GeomStr += "\nsymmetry c1\n"

Mol = psi4.geometry(GeomStr)
if Mol.multiplicity()>1:
    psi4.set_options({"reference": "uhf"})

_, wfn_HF = psi4.energy("scf", return_wfn=True)

E0, wfn = psi4.energy("scf", dft_functional=GetDFA(Opts.DFA), return_wfn=True)
XHelp = InversionHelper(wfn)
RHelp = ReferenceHelper(Opts.M, Level=Opts.Reference)

print("Occupation factors")
print(XHelp.f)

D_HF = XHelp.SymHelp.Dense(wfn_HF.Da().to_array()) + XHelp.SymHelp.Dense(wfn_HF.Db().to_array())
F_HF = XHelp.SymHelp.Dense(wfn_HF.Fa().to_array())
E_Ref, D_Ref = RHelp.CalculateReference(XHelp, Force=Opts.ForceCCSD, D_Size = D_HF)

fno, _ = la.eigh(-D_Ref, la.inv(XHelp.S_ao))
fno = -fno[np.abs(fno)>1e-4]

print("Natural occupation factors")
print(fno)

psi4.core.clean()


XHelp.SetInversionParameters(
    NIter = Opts.NIter, En_Exit = Opts.En_Exit,
    NAlwaysReport=3, NReport = min(100,int(np.ceil(Opts.NIter/5))),
    a_Max = Opts.a_Max,
    W_Cut = Opts.W_Cut,
    ResponseEvery = Opts.ResponseEvery,
    EThermal = Opts.EThermal,
    SOnly = Opts.SOnly,
)

t0 = time.perf_counter()
print("Initialising the important pairs")
XHelp.InitResponse(eps_Cut = Opts.eps_Cut)

print("Doing the inversion")
XHelp.InvertLiebResponse(D_Ref)

t1 = time.perf_counter()
print("-- time taken for inversion = %.3f s"%(t1-t0))

if Opts.M is None: quit()
CoreFileName = os.path.basename(Opts.M)
if not(Opts.Reference.lower()=="ccsd"):
    CoreFileName += "_"+Opts.Reference.upper()
if (Opts.a_Max > 10.):
    CoreFileName += "_am%.1f"%(Opts.a_Max)
    

if Opts.Calcdv and not(Opts.DFA.upper() in ("HF", "SCF")):
    PHelp = PotentialHelper(XHelp, eta=Opts.PotReg)
    xyz, w = PHelp.xyz, PHelp.w

    print("="*72)
    print("Finding xc potential")
    PHelp.Reconstruct(
        Mode = "xc",
        Normalize = True,
        ForceUpdate = True,
    )
    print("="*72)

# Calculate the linear response error factor
VxcFactor = XHelp.GetVxcFactor()


np.savez("Densities/Conv_%s_%s_%s.npz"%(CoreFileName, Opts.Basis.lower(), Opts.DFA.lower()),
         xyz_Nuc = Mol.geometry().to_array(dense=True),
         Occ = XHelp.f, S = XHelp.S_ao, H = XHelp.H_ao,
         C = XHelp.C,
         C_Ref = XHelp.C_Ref, epsilon_Ref = XHelp.epsilon_Ref,
         D_Ref = XHelp.D_Ref, D_DFA = XHelp.D,
         F_Ref = XHelp.F_Ref, F_HF_Ref = XHelp.F_HF_Ref, F0 = XHelp.F,
         FMF_Ref = XHelp.FMF_Ref, FMF0 = XHelp.FMF0,
         Ts = XHelp.Ts_Ref, Ts0 = XHelp.Ts0, TsF = XHelp.Ts_Direct,
         VxcFactor = VxcFactor,
         En_Iter = XHelp.En_Iter, Fs_Iter = XHelp.Fs_Iter,
)

if Opts.CalcPot or Opts.CalcPotc \
   and not(Opts.DFA.upper() in ("HF", "SCF")):
    PHelp = PotentialHelper(XHelp, eta=Opts.PotReg)
    xyz, w = PHelp.xyz, PHelp.w

    print("="*72)
    print("Finding xc potential")
    rho, rhoxc = PHelp.Reconstruct(
        Mode = "xc",
        Normalize = Opts.NormPot,
        ForceUpdate = True,
        Debug=True,)
    print("="*72)

    if Opts.CalcPotc:
        print("="*72)
        print("Finding c potential")
        rho, rhoc = PHelp.Reconstruct(
            Mode = "c",
            Normalize = Opts.NormPot,
            Debug=True)

        print("="*72)

        rhox = rhoxc - rhoc
        print("<n> = %.4f, <n_xc> = %.4f, <n_c> = %.4f"\
              %(np.dot(w, rho), np.dot(w, rhoxc), np.dot(w, rhoc)))
    else:
        rhox, rhoc = 0., 0.
        print("<n> = %.4f, <n_xc> = %.4f"\
              %(np.dot(w, rho), np.dot(w, rhoxc)))


    rho_u = PHelp.rho_u
    np.savez("Densities/rho_%s_%s_%s.npz"%(CoreFileName, Opts.Basis.lower(), Opts.DFA.lower()),
             epsilon_Vs = XHelp.epsilon_Ref,
             xyz=xyz, w=w,
             rho=rho, rho_u = rho_u, rhoxc=rhoxc, rhox=rhox, rhoc=rhoc,
             xyz_Nuc = Mol.geometry().to_array(dense=True),
    )
    
print("="*72)
print("Ts(Max   ) = %8.4f Ha"%(XHelp.Ts_Ref))
print("Ts(Extrap) = %8.4f Ha"%(XHelp.Ts_Extrap))
print("Ts(DFA   ) = %8.4f Ha"%(XHelp.Ts0))
print("epsilon(HOMO) = %8.4f Ha (%8.1f eV)"%(XHelp.epsilon_Ref[XHelp.kh], XHelp.epsilon_Ref[XHelp.kh]*27.211))
print("epsilon(LUMO) = %8.4f Ha (%8.1f eV)"%(XHelp.epsilon_Ref[XHelp.kl], XHelp.epsilon_Ref[XHelp.kl]*27.211))
print("="*72)
