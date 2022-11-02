#!/home/timgould/psi4conda/bin/python3

import numpy as np
import psi4

from psi4Invert.LibInvert import *
from psi4Invert.LibReference import *

import itertools
import os


from  optparse import OptionParser

psi4.set_num_threads(4)
psi4.set_memory('8 GB')

psi4.set_output_file("__Invert-Lieb.out")

parser = OptionParser()
parser.add_option('-M', type="string", default='./SysDB/lih_1.50.xyz',
                  help="A molecule file in psi4 format -- be sure to set c1 symmetry or risk nonsense")
parser.add_option('--DFA', type="string", default="svwn",
                  help="Specify the DFA")
parser.add_option('--Basis', type="string", default="cc-pvtz",
                  help="Specify the basis set")

parser.add_option('--LevelShift', type="float", default=0.,
                  help="Use a level shift in SCF")

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

parser.add_option('--NIter', type="int", default=500,
                  help="Maximum inversion iterations")
parser.add_option('--En_Exit', type="float", default=1e-8,
                  help="Terminate when Hartree energy is this small")
parser.add_option('--a_Max', type="float", default=3.0,
                  help="Maximum allowed step")

parser.add_option('--eps_Cut', type="float", default=0.3,
                  help="Ignore eps>eps_Cut in response")
parser.add_option('--W_Cut', type="float", default=0.5,
                  help="Cut response after this fraction")

(Opts, args) = parser.parse_args()



psi4.set_options({
    "basis": Opts.Basis,
    "reference": "rhf",
    "level_shift": Opts.LevelShift,
})

GeomStr = """
0 1
Li
H 1 3.0
"""

if not(Opts.M is None):
    GeomStr = ReadGeom(Opts.M, "0 1")
    CationStr = ReadGeom(Opts.M, "1 2")

Mol = psi4.geometry(GeomStr)
if Mol.multiplicity()>1:
    psi4.set_options({"reference": "uhf"})

_, wfn_HF = psi4.energy("scf", return_wfn=True)

E0, wfn = psi4.energy("scf", dft_functional=GetDFA(Opts.DFA), return_wfn=True)
XHelp = InversionHelper(wfn)
RHelp = ReferenceHelper(Opts.M)
RHelp_plus = ReferenceHelper(Opts.M+"_plus")

D_HF = XHelp.SymHelp.Dense(wfn_HF.Da().to_array()) + XHelp.SymHelp.Dense(wfn_HF.Db().to_array())
F_HF = XHelp.SymHelp.Dense(wfn_HF.Fa().to_array())

E0_Ref, D0_Ref = RHelp.CalculateReference(XHelp, Force=Opts.ForceCCSD, D_Size = D_HF)

Mol = psi4.geometry(CationStr)
if Mol.multiplicity()>1:
    psi4.set_options({"reference": "uhf"})
EC_Ref, DC_Ref = RHelp_plus.CalculateReference(XHelp, Force=Opts.ForceCCSD, D_Size = D_HF)

N0 = np.vdot(XHelp.S_ao, D0_Ref)
NC = np.vdot(XHelp.S_ao, DC_Ref)

NOcc = int(np.ceil(N0/2.0000000001))


psi4.core.clean()

XHelp.SetInversionParameters(
    NIter = Opts.NIter, En_Exit = Opts.En_Exit,
    NAlwaysReport=3, NReport = min(100,int(np.ceil(Opts.NIter/5))),
    a_Max = Opts.a_Max,
    W_Cut = Opts.W_Cut,
)

F0 = XHelp.F * 1.

qq = np.linspace(0., 1., 11)
Tsq = 0.*qq
for kq, q in enumerate(qq):
    XHelp.f = np.ones((NOcc,))*2.
    XHelp.f[NOcc-1] = 2. - q

    D_Ref = (1.-q)*D0_Ref + q*DC_Ref
    
    print("Initialising the important pairs")
    XHelp.InitResponse(eps_Cut = Opts.eps_Cut)

    print("Doing the inversion")
    XHelp.InvertLiebResponse(D_Ref, F0 = F0)

    F0 = XHelp.F_Ref * 1. # For the next step

    
    # Fs_T = XHelp.Fs_Iter
    # Fs_T = Fs_T[Fs_T>0.]
    # print("*** %5d ***"%(len(Fs_T)))
    # T = np.arange(len(Fs_T))
    # p = np.polyfit(1./T[-100:]**1.5, Fs_T[-100:], 1)
    # Tsq[kq] = np.polyval(p, 0.) #XHelp.Ts_Ref

    Tsq[kq] = XHelp.GetTs(NLast=100)

print("q = np.linspace(0., 1., 11)")
print("Ts = [ " + ", ".join(["%.5f"%(x) for x in Tsq[0:4]]) + ",")
print("       " + ", ".join(["%.5f"%(x) for x in Tsq[4:8]]) + "," )
print("       " + ", ".join(["%.5f"%(x) for x in Tsq[8:]]) + "]")
