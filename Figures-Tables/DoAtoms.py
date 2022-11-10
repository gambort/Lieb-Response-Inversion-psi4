import numpy as np
from LibHelper import *

import yaml

import matplotlib.pyplot as plt
from NiceColours import *

DFA = "svwn"
Basis = "aug-cc-pvqz-decon"
BakBasis = "aug-cc-pvqz"

DataDir = "c:/Software/Python/Atoms/pyAtoms/Potentials/DataSets/"

CCSDData = {
'lipp': -4.49997877702815, 'lip': -7.27459105351561,
    'li': -7.40454763241343, 'lim': -7.4948301875506464,
    'cpp': -35.85815145374233, 'cp': -37.41962429904764,
    'c': -37.83133466182771, 'cm': -37.87416594073721,
    'fpp': -97.78822902155841, 'fp': -99.06763776593102,
    'f': -99.70321984510227, 'fm': -99.82110942210792,
}

IDs = {
    'lip':('li',2,1), 'li':('li',3,0), 'lim':('li',4,-1),
    'cp':('c',5,1), 'c':('c',6,0), 'cm':('c',7,-1),
    'fp':('f',8,1), 'f':('f',9,0), 'fm':('f',10,-1),
       }

Ref = {}
Ref['li'] = yaml.load(open(DataDir + "AllIonData_Li_GTO.yaml"),
                     Loader=yaml.UnsafeLoader)
Ref['c'] = yaml.load(open(DataDir + "AllIonData_C_GTO.yaml"),
                     Loader=yaml.UnsafeLoader)
Ref['f'] = yaml.load(open(DataDir + "AllIonData_F_GTO.yaml"),
                     Loader=yaml.UnsafeLoader)

for S in IDs:
    El, N, Q = IDs[S]
    
    if N<=2: kh = 0
    elif N<=4: kh = 1
    else: kh = 2
    
    NOcc, T, En_Iter, Fs_Iter, Fs_Inf, Misc \
        = GetProps("%s.mol"%(S), Basis, DFA=DFA, Units=1.)
    if T is None:
        print("Reverting to %s"%(BakBasis))
        NOcc, T, En_Iter, Fs_Iter, Fs_Inf, Misc \
            = GetProps("%s.mol"%(S), BakBasis, DFA=DFA, Units=1.)

    if kh<2:
        eps_h = Misc['epsilon_k'][kh]
    else:
        eps_h = np.mean(Misc['epsilon_k'][2:5])
        
    Ts = Misc['TsF'] # Fs_Inf

    Data = Ref[El]['IonData']["%.2f"%(N)]
    #Ts_Ref = Data['E0'] - Data['EExt'] - Data['EHxc']
    Ts_Ref = Data['Ts']

    eps_h_Ref = Data['PotentialData']['Ek'][kh]


    if Q==1: Sp=El+'pp'
    elif Q==0: Sp = El+'p'
    elif Q==-1: Sp = El
    
    eps_h_CCSD = CCSDData[S] - CCSDData[Sp]

    #print(S, Sp, eps_h_CCSD, eps_h_Ref)

    Lbl = El[0].upper() + El[1:]
    if Q==1: Lbl+="$^+$"
    elif Q==-1: Lbl+="$^-$"
    
    print("%-6s & %8.3f & %8.3f & %8.3f & %8.3f & %8.3f & %8.3f & %8.3f \\\\"\
          %(Lbl, Ts, Ts_Ref, Ts-Ts_Ref,
            eps_h, eps_h_Ref, eps_h-eps_h_Ref,
            eps_h_CCSD-eps_h_Ref))


        
