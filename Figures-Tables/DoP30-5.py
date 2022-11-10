import numpy as np
from LibHelper import *

import yaml

import matplotlib.pyplot as plt
from NiceColours import *

DFA = "pbe0"
Basis = "cc-pvdz"

ShowSpecial = True

Special = [
    'SIE4x4:8', 'SIE4x4:7', 'SIE4x4:6', 'SIE4x4:5',
    'SIE4x4:4', 'SIE4x4:3',
    'DC13:8',
    'BH76:2', 'G2RC:21', 'G2RC:11'
]

Str = """
ALKBDE10_2_cao G2RC_10_66 SIE4x4_6_he2+_1.5 W4-11_128_s4-c2v
ALKBDE10_7_mgo G2RC_10_67 SIE4x4_7_he2+_1.75 W4-11_130_c2
BH76_1_n2 G2RC_20_1 W4-11_104_hnnn W4-11_134_fo2
BH76_1_n2ohts G2RC_20_13 W4-11_107_so3 W4-11_135_cloo
BH76_1_oh G2RC_20_34 W4-11_109_bn3pi W4-11_136_foof
BH76_9_hf G2RC_20_68 W4-11_113_n2o W4-11_137_o3
BH76_9_hf2ts SIE4x4_2_h2+_1.5 W4-11_114_c-hooo W4-11_138_bn
DC13_7_be4 SIE4x4_3_h2+_1.75 W4-11_116_p4 W4-11_41_sif4
G2RC_10_61 SIE4x4_4_he2+_1.0 W4-11_120_t-hooo W4-11_46_alf3
G2RC_10_62 SIE4x4_5_he2+_1.25 W4-11_124_no2
"""

Atoms = "al b be c ca cl f h he he_plus mg n o p s si"

MolList = Str.split()
AtomList = Atoms.split()

AllTs = {}
AllTs_DFA = {}

for S in MolList + AtomList:
    NOcc, T, En_Iter, Fs_Iter, Fs_Inf, Misc \
        = GetProps("%s.mol"%(S), Basis, DFA=DFA)
    
    if T is None: continue
    if En_Iter.min()>0.1: continue

    Fs_0 = Misc['Ts0']

    AllTs[S+'.xyz'] = Fs_Inf
    AllTs_DFA[S+'.xyz'] = Fs_0

    print("%-20s %9.1f %9.1f %6.1f %5.1f%%"\
          %(S, Fs_Inf, Fs_0,
            Fs_0-Fs_Inf, 100.*(Fs_0-Fs_Inf)/Fs_Inf))

Reactions = yaml.load(open("P30-5.yaml"), Loader=yaml.FullLoader)
Missing = []
Final = []
Final_DFA = []

X, LblX, Y, Z = [], [], [], []

KSpecial = []
KStandard = []
for KR, R in enumerate(list(Reactions)):
    print("Reaction %2d -- "%(KR+1), end='')
    
    Elements = Reactions[R][1:]
    WW, Ts, Ts_DFA = [], [], []
    for W,ID in [tuple(x) for x in Elements]:
        if ID in AllTs:
            WW += [W]
            Ts += [AllTs[ID]]
            Ts_DFA += [AllTs_DFA[ID]]
        else:
            Missing += [ID]

    if len(Ts)==len(Elements):
        DeltaTs = np.dot(WW, Ts)
        DeltaTs_DFA = np.dot(WW, Ts_DFA)
        print("Delta Ts = %6.1f [kcal/mol] %6.1f [DFA] %6.1f [Diff]"\
              %(DeltaTs, DeltaTs_DFA, DeltaTs_DFA-DeltaTs))
        if np.abs(DeltaTs)>200.:
            print(Reactions[R])
        
        Final += ["%6.1f"%(DeltaTs)]
        Final_DFA += ["%6.1f"%(DeltaTs_DFA)]

        X += [KR+1]
        LblX += [R]
        Y += [[DeltaTs, DeltaTs_DFA]]
        Z += [ Reactions[R][0] ]
    else:
        print("Skipped")
        Final += ["  --  "]
        Final_DFA += ["  --  "]


print("="*72)
print("Missing data:")
print(list(set(Missing)))

RList = ["%2d -- %-12s"%(K+1, R)
         for K,R in enumerate(list(Reactions))]
for K0 in range(0,30,5):
    print(", ".join(RList[K0:(K0+5)]))

print("="*72)
for K0 in (0,5,10,15,20,25):
    print(" & ".join(["%2d & %s"%(k+1,Final[k])
                      for k in range(K0,K0+5)]) + " \\\\")

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

for KX, R in enumerate(LblX):
    if R in Special:
        KSpecial += [KX]
    else:
        KStandard += [KX]


print("Number of systems computed = %4d"%(len(AllTs)))
print("ME  = %8.1f, MAE = %8.1f, MAPE = %8.1f%%"\
      %(np.mean(Y[:,1]-Y[:,0]),
        np.mean(np.abs(Y[:,1]-Y[:,0])),
        100.*np.mean(np.abs(Y[:,1]/Y[:,0]-1))))


print("Number of special systems = %4d"%(len(KSpecial)))
print("ME  = %8.1f, MAE = %8.1f, MAPE = %8.1f%%"\
      %(np.mean(Y[KSpecial,1]-Y[KSpecial,0]),
        np.mean(np.abs(Y[KSpecial,1]-Y[KSpecial,0])),
        100.*np.mean(np.abs(Y[KSpecial,1]/Y[KSpecial,0]-1))))


fig, ax = plt.subplots(1,1,figsize=(6,3))

def DoBar(ax, x, y, yCut=100., **kwargs):
    ax.bar(x, y, **kwargs)
    for k in np.argwhere(np.abs(y)>yCut).reshape((-1,)):
        AddBorder(
            ax.text(X[k]+0.05, np.sign(y[k])*yCut*0.5,
                    "%.0f"%(y[k]),
                    color=NiceColour("White"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=90,
                    fontsize=10,
                    ),
            w=0, fg='k',
        )
    

if True:
    PCut = 600
    P = (Y[:,1] - Y[:,0])

    def ff(y, y0=70):
        return y0*np.sign(y)*np.log(1. + np.abs(y)/y0)

    if ShowSpecial:
        ax.bar(X[KSpecial], ff(P[KSpecial]), width=0.9,
               color=NiceColour("Blue"),
               )
        ax.bar(X[KStandard], ff(P[KStandard]), width=0.9,
               color=NiceColour("Red"),
               )
    else:
        ax.bar(X, ff(P), width=0.9,
               color=NiceColour("Red"),
               )

    ax.scatter(X, ff(Y[:,0]), color=NiceColour("Orange"),
               marker="v",
               zorder=1000,)
    ax.scatter(X, ff(Y[:,1]), color=NiceColour("Cyan"),
               marker="^",
               zorder=1000,)
    ax.scatter(X, ff(Z), color=NiceColour("Black"),
               zorder=1000,)

    YTM = list(range(-100,101,10))
    YT = [-800,-400,-200,-100,-50,-20,0, 20, 50, 100, 200, 400, 800]
    ax.set_yticks(ff(YTM), minor=True)
    ax.set_yticks(ff(YT))
    ax.set_yticklabels(YT)

    ax.set_ylabel("$\\Delta T_s$ DFA vs inverted [kcal]", fontsize=14)
    ax.axis([X.min()-0.6, X.max()+0.6, ff(-1000), ff(1000)])
else:
    PCut = 300    
    P = (Y[:,1]/Y[:,0]-1.)*100.

    if ShowSpecial:
        ax.bar(X[KSpecial], P[KSpecial], width=0.9,
               color=NiceColour("Blue"),
               )
        ax.bar(X[KStandard], P[KStandard], width=0.9,
               color=NiceColour("Red"),
               )
    else:
        ax.bar(X, P, width=0.9,
               color=NiceColour("Red"),
               )

    for k in np.argwhere(np.abs(P)>PCut).reshape((-1,)):
        AddBorder(
            ax.text(X[k]+0.05, np.sign(P[k])*PCut*0.5,
                    "%.0f"%(P[k]),
                    color=NiceColour("White"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=90,
                    fontsize=10,
                    ),
            w=0, fg='k',
        )
    

    ax.set_ylabel("$\\Delta T_s$ DFA vs inverted [%]", fontsize=14)
    ax.axis([X.min()-0.6, X.max()+0.6, -PCut, PCut])
    
ax.set_xticks(X)
ax.set_xticklabels([int(x) for x in X], fontsize=10, rotation=90)
    
fig.tight_layout(pad=0.1)
fig.savefig("DTs-P30-5.pdf")
        
plt.show()
