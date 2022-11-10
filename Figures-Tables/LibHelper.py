import numpy as np
import scipy.optimize as opt

NInf = 30 # use last this many points to extrapolate Fs

Ha = 1.
kcal = 627.5
eV = 27.21

def GetProps(Mol, Basis, DFA='pbe', Units=kcal, NWork=None):
    FileName = "Densities/Conv_%s_%s_%s.npz"%(Mol, Basis, DFA)
    try:
        X = np.load(FileName, allow_pickle=True)
    except:
        print(FileName + " does not exist")
        return None, None, None, None, None, None

    Occ = X['Occ']
    for k in range(len(Occ)):
        if Occ[k]>0.: NOcc = k+1

    
    En_Iter = X['En_Iter']
    Fs_Iter = X['Fs_Iter']
    if 'TsF' in X: Ts_Direct = X['TsF']
    else: Ts_Direct = None
    
    C_Ref = X['C_Ref']

    if not(NWork is None):
        En_Iter = En_Iter[:NWork]
        Fs_Iter = Fs_Iter[:NWork]

    if 'Ts0' in X: Ts0 = X['Ts0']
    else: Ts0 = 0.

    if 'TsF' in X: TsF = X['TsF']
    else: TsF = 0.

    eps_k = X['epsilon_Ref'][:(NOcc+1)]

    # Filter to non-zero (i.e. actually evaluated) value
    ii = En_Iter>0.
    En_Iter = En_Iter[ii]
    Fs_Iter = Fs_Iter[ii]
    T = np.arange(len(En_Iter))

    # Evaluate the FL error
    if len(T)>NInf:
        x = np.exp(-T[-NInf:]/166)
        y = Fs_Iter[-NInf:]
        
        p = np.polyfit(x, y, 1)
        Fs_Inf = np.polyval(p, 0.)

        # Check the quality of the fit
        y_Fit = np.polyval(p, x)
        ErrMetric = np.mean(np.abs(np.diff(y)))\
            / np.mean(np.abs(np.diff(y_Fit)))

        if ErrMetric>2.:
            print("Bad error [%.3f] in %s"%(ErrMetric, FileName))
            print("-- using Fs_Inf = min(Fs)-1e-5"%(ErrMetric))
            Fs_Inf = np.min(y)-1e-7
    else:
        if not(Ts_Direct is None):
            Fs_Inf = Ts_Direct
        else:
            Fs_Inf = Fs_Iter.min()-1e-7
    
    return NOcc, T, En_Iter*Units, Fs_Iter*Units, \
        Fs_Inf * Units, {'epsilon_k': eps_k * Units,
                         'Ts0':Ts0*Units, 'TsF':TsF*Units}
