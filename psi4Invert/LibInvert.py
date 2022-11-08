import psi4
import numpy as np
import scipy.linalg as la

from psi4Invert.LibPairDens import GetDensities

import itertools

eV = 27.211

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

###############################################
def ReadGeom(FileName, Hdr = "0 1"):
    try:
        F = open(FileName)
        if FileName[-3:]=="xyz":
            N = int(F.readline())
            F.readline()
            GS = "%s\n\n"%(Hdr)
            for i in range(N):
                GS += F.readline()
            print("xyz file :")
            print(GS)
            print("end xyz file")
        else:
            GS = "".join(list(F))
        F.close()
        return GS
    except:
        print("Quitting! Could not load %s"%(FileName))
        quit()

##### Process density-fitting ####

def GetDensityFit(wfn, basis, mints, aux_basis=None):
    if aux_basis is None:
        aux_basis = psi4.core.BasisSet.build\
            (wfn.molecule(), "DF_BASIS_SCF", "",
             "RIFIT", basis.name())
        
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    SERIApq = np.squeeze(mints.ao_eri(aux_basis, zero_basis, basis, basis))
    metric = mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
    SERIAB = np.squeeze(metric.to_array(dense=True))

    metric.power(-0.5, 1e-14)
    metric = np.squeeze(metric)
    ERIA = np.tensordot(metric, SERIApq, axes=[(1,),(0,)])

    SAB = mints.ao_overlap(aux_basis, aux_basis).to_array(dense=True)

    QA = metric.dot(np.squeeze(mints.ao_overlap(aux_basis, zero_basis)))

    return ERIA, SERIAB, SAB, QA, aux_basis

##### This is a hack to convert a UKS superfunctional to its RKS equivalent
# Internal routine
# https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/dft/dft_builder.py#L251
# sf_from_dict =  psi4.driver.dft.build_superfunctional_from_dictionary
# # My very hacky mask
# def sf_RKS_to_UKS(DFA):
#     DFA_Dict = { 'name':DFA.name()+'_u'}
#     DFA_Dict['x_functionals']={}
#     DFA_Dict['c_functionals']={}
#     for x in DFA.x_functionals():
#         Name = x.name()[3:]
#         alpha = x.alpha()
#         DFA_Dict['x_functionals'][Name] = {"alpha": alpha,}
#     for c in DFA.c_functionals():
#         Name = c.name()[3:]
#         alpha = c.alpha()
#         DFA_Dict['c_functionals'][Name] = {"alpha": alpha,}

#     npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
#     DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)
#     return DFAU
##### End hack

# For nice debug printing
def NiceArr(X):
    return "[ %s ]"%(",".join(["%8.3f"%(x) for x in X]))
def NiceArrInt(X):
    return "[ %s ]"%(",".join(["%5d"%(x) for x in X]))
def NiceMat(X):
    N = X.shape[0]
    if N==0:
        return "[]"
    elif N==1:
        return "["+NiceArr(X[0,:])+"]"
    elif N==2:
        return "["+NiceArr(X[0,:])+",\n "+NiceArr(X[1,:])+"]"
    else:
        R = "["
        for K in range(N-1):
            R+=NiceArr(X[K,:])+",\n "
        R+=NiceArr(X[N-1,:])+"]"
        return R

# Handle PBE0_XX calculations
def GetDFA(DFA):
    if DFA[:5].lower()=="pbe0_":
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: f_c = max(float(X[2])/100.,1e-5)
        else: f_c = 1.
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_PBE": {"alpha":1.-alpha, }},
            'c_functionals': {"GGA_C_PBE": {"alpha":f_c, }},
            'x_hf': {"alpha":alpha, },
            }
    else:
        return DFA

# Get the degeneracy of each orbital
def GetDegen(epsilon, eta=1e-5):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen

#################################################################################################
# This code handles degeneracies detected and used by psi
#################################################################################################

class SymHelper:
    def __init__(self, wfn):
        self.NSym = wfn.nirrep()
        self.NBasis = wfn.nmo()
        
        self.eps_so = wfn.epsilon_a().to_array()
        self.C_so = wfn.Ca().to_array()
        self.ao_to_so = wfn.aotoso().to_array()
        
        if self.NSym>1:
            self.eps_all = np.hstack(self.eps_so)
            self.k_all = np.hstack([ np.arange(len(self.eps_so[s]), dtype=int)
                                       for s in range(self.NSym)])
            self.s_all = np.hstack([ s * np.ones((len(self.eps_so[s]),), dtype=int)
                                       for s in range(self.NSym)])
        else:
            self.eps_all = self.eps_so * 1.
            self.k_all = np.array(range(len(self.eps_all)))
            self.s_all = np.zeros((len(self.eps_all),), dtype=int)

        self.ii_sorted = np.argsort(self.eps_all)
        self.eps_sorted = self.eps_all[self.ii_sorted]
        self.k_sorted = self.k_all[self.ii_sorted]
        self.s_sorted = self.s_all[self.ii_sorted]

        self.ks_map = {}
        for q in range(len(self.ii_sorted)):
            self.ks_map[(self.s_sorted[q], self.k_sorted[q])] = q

    # Do a symmetry report to help identifying orbitals
    def SymReport(self, kh, eta=1e-5):
        epsh = self.eps_sorted[kh] + eta
        print("Orbital indices by symmetry - | indicates virtual:")
        for s in range(self.NSym):
            Str = "Sym%02d : "%(s)
            eps = self.eps_so[s]
            if not(hasattr(eps, '__len__')) or len(eps)==0: continue

            kk_occ = []
            kk_unocc = []
            for k, e in enumerate(eps):
                if e<epsh: kk_occ += [ self.ks_map[(s,k)] ]
                else: kk_unocc += [ self.ks_map[(s,k)] ]

            Arr = ["%3d"%(k) for k in kk_occ] + [" | "] \
                + ["%3d"%(k) for k in kk_unocc]
            if len(Arr)<=16:
                print("%-8s"%(Str) + " ".join(Arr))
            else:
                for k0 in range(0, len(Arr), 16):
                    kf = min(k0+16, len(Arr))
                    if k0==0:
                        print("%-8s"%(Str) + " ".join(Arr[k0:kf]))
                    else:
                        print(" "*8 + " ".join(Arr[k0:kf]))

    # Report all epsilon
    def epsilon(self):
        return self.eps_sorted

    # Report a given orbital, C_k
    def Ck(self, k):
        if self.NSym==1:
            return self.C_so[:,k]
        else:
            s = self.s_sorted[k]
            j = self.k_sorted[k]

            return self.ao_to_so[s].dot(self.C_so[s][:,j])

    # Report all C
    def C(self):
        if self.NSym==1:
            return self.C_so * 1.
        else:
            C = np.zeros((self.NBasis, self.NBasis))
            k0 = 0
            for k in range(self.NSym):
                C_k = self.ao_to_so[k].dot(self.C_so[k])
                dk = C_k.shape[1]
                C[:,k0:(k0+dk)] = C_k
                k0 += dk
            return C[:,self.ii_sorted]

    # Convert the so matrix to dense form
    def Dense(self, X):
        if self.NSym==1:
            return X
        else:
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Convert the ao matrix to pseudo-sparse form
    def Sparse(self, X):
        if self.NSym==1:
            return X
        else:
            print("Sparse is not working!")
            quit()
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Solve a Fock-like equation using symmetries
    # if k0>0 use only the subspace spanned by C[:,k0:]
    def SolveFock(self, F, S=None, k0=-1):
        # Note, k0>=0 means to solve only in the basis from C[:,k0:]
        if self.NSym==1:
            if k0<=0:
                return la.eigh(F, b=S)
            else:
                # FV = SVw
                # V=CU
                # FCU = SCUw
                # (C^TFC)U = (C^TSC)Uw
                # XU = Uw
                
                C = self.C()[:,k0:]
                F_C = (C.T).dot(F).dot(C)
                w, U = la.eigh(F_C)
                return w, C.dot(U)
        else:
            k0s = [0]*self.NSym
            ws = [None]*self.NSym
            Cs = [None]*self.NSym

            if k0>0:
                # Use no terms
                for s in range(self.NSym):
                    k0s[s] = self.NBasis
                    
                # Evaluate the smallest k value for each symmetry
                for i in range(k0,self.NBasis):
                    s = self.s_sorted[i]
                    k0s[s] = min(k0s[s],self.k_sorted[i])

            for s in range(self.NSym):
                # Project onto the subset starting at k0s
                C_ao = self.ao_to_so[s].dot(self.C_so[s][:,k0s[s]:])
                F_C = (C_ao.T).dot(F).dot(C_ao)
                if not(S is None):
                    S_C = (C_ao.T).dot(S).dot(C_ao)
                else: S_C = None

                if F_C.shape[0]>0:
                    ws[s], Us = la.eigh(F_C, b=S_C)
                    Cs[s] = C_ao.dot(Us)
                else:
                    ws[s] = []
                    Cs[s] = [[]]

            # Project back onto the main set
            k0 = max(k0, 0)
            NShift = self.NBasis - k0
            w = np.zeros((NShift,))
            C = np.zeros((self.NBasis,NShift))
            for i in self.ii_sorted[k0:]:
                s = self.s_sorted[i]
                k = self.k_sorted[i]                
                w[i-k0] = ws[s][k-k0s[s]]
                C[:,i-k0] = Cs[s][:,k-k0s[s]]
                
            return w, C
        

def ThermalOcc(E, Eth, NOcc, Cut=50, muOnly=False):
    NI = int(np.floor(NOcc))
    fRem = NOcc - 2*(NI//2)
    epsC = E[NI//2-1] # Core energy
    epsF = E[NI//2+0] # Frontier energy
    
    if np.abs(fRem)<1e-4:
        mu = epsC
    else:
        mu = epsF - Eth*np.log(2./fRem-1)
        
    for iter in range(10):
        q = (E-mu)/Eth
        q = np.minimum(np.maximum(q, -Cut), Cut)
        x = np.exp(q)
        
        f = 2/(1+x)
        df = 2/Eth * x/(1+x)**2
    
        if np.abs(np.sum(f)-NOcc)<1e-6:
            break
        
        mu -= (np.sum(f)-NOcc)/np.sum(df)
        

    if muOnly: return mu0
    return f[f>1e-6]

#################################################################################################
# This is the main Inversion code
#################################################################################################

class InversionHelper:
    def __init__(self, wfn,
                 aux_basis = None, # Can pass in an aux_basis
                 Report=10,
    ):
        self.Report = Report
        
        self.wfn = wfn

        self.SymHelp = SymHelper(wfn)
        
        self.F = self.SymHelp.Dense(wfn.Fa().to_array())
        self.H = self.SymHelp.Dense(wfn.H().to_array())
        self.Da = self.SymHelp.Dense(wfn.Da().to_array())
        self.Db = self.SymHelp.Dense(wfn.Db().to_array())
        self.D = self.Da + self.Db
        self.epsilon = self.SymHelp.epsilon()
        self.C = self.SymHelp.C()

        basis = wfn.basisset()
        self.basis = basis
        self.nbf = self.wfn.nmo() # Number of basis functions
        self.NAtom = self.basis.molecule().natom()
        
        self.mints = psi4.core.MintsHelper(self.basis)
        self.S_ao = self.mints.ao_overlap().to_array(dense=True)
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True)
        self.V_ao = self.mints.ao_potential().to_array(dense=True)
        self.H_ao = self.T_ao + self.V_ao
        self.Di_ao = [x.to_array(dense=True) for x in self.mints.ao_dipole()]


        X = self.H_ao - self.H
        if np.mean(X**2)>1e-8:
            print("Inconsistent ao expansions")
            quit()
        
        self.ERIA, self.SERIAB, self.SAB, self.QA, self.aux_basis \
            = GetDensityFit(self.wfn, self.basis, self.mints, aux_basis)

        self.NBas = wfn.nmo()
        self.NOcc = wfn.nalpha()
        self.kh = wfn.nalpha()-1
        self.kl = self.kh+1
        self.Degen = GetDegen(self.epsilon)

        if wfn.nalpha() == wfn.nbeta():
            self.f = 2.*np.ones((self.NOcc,))
        else:
            self.f = np.ones((wfn.nalpha(),))
            self.f[:wfn.nbeta()] += 1.
        self.kDegen = None
            
        if self.SymHelp.NSym>1:
            self.SymHelp.SymReport(self.kh)


        if self.Report>0:
            print("f   = %s"%(NiceArr(self.f[max(0,self.kh-2):min(self.NBas,self.kl+3)])))
            print("eps = %s eV"%(NiceArr(self.epsilon[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))
            print("eps_h = %8.3f/%8.2f, eps_l = %8.3f/%8.2f [Ha/eV]"\
                  %(self.epsilon[self.kh], self.epsilon[self.kh]*eV,
                    self.epsilon[self.kl], self.epsilon[self.kl]*eV,))

        self.Params = {}
        self.SetInversionParameters(
            En_Exit = 1e-5, NIter = 100, # Iteration parameters
            NAlwaysReport = 5, # Always show this many first
            NReport = 20, # Report every this steps
            a_Max = 3., # Convergence parameters
            W_Cut = 0.50, W_Most = 0.90, En_Most = 1e-9, # Acceleration parameters
            EThermal = None, # Set to about 10 mHa for better convergence of radicals
            LevelShift = None,
        )

        self.IJ = None

    def UpdateBasis(self, wfn_new):
        basis = wfn_new.basisset()
        self.mints = psi4.core.MintsHelper(basis)
        self.S_ao = self.mints.ao_overlap().to_array(dense=True)
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True)
        self.V_ao = self.mints.ao_potential().to_array(dense=True)
        self.H_ao = self.T_ao + self.V_ao
        self.Di_ao = [x.to_array(dense=True) for x in self.mints.ao_dipole()]

        self.ERIA, self.SERIAB, self.SAB, self.QA, self.aux_basis \
            = GetDensityFit(None, self.basis, self.mints, self.aux_basis)

    def SetOcc(self, f=None):
        if not(f is None):
            self.NOcc = len(f)
            for k in range(len(f)):
                if f[k]>0.: self.NOcc = (k+1)
            self.f = np.array(f)
            self.kh = self.NOcc-1
            self.kl = self.NOcc

    def SetDegen(self): # Check for degenerate frontier orbitals
        epsh = self.epsilon[self.kh]
        kk = []
        for k in range(len(self.f)):
            if np.abs(self.epsilon[k]-epsh):
                kk += [k,]
        if len(kk)>0: self.kDegen = kk

    def GetVHA(self, D=None, C1=None, C2=None):
        # Prefer C1, C2 if given
        if not(C1 is None):
            # C2 defaults to C1
            if C2 is None: C2 = C1

            t = np.tensordot(self.ERIA, C1, axes=((2,),(0,)))
            t = np.tensordot(t, C2, axes=((1,),(0,)))
        else:
            t = np.tensordot(self.ERIA, D, axes=((1,2,),(0,1)))

        return t

    def GetVH(self, D=None, C1=None, C2=None, q=None):
        if q is None:
            return np.tensordot(self.ERIA, self.GetVHA(D=D, C1=C1, C2=C2),
                                axes=((0,),(0,)))
        else:
            return np.tensordot(self.ERIA, q, axes=((0,),(0,)))


    def GetEH(self, D=None, C1=None, C2=None):
        VH = 0.5*self.GetVH(D=D, C1=C1, C2=C2)
        # Prefer C1, C2 if given
        if not(C1 is None):
            # C2 defaults to C1
            if C2 is None: C2 = C1
            return C1.dot(VH).dot(C2)
        else:
            return np.vdot(D, VH)        

    def GetVx(self, D):
        # [A|pq] Drq -> ([A|p*] Dr*)
        t = np.tensordot(self.ERIA, D, axes=((2,), (1,)))
        # [A|qr] ([A|p*] Dr*) -> ( [A|q*] [A|p*] D** )
        # = [pr|qs] Drs
        return np.tensordot(self.ERIA, t, axes=((0,2), (0,2)))

    def GetEx(self, D):
        Vx = 0.5*self.GetVx(D)
        return np.vdot(D, Vx)

    def SetInversionParameters(self, **kwargs):
        for P in kwargs:
            self.Params[P] = kwargs[P]
            if self.Report>10:
                print("%s = %s"%(P, str(kwargs[P])))
        
    def InitResponse(self, f=None, epsilon=None, C=None,
                     eps_Cut = 1e5,
                     f_Cut = 1e-4,
                     N = None, Quick=False):
        if f is None: f = self.f
        if epsilon is None: epsilon = self.epsilon
        if C is None: C = self.C

        # Thermal overrides
        if not(self.Params['EThermal'] is None) \
               and not(self.Params['EThermal'] < 1e-4):
            N_Ref = np.sum(f)
            f = ThermalOcc(epsilon, self.Params['EThermal'], N_Ref)

        
        # By default do all < epsilon_Cut
        if N is None:
            N = len(epsilon[epsilon<eps_Cut])

        # Ensure we have at least one excited state
        N = max(len(f)+1, N)

        fa = np.zeros((N,))
        if len(f)<=N: fa[:len(f)] = f
        else: fa = f[:N]

        if Quick:
            d2 = np.zeros((N,))
            for i in range(N):
                d = self.GetVHA(C[:,i], C[:,i])
                d2[i] = np.dot(d,d)

        NTot = 0
        for i, j in itertools.product(range(N), range(N)):
            if (j>i) and (np.abs(fa[i]-fa[j])>f_Cut): NTot +=1


        if NTot>1000: # Show updates
            print("Progress: ", end='', flush=True)
            NDone = 0
        IJ = []
        W = []
        for i, j in itertools.product(range(N), range(N)):
            if (j>i) and (np.abs(fa[i]-fa[j])>f_Cut):
                if Quick: dij = np.sqrt(d2[i]*d2[j])
                else:
                    d = self.GetVHA(C1=C[:,i], C2=C[:,j])
                    dij = np.dot(d,d)

                deps = epsilon[i]-epsilon[j]
                x = 4.*(fa[i]-fa[j]) * deps / (deps**2 + 1e-6)

                IJ += [(i,j)]
                W += [x*dij]

                if NTot>1000: # Show updates
                    if (NDone%(NTot//50)==0): print("X", end='', flush=True)
                    NDone+=1
        if NTot>1000: # Show updates
            print()
            
        kk = np.argsort(W)
        IJ = [IJ[k] for k in kk]
        W = np.array(W)/np.sum(W)
        W = W[kk]

        #print("*** NIJ = %d, NTot = %d, N = %d ***"%(len(IJ), NTot, N))
        #print(NiceArr(epsilon))
        #quit()

        NAll = len(IJ)
        NCut10 = max(min(10, NAll), int(np.ceil(NAll/20))) # 10% or 10, whichever is the larger
        
        self.IJ_All = IJ

        if self.Params['W_Cut' ]>=1.0:
            NCut = NAll
            NMost = NAll
            print(self.Params['W_Cut' ], NCut, NAll)
        else:
            NCut = NCut10
            NMost = NCut10
            for i in range(NCut10,NAll):
                WC = np.sum(W[:(i+1)])
                if WC<=self.Params['W_Most']: NMost = i+1
                if WC<=self.Params['W_Cut' ]: NCut  = i+1

        self.IJ_Most = IJ[:NMost]
        self.IJ = IJ[:NCut ]

        if self.Report>0:
            print("Selecting %4d/%.1f%% (or %4d/%.1f%%) response terms out of %4d"\
                  %(NCut, 100.*NCut/NAll, NMost, 100.*NMost/NAll, NAll))

        return self.IJ

    def GetResponse(self, f, epsilon, C, Mode="Cut"):
        # Ensure f is right size
        N = len(epsilon)
        fa = np.zeros((N,))
        if len(f)<=N: fa[:len(f)] = f
        else: fa = f[:N]

        if Mode.upper() in ("M", "MOST"):
            IJ = self.IJ_Most
        elif Mode.upper() in ("A", "ALL"):
            IJ = self.IJ_All
        else:
            IJ = self.IJ

        X = 0.
        for i, j in IJ:
            d = self.GetVHA(C1=C[:,i], C2=C[:,j])
            deps = (epsilon[i]-epsilon[j])
            x = 4.*(fa[i]-fa[j]) * deps/(deps**2 + 1e-6)
            X += x*np.outer(d,d)

        return X

    def InvertLiebResponse(self, D_Ref, q_Ref=None,
                           F0 = None, IP=None):
        # D_Ref is None uses the current D but switches off Lieb max
        if D_Ref is None:
            D_Ref = self.Da
            UseFsMax = False
        else: UseFsMax = True
            
        # This over-rides D_Ref if provided
        if q_Ref is None:
            q_Ref = self.GetVHA(D=D_Ref)


        f = self.f
        N_Ref = np.vdot(self.S_ao, D_Ref)
            
        def QSolve(F):
            eps0, C0 = self.SymHelp.SolveFock(F, self.S_ao)

            if not(self.Params['EThermal'] is None) \
               and not(self.Params['EThermal'] < 1e-3):
                f = ThermalOcc(eps0, self.Params['EThermal'], N_Ref)
            else:
                f = self.f
                
            D0 = np.einsum('k,pk,qk->pq', f, C0[:,:len(f)], C0[:,:len(f)])
            q0 = self.GetVHA(D=D0)
            
            q = q0 - q_Ref
            En = 0.5 * np.dot(q, q)

            return F, eps0, C0, D0, q, En, f
            

        if F0 is None: F0 = self.F*1.
        else: F0 = F0*1.
        
        F0, eps0, C0, D0, q, En, f = QSolve(F0)

        D00 = 1.*D0
        self.Ts0 = np.tensordot(self.T_ao, D00)
        
        F0 = self.F * 1.
        V0 = self.F - self.T_ao
        vA = 0. * q

        # Make sure the response is initalized
        if self.IJ is None:
            self.InitResponse(f, eps0, C0)

        a = 0.
        FsMax = -1e10
        EnMin =  1e10
        F0Opt, vAOpt = None, None

        self.En_Iter = np.zeros((self.Params['NIter']))
        self.Fs_Iter = np.zeros((self.Params['NIter']))
            
        for iteration in range(self.Params['NIter']):
            F0, eps0, C0, D0, q, En, f = QSolve(F0)
            
            self.En_Iter[iteration] = En

            if UseFsMax:
                Fs = np.dot(f, eps0[:len(f)]) - np.vdot(V0, D_Ref)
            else:
                Fs = np.tensordot(D0, self.T_ao)

            self.Fs_Iter[iteration] = Fs

            # Maximal Fs if UseFsMax otherwise minimal En
            if (Fs>FsMax and UseFsMax) \
               or (En<EnMin and not(UseFsMax)):
                iterMin = iteration
                EnMin = En
                FsMax = Fs*1.
                F0Opt = F0*1.
                vAOpt = vA*1.

            if (iteration<self.Params['NAlwaysReport']) \
               or (iteration%self.Params['NReport']==0):
                Ts = np.vdot(self.T_ao, D0)

                # Dipole
                X = [np.vdot(x, D0-D_Ref) for x in self.Di_ao]
                DV = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)

                if self.Report>-1:
                    print("%4d | %8.4f %8.4f | %8.4f | %8.4f | %10.7f | a = %.3f"\
                          %(iteration, Fs, Ts,  DV,  np.dot(vA,vA), En,  a))
                    # if not(self.Params['EThermal'] is None) \
                    #    and not(self.Params['EThermal'] < 1e-3): 1
                    #print(NiceArr(q))
                    #print("- f   = %s"%(NiceArr(f[max(0,self.kh-2):min(self.NBas,self.kl+3)])))
                    #print("- eps = %s eV"%(NiceArr(eps0[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))

            if En < self.Params['En_Exit']: break
                
            if En < self.Params['En_Most']:
                X = self.GetResponse(f, eps0, C0, Mode='Most')
            else:
                X = self.GetResponse(f, eps0, C0)

            A1 = q.dot(X).dot(q)
            A2 = q.dot(X).dot(X).dot(q)

            if A1**2>2.*A2*En:
                a = (-A1 + np.sqrt(A1**2 - 2*A2*En)) / (2*A2)
            elif np.abs(A1)>0.:
                a = -En/A1
            else:
                a = self.Params['a_Max']

            a = min(max(a, 1e-3), self.Params['a_Max'])

            DF = a*np.einsum('Apq,A', self.ERIA, q, optimize=True)
            F0 += DF
            V0 += DF
            vA += a*q

            eps0, C0 = self.SymHelp.SolveFock(F0, self.S_ao)

            if not(self.Params['LevelShift'] is None):
                epsr = eps0
                epsr[self.NOcc:] += self.Params['LevelShift']

                SC0 = np.dot(self.S_ao, C0)
                F0 = np.einsum('pk,qk,k->pq', SC0, SC0, epsr)

        if UseFsMax:
            if iteration>50:
                iX0 = max(iteration-100,50)
                iXF = min(iX0+100, iteration)       
                pX = np.polyfit(np.exp(-np.arange(iX0, iXF)/166), self.Fs_Iter[iX0:iXF], 1)
                FsX = np.polyval(pX, 0.)
            else:
                FsX = FsMax
        else:
            kMin = np.argmin(self.En_Iter)
            FsX = self.Fs_Iter[kMin]


        self.Ts_Extrap = FsX

        F0Opt, eps0, C0, D0, q, En, f = QSolve(F0Opt)
        if not(self.Params['EThermal'] is None) \
           and not(self.Params['EThermal'] < 1e-3):
            self.f = f
            self.NOcc = len(f[f>1e-7])
            
        Ts = np.vdot(self.T_ao, D0)
        if UseFsMax:
            Fs = np.dot(f, eps0[:len(f)]) - np.vdot(V0, D_Ref)
        else:
            Fs = Ts

        # Dipole
        X = [np.vdot(x, D0-D_Ref) for x in self.Di_ao]
        DV = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)


        if self.Report>-1:
            print("%4d | %8.4f %8.4f | %8.4f | %8.4f | %10.7f"\
                  %(iterMin, Fs, Ts,  DV,  np.dot(vAOpt, vAOpt), EnMin))

        if FsX-FsMax<0.:
            print("*** WARNING! Inconsistent Fs values ***")
            
        print("FsMax = %9.4f [Ref.   ], FsX   = %9.4f [%7.4f]"\
              %(FsMax, FsX, FsX-FsMax))
        print("TsF   = %9.4f [%7.4f], Ts0   = %9.4f [%7.4f]"\
              %(Ts, Ts-FsMax, self.Ts0, self.Ts0-FsMax))

        if IP is None: depsH = 0.
        else: depsH = -IP - eps0[self.kh]

        self.Ts_Ref = Fs
        self.UpdateFRef(F0Opt + depsH * self.S_ao)
        self.vA_Ref = vAOpt*1.

        self.D_In = D_Ref*1.
        self.F_HF_Ref = self.H + self.VH_Ref + self.Vx_Ref

        return F0

    def UpdateFRef(self, F):
        self.F_Ref = F * 1.
        self.epsilon_Ref, self.C_Ref = self.SymHelp.SolveFock(self.F_Ref, self.S_ao)
        self.D_Ref = np.einsum('k,pk,qk->pq', self.f,
                               self.C_Ref[:,:len(self.f)], self.C_Ref[:,:len(self.f)])

        self.Ts_Direct = np.tensordot(self.D_Ref, self.T_ao)

        # Calculate the Fock potential from the reference density matrix
        self.VH_Ref = self.GetVH(D=self.D_Ref)

        # Calculate Vx
        self.Vx_Ref = 0.
        for k in range(self.NOcc):
            ta = np.tensordot(self.ERIA, self.C_Ref[:,k], axes=((2,),(0,)))
            vv = np.tensordot(ta, ta, axes=((0,), (0,)))
            self.Vx_Ref -= self.f[k]/2. * vv


        self.EvH_Ref = np.tensordot(self.D_Ref, (self.VH_Ref))
        self.Evx_Ref = np.tensordot(self.D_Ref, (self.Vx_Ref))
        self.Evxc_Ref = np.tensordot(self.D_Ref, (self.F_Ref - self.H_ao - self.VH_Ref))

        if self.Report>0:
            print("Ts = %10.3f , <vH> = %10.3f <vx> = %10.3f, <vxc> = %10.3f"\
                  %(self.Ts_Ref, self.EvH_Ref, self.Evx_Ref, self.Evxc_Ref))
        
        if self.Report>0:
            print("f   = %s"%(NiceArr(self.f[max(0,self.kh-2):min(self.NBas,self.kl+3)])))
            print("eps = %s eV"%(NiceArr(self.epsilon_Ref[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))
            print("eps_h = %8.3f/%8.2f, eps_l = %8.3f/%8.2f [Ha/eV]"\
                  %(self.epsilon_Ref[self.kh], self.epsilon_Ref[self.kh]*eV,
                    self.epsilon_Ref[self.kl], self.epsilon_Ref[self.kl]*eV,))


    def GetTs(self, NWarm=100, NLast=None):
        Fs = self.Fs_Iter[self.Fs_Iter>0.]
        T  = np.arange(len(Fs))
        
        if NLast is None:
            Fs = Fs[-NLast:]
            T  = T[-NLas:]
        else:
            Fs = Fs[NWarm:]
            T  = T[NWarm:]

        x = np.exp(-T/166)
        p = np.polyfit(x, Fs, 1)
        
        Ts = np.polyval(p, 0)
        return Ts
            
            
    def MatrixReport(self, F = None, ID="F"):
        if F is None: F = self.F
        N1 = self.NOcc + 4
        N0 = max(0, N1-8)
        C_Ref = self.C_Ref[:,N0:N1]
        print("%s - F_HF_Ref = "%(ID))
        print(NiceMat((C_Ref.T).dot(F-self.F_HF_Ref).dot(C_Ref)))
        print("F - F_HF_Ref = ")
        print(NiceMat((C_Ref.T).dot(self.F-self.F_HF_Ref).dot(C_Ref)))
        print("F_Ref - F_HF_Ref = ")
        print(NiceMat((C_Ref.T).dot(self.F_Ref-self.F_HF_Ref).dot(C_Ref)))
    

class PotentialHelper:
    def __init__(self, XHelp, xyz = None, w = None,
                 eta = 1e-5, NExtra = 0,
    ):
        self.XHelp = XHelp
        self.NBas = self.XHelp.NBas
        self.NDFBas = self.XHelp.ERIA.shape[0]
        self.NOcc = self.XHelp.NOcc
        self.NUsed = min(self.NOcc + NExtra, self.NBas)
        self.eta = eta
        
        if (xyz is None) or (w is None):
            xyz, w, phiD = GetDensities(None, wfn=XHelp.wfn, return_xyz=True)
        else:
            phiD = GetDensities(xyz, wfn=XHelp.wfn)

        self.xyz = xyz
        self.w = w
        self.phiD = phiD

    def UpdateGrid(self, xyz, w):
        self.phiD = GetDensities(xyz, wfn=self.XHelp.wfn)
        self.xyz = xyz
        self.w = w

    def UpdateMatrices(self, C=None, V=None, Extend=False):
        if C is None:
            if Extend: C= self.XHelp.C_Ref
            else: C = self.XHelp.C_Ref[:,:self.NUsed]
        
        self.phiC = np.dot(self.phiD, C)
        Vsij = (C.T).dot(V).dot(C)

        self.IJ = []
        for i, j in itertools.product(range(self.NUsed), range(self.NUsed)):
            if j==i: self.IJ += [(i,i,1.)]
            if j>i: self.IJ += [(i,j,2.)]
        self.NIJ = len(self.IJ)


        if Extend:
            self.IJX = [x for x in self.IJ]
            for i, j in itertools.product(range(self.NUsed), range(self.NUsed, C.shape[1])):
                self.IJX += [(i,j,2.)]
            
            self.NIJX = len(self.IJX)
        else:
            self.IJX = self.IJ
            self.NIJX = self.NIJ


        #print("NIJ = %4d, NIJX = %4d"%(self.NIJ, self.NIJX))
        
        # Calculate the overlap in density fit
        self.Aij = np.zeros((self.NDFBas, self.NIJX))
        self.Vij = np.zeros((self.NIJX,))
        self.Sij = np.zeros((self.NIJX,))
        for K, (i,j,Pre) in enumerate(self.IJX):
            self.Aij[:,K] = Pre * self.XHelp.GetVHA(C1=C[:,i], C2=C[:,j])
            self.Vij[K] = Pre / 2. * (Vsij[i,j] + Vsij[j,i])
            if i==j: self.Sij[K] = 1.

        if Extend:
            #S1 = (self.Aij[:,:self.NIJ].T).dot(self.Aij)
            #self.SP = S1.dot(S1.T)
            #self.SP += self.eta*np.diag(np.diag(self.SP)) # Regularize
            #self.SPI = la.inv(self.SP).dot(S1)
            self.SP = (self.Aij.T).dot(self.Aij)
            #self.SP += self.eta*np.diag(np.diag(self.SP)) # Regularize
            self.SP += self.eta*np.eye(self.SP.shape[0])

            self.SPI = la.inv(self.SP)
        else:
            self.SP = (self.Aij.T).dot(self.Aij)
            #self.SP += self.eta*np.diag(np.diag(self.SP)) # Regularize
            self.SP += self.eta*np.eye(self.SP.shape[0])

            self.SPI = la.inv(self.SP)

        self.CP  = np.dot(self.SPI, self.Vij)
        self.CSP = np.dot(self.SPI, self.Sij)

        
    def Reconstruct(self,
                    F = None,
                    Mode = "xc",
                    Extend = True,
                    Normalize = False,
                    ForceHole = None, # Give this a value to override the default
                    Debug = False,
                    ForceUpdate = False,
    ):
        XHelp = self.XHelp

        epsH_Ref = 0. # XHelp.epsilon_Ref[XHelp.kh]
        epsH = 0. # epsH_Ref
        
        if F is None: F = XHelp.F_Ref*1.
        
        H0 = self.XHelp.H
        V0 = 0.
        Veff = F - H0

        N_Ref = np.vdot(self.XHelp.S_ao, self.XHelp.D_Ref)
        Hole = N_Ref - 1.
        
        def QReport(V):
            CC = self.XHelp.C_Ref[:,max(self.NOcc-5,0):(self.NOcc+1)]
            print(NiceMat( (CC.T).dot(V).dot(CC) ))
        
        if Mode.upper()=="C":  # Correlation only
            V0 = XHelp.VH_Ref + XHelp.Vx_Ref
            Veff = Veff - V0
            Hole = 0.
        elif Mode.upper()=="X":  # Correlation only
            F = H0 + XHelp.VH_Ref + XHelp.Vx_Ref
            Veff = XHelp.Vx_Ref
            V0   = XHelp.VH_Ref
            Hole = -1. + np.ceil(N_Ref-1e-4) - N_Ref
        elif Mode.upper()=="XC":  # Full xc
            Veff -= XHelp.VH_Ref
            V0   += XHelp.VH_Ref
            Hole  = -1. + np.ceil(N_Ref-1e-4) - N_Ref

        # Override the default hole if requested
        if not(ForceHole is None): Hole = ForceHole

        QReport(Veff)

        C = XHelp.C_Ref[:,:self.NUsed]
        phiC = np.dot(self.phiD, C)
        rho = np.einsum('k,xk->x', XHelp.f, phiC[:,:len(XHelp.f)]**2)
        
        self.UpdateMatrices(V=Veff, Extend=Extend) # Eventually we can make this quicker by caching

        rhos = 0.
        rhoc = 0.
        for K, (i,j,Pre) in enumerate(self.IJ):
            rhos += Pre * self.CP[K] * phiC[:,i] * phiC[:,j]
            rhoc += Pre * self.CSP[K] * phiC[:,i] * phiC[:,j]

        if Normalize:
            Nxc = np.dot(self.w, rhos)
            Nconst = np.dot(self.w, rhoc)

            print("N = %.3f, Nxc = %.3f, Nconst = %.3f, Hole = %.3f"%(N_Ref, Nxc, Nconst, Hole))

            epsH_Ref = XHelp.epsilon_Ref[XHelp.kh]
            F = (Hole-Nxc)/Nconst
            rhos += F * rhoc
            self.CP += F*self.CSP
            print("eps_H = %.4f -> eps_H = %.4f [shift = %.3f]"%(epsH_Ref, epsH_Ref + F, F))

            
        if Debug:
            NUsed = self.NUsed
            NUsedLUMO = max(self.NUsed, self.NOcc+1)
            N0 = max(NUsed-8, 0)

            CX = XHelp.C_Ref[:,:NUsedLUMO]
            
            def QArr(e, e2=None):
                if not(e2 is None):
                    m = min(len(e), len(e2))
                    return QArr(e[:m]-e2[:m])
                    
                m = len(e)
                return NiceArr(e[N0:min(NUsedLUMO,m)] - e[self.NOcc-1])

            rho_Ref = np.einsum('xp,xq,pq->x', self.phiD, self.phiD, self.XHelp.D_In)
        
            F  = (CX.T).dot(H0 + V0 + Veff).dot(CX)
            eps  = np.diag(F)

            VeffD = XHelp.GetVH(q=self.Aij[:,:len(self.CP)].dot(self.CP))
            FD = (CX.T).dot(H0 + V0 + VeffD).dot(CX)
            epsD = np.diag(FD)

            epsDp, _ = self.XHelp.SymHelp.SolveFock(H0+V0 + VeffD, XHelp.S_ao)
            
            print("error in density = %.7f"%( np.dot(self.w, np.abs(rho-rho_Ref)) ))
            print("eps0 : %s"%(QArr(XHelp.epsilon )))
            print("eps  : %s"%(QArr(XHelp.epsilon_Ref )))
            print("eps  : %s"%(QArr(eps )))
            print("epsD : %s"%(QArr(epsD )))
            print("Error: %s"%(QArr(eps, epsD )))
            print("epsD': %s"%(QArr(epsDp )))
            print("Error: %s"%(QArr(eps, epsDp )))
            print("Weights: %.3f"%( np.dot(self.CP, self.Sij[:len(self.CP)]) ))
            print("<const> = %.4f"%(np.dot(self.w, rhoc)))

            

        if ForceUpdate:
            # Compute Vs and F from the density-fitted potential
            VeffD = XHelp.GetVH(q=self.Aij[:,:len(self.CP)].dot(self.CP))
            FD = H0 + V0 + VeffD

            # Project onto the orbitals
            CAll = XHelp.C_Ref
            FD_C = np.einsum('pq,pj,qk->jk', FD, CAll, CAll)

            # Remove cross-contamination between spaces
            FD_C[:self.NUsed, self.NUsed:] = 0.
            FD_C[self.NUsed:, :self.NUsed] = 0.

            # Return to normal space
            SC = XHelp.S_ao.dot(CAll)
            FD = np.einsum('jk,pj,qk->pq', FD_C, SC, SC)

            XHelp.UpdateFRef(FD)
        
        self.rho = rho
        self.rhos = rhos
        self.rhoc = rhoc

        return rho, rhos
                               

if __name__ == "__main__":
    psi4.set_output_file("__perturb.dat")
    psi4.set_options({
        'basis' : 'def2-tzvp',
        'reference': 'rhf',
    })
    psi4.geometry("""
0 1
Be
symmetry c1""")

    _, wfn_HF = psi4.energy("scf", return_wfn=True)
    E, wfn = psi4.energy("pbe", return_wfn=True)

    D_Ref = wfn_HF.Da().to_array(dense=True)
    
    XHelp = InversionHelper(wfn)
    F_Ref = XHelp.Invert(D_Ref)
