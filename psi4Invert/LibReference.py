import psi4
import numpy as np

import os

class ReferenceHelper:
    def __init__(self, ID, Level = "ccsd", CacheDir = "./Cache/", Sym = False):
        ##### Read or create the cache data
        self.CacheFile = CacheDir + "Cache_Reference.npz"
        
        try:
            X = np.load(self.CacheFile, allow_pickle=True)
            self.Data = X['Data'][()]
            self.IP = X['IP'][()]
        except:
            self.Data = {}
            self.IP = {}

        self.Update(ID, Level, Sym)

    def Update(self, ID, Level = "ccsd", Sym = False):
        self.ID = os.path.basename(ID)        
        self.Level = Level.lower()
        self.Sym = Sym # Not actually implemented yet

        if not(self.ID) in self.Data: self.Data[self.ID] = {}
        if not(self.Level) in self.Data[self.ID]: self.Data[self.ID][self.Level] = {}

        self.SaveCache()

    def Report(self):
        print("Molecules in cache : " + ", ".join(list(self.Data)))
        for M in list(self.Data):
            for L in list(self.Data[M]):
                for B in list(self.Data[M][L]):
                    if not(self.Data[M][L][B]=={}):
                        print("- %s->%s->%s in cache"%(M, L, B))

    def SaveCache(self):
        np.savez(self.CacheFile, Data=self.Data, IP=self.IP)

    def CalculateReference(self, XHelp, Force = False, D_Size = None):
        Basis = XHelp.basis.name().lower()
        if self.Sym: Basis += "_sym"
        
        if (Force is True) or not(Basis in self.Data[self.ID][self.Level]):
            print("Computing %s density"%(self.Level.upper()))
            try:
                _, wfn_Ref = psi4.gradient(self.Level, return_wfn=True)
                #_, wfn_Ref = psi4.properties(self.Level, return_wfn=True)
            except:
                print("Failed to produce CCSD density")
                quit()
            E_Ref = 0.
            #E_Ref = psi4.energy(self.Level)

            D_Ref = XHelp.SymHelp.Dense(wfn_Ref.Da().to_array()) \
                +   XHelp.SymHelp.Dense(wfn_Ref.Db().to_array())

            #self.FixCore(DRef, XHelp)

            self.Data[self.ID][self.Level][Basis] = { 'E_Ref':E_Ref, 'D_Ref':D_Ref }

            self.SaveCache()
        else:
            E_Ref = self.Data[self.ID][self.Level][Basis]['E_Ref']
            D_Ref = self.Data[self.ID][self.Level][Basis]['D_Ref']

            if not(D_Size is None) and not(D_Size.shape == D_Ref.shape):
                # Force calculations if reference size is not appropriate
                self.CalculateReference(XHelp, Force=True)

            print("Reading %s density from cache"%(self.Level.upper()))

        return E_Ref, D_Ref

    
if __name__ == "__main__":
    RH = ReferenceHelper("")
    RH.Report()
