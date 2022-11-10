import itertools
import numpy as np

from scipy.special import erf

def GetPlane(xyz_N, P0=(0,1), P1=(0,2)):
    v1 = xyz_N[P0[1],:] - xyz_N[P0[0],:]
    if not(P1 is None):
        v2 = xyz_N[P1[1],:] - xyz_N[P1[0],:]
    else:
        v2 = v1.dot([[1,0,-1],[1,-2,1],[1,1,1]])

    e1 = v1/np.sqrt((v1**2).sum())
    v2 = v2 - np.dot(v2, e1)*e1
    e2 = v2/np.sqrt((v2**2).sum())

    e3 = np.cross(e1, e2)

    U = np.vstack((e1, e2, e3)).T
    
    t = xyz_N.dot(U)
    Range = (t[:,0].min(), t[:,0].max(),
             t[:,1].min(), t[:,1].max(),)

    print(e1)
    print(e2)
    print(e3)
    print("X in [%7.3f %7.3f], Y in [%7.3f %7.3f]"\
          %Range)

    return U, Range 

class Projector:
    def __init__(self, xyz, w, xyz_Nuc,
                 U = np.eye(3), P0=(0,1),
                 sigma = 0.1,
                 z0 = None):

        self.K0 = P0[0]
        self.K1 = P0[1]
        
        self.xyz_0 = 0. * xyz_Nuc[self.K0,:]
        self.xyz = (xyz-self.xyz_0).dot(U)*1.
        self.w = w*1.
        self.xyz_Nuc = (xyz_Nuc-self.xyz_0).dot(U)
        self.sigma = sigma

        if not(z0 is None):
            self.Prescreen(sigma, z0)

    def GetLine(self, h=0.02, Pad=3.0):
        X0, Y0 = self.xyz_Nuc[self.K0,0], self.xyz_Nuc[self.K0,1]
        X1, Y1 = self.xyz_Nuc[self.K1,0], self.xyz_Nuc[self.K1,1]

        N = int(np.ceil( np.sqrt((X1-X0)**2 + (Y1-Y0)**2) / np.abs(h) ))
        print(N)
        
        sx = np.sign(X1-X0)
        sy = np.sign(Y1-Y0)
        xp = np.linspace(X0 - sx*Pad, X1 + sx*Pad, N)
        yp = np.linspace(Y0 - sy*Pad, Y1 + sy*Pad, N)
        return xp, yp

    def GetPlane(self, Range, h = 0.1, Pad = 1.0, All=True):
        DX = (Range[1]-Range[0])+2.*Pad
        DY = (Range[3]-Range[2])+2.*Pad

        if h>0.:    
            NX = int(np.ceil(DX/h))
            NY = int(np.ceil(DY/h))
        elif h==0.:
            NX = 1
            NY = 1
            h = DY
        else:
            NY = int(np.ceil(10/-h))
            h = DY/NY
            NX = int(np.ceil(DX/h))

        x = (Range[1]+Range[0])/2. + self.xyz_0[0] \
            + h * (np.arange(NX) - (NX-1)/2.)
        y = (Range[3]+Range[2])/2. + self.xyz_0[1] \
            + h * (np.arange(NY) - (NY-1)/2.)

        print("%2d x %2d grid - [ %.2f %.2f %.2f %.2f ]"\
              %(NX, NY, Range[0], Range[1], Range[2], Range[3]))

        if not(All):
            x = h * (np.arange(np.ceil(NX/2)) + 0.5)
            y = h * (np.arange(np.ceil(NY/2)) + 0.5)

        return x,y, [x.min()-h/2, x.max()+h/2, y.max()+h/2, y.min()-h/2, ]


    def Prescreen(self, sigma = None, z0 = 0.):
        if not(sigma is None): self.sigma = sigma

        def d(z):
            return (z-z0)**2<32*self.sigma**2

        self.kPre = d(self.xyz[:,2])
        self.xyz_Nuc_S = self.xyz_Nuc[d(self.xyz_Nuc[:,2]),:]
        

    def Project(self, x, y, z, rhoList):
        # Identify if we have a single of multi value
        # projection here
        
        xyz = self.xyz[self.kPre,:].reshape((-1,3))
        w = self.w[self.kPre]

        rhoList = [x[self.kPre] for x in rhoList]

        R2 = (xyz[:,0]-x)**2 + (xyz[:,1]-y)**2 + (xyz[:,2]-z)**2
        kk = R2<(32*self.sigma**2)
        
        G = np.exp(-0.5*R2[kk]/self.sigma**2)
        G /= np.dot(w[kk], G)
        
        return tuple([ np.dot(w[kk]*rho_x[kk], G) for rho_x in rhoList])
        
    def Potential(self, x, y, z, rhoList, sigma=0.03):
        R = np.sqrt((self.xyz[:,0] - x)**2
                    + (self.xyz[:,1] - y)**2
                    + (self.xyz[:,2] - z)**2)
        wV = erf(R/sigma)/R * self.w

        return tuple([np.dot(wV, rho) for rho in rhoList])
