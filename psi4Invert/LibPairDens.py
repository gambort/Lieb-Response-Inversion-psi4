import numpy as np
import psi4

def GetDensity_xyz(wfn, D):
    xyz, w, (rho,), _ = GetDensities(None, w=w, D1List=(D,), wfn=wfn,
                                     return_w=True, return_xyz=True)
    return rho, xyz, w


def GetDensity(wfn, D, xyz=None, w=None):
    w, (rho,), _ = GetDensities(xyz, w=w, D1List=(D,), wfn=wfn, return_w=True)
    return rho, w

def GetDensityGradient(wfn, D, xyz=None, w=None):
    w, (drho,), _ = GetDensities(xyz, w=w, D1List=(D,), wfn=wfn, return_w=True, Gradient=True)
    return drho[:,1], w

def GetGGAProps(wfn, D, xyz=None, w=None):
    w, (rho,), _ = GetDensities(xyz, w=w, D1List=(D,), wfn=wfn, return_w=True, Gradient=True)
    rs = 0.62035 * rho[:,0]**(-1/3)
    s  = 0.16162 * (rho[:,1]/rho[:,0])/rho[:,0]**(1/3)
    return rs, s, w

def GetMGGAProps(wfn, D, xyz=None, w=None):
    w, (rho,), _ = GetDensities(xyz, w=w, D1List=(D,), wfn=wfn, return_w=True,
                                Gradient=True, Kinetic=True)
    rs = 0.62035 * rho[:,0]**(-1/3)
    s  = 0.16162 * (rho[:,1]/rho[:,0])/rho[:,0]**(1/3)
    t = rho[:,2]/rho[:,0]
    return rs, s, t, w

def GetDensities(xyz, w=None, D1List = [], D2List = [],
                 DMap=None,
                 Gradient=False, Kinetic=False,
                 Loud=False,
                 delta=1e-8,
                 wfn=None,
                 basis=None,
                 return_w=False,
                 return_xyz=False,
):
    if basis is None:
        try:
            basis = wfn.basisset()
        except:
            print("No basis set or wfn specified")
            quit()

    if xyz is None:
        # Use the default grid
        
        Vpot = wfn.V_potential()
        x_A, y_A, z_A = [], [], []
        w_A = []

        NTot = 0
        for b in range(Vpot.nblocks()):
            block = Vpot.get_block(b)
            NTot += block.npoints()
            x_A += [block.x()]
            y_A += [block.y()]
            z_A += [block.z()]
            w_A += [block.w()]

        xyz = np.zeros((NTot,3))
        w = np.zeros((NTot,))
        k0 = 0
        for k in range(len(w_A)):
            N = w_A[k].shape[0]
            xyz[k0:(k0+N),0] = x_A[k]
            xyz[k0:(k0+N),1] = y_A[k]
            xyz[k0:(k0+N),2] = z_A[k]
            w[k0:(k0+N)] = w_A[k]
            k0 += N

        return_w = True # Force return of weights
        # Done
    
    X = psi4.core.Vector.from_array(xyz[:,0])
    Y = psi4.core.Vector.from_array(xyz[:,1])
    Z = psi4.core.Vector.from_array(xyz[:,2])
    if w is None:
        W = psi4.core.Vector.from_array(0.*xyz[:,0])
    else:
        W = psi4.core.Vector.from_array(w)
        
    blockopoints = psi4.core.BlockOPoints(X,Y,Z,W,
                                          psi4.core.BasisExtents(basis,delta))

    npoints = blockopoints.npoints()

    lpos = np.array(blockopoints.functions_local_to_global())

    funcs = psi4.core.BasisFunctions(basis,npoints,basis.nbf())
    funcs.compute_functions(blockopoints)
    lphi = funcs.basis_values()["PHI"].to_array(dense=True)
    if not(DMap is None):
        phiD = np.dot(lphi, DMap[lpos,:])
    else:
        E = np.eye(wfn.nmo())
        phiD = np.dot(lphi, E[lpos,:])
        
    if Gradient or Kinetic:
        funcs.set_deriv(1)
        funcs.deriv()
        funcs.compute_functions(blockopoints)
        lphi_x = funcs.basis_values()["PHI_X"].to_array(dense=True)
        lphi_y = funcs.basis_values()["PHI_Y"].to_array(dense=True)
        lphi_z = funcs.basis_values()["PHI_Z"].to_array(dense=True)
        if not(DMap is None):
            phiD_x = np.dot(lphi_x, DMap[lpos,:])
            phiD_y = np.dot(lphi_y, DMap[lpos,:])
            phiD_z = np.dot(lphi_z, DMap[lpos,:])
        else:
            E = np.eye(wfn.nmo())
            phiD_x = np.dot(lphi_x, E[lpos,:])
            phiD_y = np.dot(lphi_y, E[lpos,:])
            phiD_z = np.dot(lphi_z, E[lpos,:])

    if Kinetic:
        funcs.set_deriv(2)
        funcs.deriv()
        funcs.compute_functions(blockopoints)
        lphi_xx = funcs.basis_values()["PHI_XX"].to_array(dense=True)
        lphi_yy = funcs.basis_values()["PHI_YY"].to_array(dense=True)
        lphi_zz = funcs.basis_values()["PHI_ZZ"].to_array(dense=True)
        if not(DMap is None):
            phiD_xx = np.dot(lphi_xx, DMap[lpos,:])
            phiD_yy = np.dot(lphi_yy, DMap[lpos,:])
            phiD_zz = np.dot(lphi_zz, DMap[lpos,:])


    if len(D1List)==0:
        if Kinetic:
            return phiD, phiD_x, phiD_y, phiD_z, phiD_xx, phiD_yy, phiD_zz
        elif Gradient:
            return phiD, phiD_x, phiD_y, phiD_z
        elif return_xyz:
            return xyz, w, phiD
        else:
            return phiD

    rho1List = [None]*len(D1List)
    for k,D in enumerate(D1List):
        if D is None: continue
        if Loud:
            print("%03d of %03d D1"%(k,len(D1List)))

        if DMap is None:
            lD = D[(lpos[:, None], lpos)]
            if Gradient or Kinetic:
                if Kinetic:
                    T = np.zeros((lphi.shape[0],4))
                else:
                    T = np.zeros((lphi.shape[0],2))

                T[:,0] = np.einsum("ip,iq,pq->i", lphi, lphi, lD,
                                   optimize=True)
                Q_x = (np.einsum("ip,iq,pq->i", lphi_x, lphi, lD, optimize=True)
                       + np.einsum("ip,iq,pq->i", lphi, lphi_x, lD, optimize=True))
                Q_y = (np.einsum("ip,iq,pq->i", lphi_y, lphi, lD, optimize=True)
                       + np.einsum("ip,iq,pq->i", lphi, lphi_y, lD, optimize=True))
                Q_z = (np.einsum("ip,iq,pq->i", lphi_z, lphi, lD, optimize=True)
                       + np.einsum("ip,iq,pq->i", lphi, lphi_z, lD, optimize=True))

                T[:,1]  = np.sqrt(Q_x**2 + Q_y**2 + Q_z**2)

                if Kinetic:
                    T[:,2] = 0.5 * ( np.einsum("ip,iq,pq->i", lphi_x, lphi_x, lD, optimize=True)
                                     + np.einsum("ip,iq,pq->i", lphi_y, lphi_y, lD, optimize=True)
                                     + np.einsum("ip,iq,pq->i", lphi_z, lphi_z, lD, optimize=True) )
                rho1List[k] = T*1.
            else:
                rho1List[k] = np.einsum("ip,iq,pq->i", lphi, lphi, lD,
                                        optimize=True)
        else:
            rho1List[k] = np.einsum("ip,iq,pq->i", phiD, phiD, D,
                                    optimize=True)
            
    rho2List = [None]*len(D2List)        
    for k,D2 in enumerate(D2List):
        if D2 is None: continue
        if Loud:
            print("%03d of %03d D1"%(k,len(D2List)))

        if DMap is None:
            lD2 = D2[(lpos[:, None, None, None], lpos[:, None, None], lpos[:, None], lpos)]
            rho2List[k] = np.einsum("p,q,jr,js,pqrs->j",
                                    lphi[0,:], lphi[0,:], lphi, lphi, lD2,
                                    optimize=True)
        else:
            rho2List[k] = np.einsum("p,q,jr,js,pqrs->j",
                                    phiD[0,:], phiD[0,:], phiD, phiD, D2,
                                    optimize=True)

    if not(return_w):
        return tuple(rho1List), tuple(rho2List)
    else:
        if return_xyz:
            return xyz, w, tuple(rho1List), tuple(rho2List)
        else:
            return w, tuple(rho1List), tuple(rho2List)
