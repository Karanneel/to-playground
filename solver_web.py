"""
SIMP 2D solver, browser version for Pyodide.
Same algorithm as simp_solver.py, stripped of matplotlib and print output.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def simp_2d(nelx, nely, volfrac, penal, rmin, ft=1, max_iter=200):
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3

    # element stiffness matrix, bilinear quad, plane stress
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0],
                     [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6],
                     [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4],
                     [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2],
                     [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = 1 / (1 - nu**2) / 24 * (
        np.block([[A11, A12], [A12.T, A11]]) +
        nu * np.block([[B11, B12], [B12.T, B11]])
    )

    # DOF numbering
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        (1 + nely, 1 + nelx), order='F')
    edofVec = (2 * nodenrs[0:-1, 0:-1] + 1).reshape(-1, order='F')
    edofMat = np.tile(edofVec, (8, 1)).T + np.tile(
        np.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1]),
        (nelx * nely, 1))

    iK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()

    # half-MBB beam BCs
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = lil_matrix((ndof, 1))
    U = np.zeros((ndof, 1))
    F[1, 0] = -1.0

    fixeddofs = np.union1d(
        np.arange(0, 2 * (nely + 1), 2),
        np.array([2 * (nelx + 1) * (nely + 1) - 1])
    )
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    F = F.tocsc()

    # filter weights
    iH, jH, sH = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            imin = max(i1 - int(np.ceil(rmin) - 1), 0)
            imax = min(i1 + int(np.ceil(rmin)), nelx)
            jmin = max(j1 - int(np.ceil(rmin) - 1), 0)
            jmax = min(j1 + int(np.ceil(rmin)), nely)
            for i2 in range(imin, imax):
                for j2 in range(jmin, jmax):
                    e2 = i2 * nely + j2
                    dist = max(0, rmin - np.sqrt((i1-i2)**2 + (j1-j2)**2))
                    iH.append(e1)
                    jH.append(e2)
                    sH.append(dist)

    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()

    x = np.full(nely * nelx, volfrac)
    xPhys = x.copy()
    xold = x.copy()

    history = {'compliance': [], 'volume': [], 'change': []}
    loop = 0
    change = 1.0

    while change > 0.01 and loop < max_iter:
        loop += 1

        sK = ((KE.flatten()[np.newaxis]).T *
              (Emin + xPhys**penal * (E0 - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK-1, jK-1)), shape=(ndof, ndof)).tocsc()
        U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])

        ce = np.zeros(nely * nelx)
        for i in range(nely * nelx):
            Ue = U[edofMat[i, :] - 1, 0]
            ce[i] = Ue @ KE @ Ue

        c = np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce
        dv = np.ones(nely * nelx)

        if ft == 0:
            dc = np.array(H @ (x * dc) / Hs / np.maximum(x, 1e-3))
        elif ft == 1:
            dc = np.array(H @ (dc / Hs))
            dv = np.array(H @ (dv / Hs))

        # OC update
        l1, l2 = 0, 1e9
        move = 0.2
        xnew = np.copy(x)

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0,
                    np.maximum(x - move,
                    np.minimum(1.0,
                    np.minimum(x + move,
                    x * np.sqrt(-dc / dv / lmid)))))

            if ft == 0:
                xPhys = xnew.copy()
            elif ft == 1:
                xPhys = np.array(H @ (xnew / Hs))

            if xPhys.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid

        change = np.max(np.abs(xnew - xold))
        xold = xnew.copy()
        x = xnew.copy()

        vol = xPhys.sum() / (nelx * nely)
        history['compliance'].append(float(c))
        history['volume'].append(float(vol))
        history['change'].append(float(change))

    return xPhys.reshape((nely, nelx), order='F'), history
