"""
2D SIMP topology optimization, half MBB beam.
Based on the 88 line MATLAB code by Andreassen et al. (2011), rewritten in
python with the parameters exposed and plots added.

run it with:  python simp_solver.py
change the parameters at the bottom and run again.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
import time


def simp_2d(nelx, nely, volfrac, penal, rmin, ft=1, max_iter=200):
    E0 = 1.0
    Emin = 1e-9       # not zero, keeps the stiffness matrix non singular
    nu = 0.3

    # 8x8 element stiffness for a bilinear quad, plane stress
    # every element reuses this, scaled by the penalized density
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

    # map each element to its 8 dofs
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        (1 + nely, 1 + nelx), order='F')
    edofVec = (2 * nodenrs[0:-1, 0:-1] + 1).reshape(-1, order='F')
    edofMat = np.tile(edofVec, (8, 1)).T + np.tile(
        np.array([0, 1, 2 * nely + 2, 2 * nely + 3,
                  2 * nely, 2 * nely + 1, -2, -1]),
        (nelx * nely, 1))

    iK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()

    # half MBB beam: unit load top left, x fixed on the left edge (symmetry),
    # y fixed at the bottom right corner (roller)
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

    # filter weights: for each element collect neighbours within rmin,
    # weight = rmin - distance. this is what stops the checkerboarding
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
                    dist = max(0, rmin - np.sqrt((i1 - i2)**2 + (j1 - j2)**2))
                    iH.append(e1)
                    jH.append(e2)
                    sH.append(dist)

    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()

    x = np.full(nely * nelx, volfrac)
    xPhys = x.copy()
    xold = x.copy()

    history = {'compliance': [], 'volume': [], 'change': []}
    loop = 0
    change = 1.0

    while change > 0.01 and loop < max_iter:
        loop += 1

        # assemble K and solve K U = F on the free dofs
        # stiffness = Emin + xPhys^p * (E0 - Emin), the SIMP interpolation
        sK = ((KE.flatten()[np.newaxis]).T *
              (Emin + xPhys**penal * (E0 - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK - 1, jK - 1)), shape=(ndof, ndof)).tocsc()
        U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])

        # compliance and its sensitivity
        ce = np.zeros(nely * nelx)
        for i in range(nely * nelx):
            Ue = U[edofMat[i, :] - 1, 0]
            ce[i] = Ue @ KE @ Ue

        c = np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
        dc = -penal * xPhys**(penal - 1) * (E0 - Emin) * ce
        dv = np.ones(nely * nelx)

        if ft == 0:
            # sensitivity filter
            dc = np.array(H @ (x * dc) / Hs / np.maximum(x, 1e-3))
        elif ft == 1:
            # density filter
            dc = np.array(H @ (dc / Hs))
            dv = np.array(H @ (dv / Hs))

        # OC update, bisection on the lagrange multiplier for the volume
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
        history['compliance'].append(c)
        history['volume'].append(vol)
        history['change'].append(change)

        if loop % 10 == 0 or loop == 1:
            print(f"  iter {loop:4d}   c {c:10.4f}   vol {vol:.4f}   change {change:.4f}")

    print(f"\n  stopped at iter {loop}, compliance {history['compliance'][-1]:.4f}")
    return xPhys.reshape((nely, nelx), order='F'), history


def plot_results(xPhys, history, params, save_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"SIMP TO | mesh={params['nelx']}x{params['nely']} | "
        f"volfrac={params['volfrac']} | p={params['penal']} | "
        f"rmin={params['rmin']} | filter={'density' if params['ft'] == 1 else 'sensitivity'}",
        fontsize=12, fontweight='bold'
    )

    ax = axes[0, 0]
    im = ax.imshow(1 - xPhys, cmap='gray', interpolation='none',
                   vmin=0, vmax=1, aspect='equal')
    ax.set_title('optimized topology')
    ax.set_xlabel('x elements')
    ax.set_ylabel('y elements')
    plt.colorbar(im, ax=ax, label='density (black solid, white void)')

    ax = axes[0, 1]
    ax.plot(history['compliance'], 'b-', linewidth=1.5)
    ax.set_title('compliance')
    ax.set_xlabel('iteration')
    ax.set_ylabel('compliance')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(history['compliance']))

    ax = axes[1, 0]
    ax.plot(history['volume'], 'r-', linewidth=1.5)
    ax.axhline(y=params['volfrac'], color='k', linestyle='--',
               alpha=0.5, label=f"target = {params['volfrac']}")
    ax.set_title('volume fraction')
    ax.set_xlabel('iteration')
    ax.set_ylabel('volume fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(history['volume']))

    ax = axes[1, 1]
    ax.semilogy(history['change'], 'g-', linewidth=1.5)
    ax.axhline(y=0.01, color='k', linestyle='--', alpha=0.5, label='threshold')
    ax.set_title('design change')
    ax.set_xlabel('iteration')
    ax.set_ylabel('max density change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(history['change']))

    plt.tight_layout()

    if save_dir:
        fname = (f"to_result_p{params['penal']}_r{params['rmin']}_"
                 f"v{params['volfrac']}_ft{params['ft']}.png")
        fpath = os.path.join(save_dir, fname)
    else:
        fpath = 'to_result.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  plot saved to {fpath}")
    return fpath


if __name__ == "__main__":
    params = {
        'nelx': 150,
        'nely': 50,
        'volfrac': 0.5,
        'penal': 3.0,
        'rmin': 1.5,
        'ft': 1,
    }

    save_dir = os.path.dirname(os.path.abspath(__file__))

    start = time.time()
    xPhys, history = simp_2d(**params)
    print(f"  solver time {time.time() - start:.1f}s")

    plot_results(xPhys, history, params, save_dir=save_dir)
