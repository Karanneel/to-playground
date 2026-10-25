"""
Sweeps one solver parameter at a time, holds the rest fixed, and plots how
it changes the final compliance, the grey fraction and the convergence.
writes the plots into images/ for the readme.

run it with:  python param_sweep.py
each sweep takes a few minutes, drop nelx/nely for faster runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simp_solver import simp_2d

BASE = {
    'nelx': 90,
    'nely': 30,
    'volfrac': 0.5,
    'penal': 3.0,
    'rmin': 1.5,
    'ft': 1,
    'max_iter': 150,
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(SAVE_DIR, exist_ok=True)


def sweep_parameter(param_name, values, base=BASE):
    results = []
    for val in values:
        params = base.copy()
        params[param_name] = val
        print(f"\n  sweeping {param_name} = {val}")
        t0 = time.time()
        xPhys, history = simp_2d(**params)
        elapsed = time.time() - t0

        # grey = elements that didn't resolve to solid or void
        grey = np.sum((xPhys > 0.1) & (xPhys < 0.9)) / xPhys.size

        results.append({
            'value': val,
            'compliance': history['compliance'][-1],
            'iterations': len(history['compliance']),
            'grey_fraction': grey,
            'time': elapsed,
            'xPhys': xPhys,
            'history': history,
        })
    return results


def plot_sweep(param_name, results, label, save_dir=SAVE_DIR):
    values = [r['value'] for r in results]
    compliances = [r['compliance'] for r in results]
    greys = [r['grey_fraction'] * 100 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'parameter sweep: {label}', fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(values, compliances, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(label)
    ax.set_ylabel('final compliance')
    ax.set_title('compliance vs parameter')
    ax.grid(True, alpha=0.3)

    # label the curve shape: compare the midpoint to a straight line
    # between the end points
    c = np.array(compliances)
    v = np.array(values)
    if len(c) > 2:
        mid = len(c) // 2
        linear_mid = c[0] + (c[-1] - c[0]) * (v[mid] - v[0]) / (v[-1] - v[0])
        dev = abs(c[mid] - linear_mid) / max(abs(c[-1] - c[0]), 1e-9)
        if dev < 0.05:
            shape = "~linear"
        elif c[mid] < linear_mid:
            shape = "concave (diminishing returns)"
        else:
            shape = "convex (accelerating)"
        ax.annotate(f'shape: {shape}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', color='darkblue',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    ax = axes[0, 1]
    ax.plot(values, greys, 'rs-', linewidth=2, markersize=8)
    ax.set_xlabel(label)
    ax.set_ylabel('grey elements (%)')
    ax.set_title('grey zone vs parameter')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        ax.plot(r['history']['compliance'], color=cmap[i],
                linewidth=1.5, label=f'{param_name}={r["value"]}')
    ax.set_xlabel('iteration')
    ax.set_ylabel('compliance')
    ax.set_title('convergence, all runs')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # small multiples of the final shapes, packed into one image
    ax = axes[1, 1]
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    pad = 2
    h, w = results[0]['xPhys'].shape
    composite = np.ones((rows * (h + pad) - pad, cols * (w + pad) - pad))
    for i, r in enumerate(results):
        y0 = (i // cols) * (h + pad)
        x0 = (i % cols) * (w + pad)
        composite[y0:y0+h, x0:x0+w] = 1 - r['xPhys']

    ax.imshow(composite, cmap='gray', interpolation='none',
              vmin=0, vmax=1, aspect='equal')
    ax.set_title('topologies (left to right, top to bottom)')
    for i, r in enumerate(results):
        y0 = (i // cols) * (h + pad) + h // 2
        x0 = (i % cols) * (w + pad) + w // 2
        ax.text(x0, y0, f'{param_name}={r["value"]}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fpath = os.path.join(save_dir, f"sweep_{param_name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {fpath}")
    return fpath


def plot_penalization_curve():
    # the raw SIMP curve E(rho) = rho^p and the stiffness per unit density
    # rho^(p-1), which is why intermediate density gets penalized at high p
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SIMP penalization curve  E(rho) = rho^p',
                 fontsize=13, fontweight='bold')

    rho = np.linspace(0, 1, 200)

    ax = axes[0]
    for p in [1, 1.5, 2, 3, 5, 8]:
        ax.plot(rho, rho**p, linewidth=2, label=f'p = {p}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='linear (p=1)')
    ax.set_xlabel('density (rho)')
    ax.set_ylabel('relative stiffness E/E0')
    ax.set_title('stiffness interpolation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    rho_nz = np.linspace(0.01, 1, 200)
    for p in [1, 1.5, 2, 3, 5, 8]:
        ax.plot(rho_nz, rho_nz**(p - 1), linewidth=2, label=f'p = {p}')
    ax.set_xlabel('density (rho)')
    ax.set_ylabel('stiffness per unit density')
    ax.set_title('material efficiency, why grey is wasteful at high p')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.annotate('at p=3 a cell at rho=0.5\ngets only 25% efficiency',
                xy=(0.5, 0.5**2), xytext=(0.15, 0.6),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.tight_layout()
    fpath = os.path.join(SAVE_DIR, 'penalization_curves.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {fpath}")
    return fpath


if __name__ == "__main__":
    plot_penalization_curve()

    p_values = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    plot_sweep('penal', sweep_parameter('penal', p_values), 'penalization power (p)')

    r_values = [0.5, 1.0, 1.5, 2.5, 4.0, 6.0]
    plot_sweep('rmin', sweep_parameter('rmin', r_values), 'filter radius (rmin)')

    v_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    plot_sweep('volfrac', sweep_parameter('volfrac', v_values), 'volume fraction')

    print("\n  all sweeps done")
