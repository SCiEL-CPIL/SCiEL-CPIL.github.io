"""Packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(centers):
    n = len(centers)
    mb = 0.5 - np.max(np.abs(centers), axis=1)
    mb = float(np.min(mb))
    mp = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d * 0.5 < mp:
                mp = d * 0.5
    return min(mb, mp)


def construct():
    t0 = time.time()
    deadline = t0 + 125.0

    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].astype(float).copy()
    best_r = _radius(centers)
    best_c = centers.copy()

    # Phase 1: Force-based refinement
    lr = 0.003
    for it in range(80):
        if time.time() > deadline - 20.0:
            break
        forces = np.zeros_like(centers)
        for i in range(N):
            for k in range(2):
                margin = 0.5 - abs(centers[i, k])
                if margin < 0.05:
                    forces[i, k] -= np.sign(centers[i, k]) * (0.05 - margin) * 2.0
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([1e-8, 0.0])
                    dist = 1e-8
                if dist < 0.1:
                    f = (0.1 - dist) / dist
                    forces[i] += diff / dist * f
                    forces[j] -= diff / dist * f
        centers += lr * forces
        centers = np.clip(centers, -0.49999, 0.49999)
        r = _radius(centers)
        if r > best_r:
            best_r = r
            best_c = centers.copy()
        lr *= 0.985

    # Phase 2: Scipy polish
    if time.time() < deadline - 10.0:
        def obj(c_flat):
            c = c_flat.reshape(N, 2)
            mb = 0.5 - np.max(np.abs(c), axis=1)
            mb = float(np.min(mb))
            mp = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                    if d * 0.5 < mp:
                        mp = d * 0.5
            return -min(mb, mp)

        x0 = best_c.flatten()
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-9})
        xc = res.x.reshape(N, 2)
        ac = _radius(xc)
        if ac > best_r:
            best_r = ac
            best_c = xc.copy()

    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    return best_c
# EVOLVE-BLOCK-END

