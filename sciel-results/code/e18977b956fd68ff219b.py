"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin under 150s wall-clock


def _rect_for_positions(pos, r):
    # pos: (N,2), r: (N,). Returns (W, H, valid).
    x, y = pos[:, 0], pos[:, 1]
    W = (x + r).max() - (x - r).min()
    H = (y + r).max() - (y - r).min()
    valid = (W > 0) and (H > 0) and (W + H <= 2.0 + 1e-9)
    return W, H, valid


def _penalty(pos, r):
    # Smooth min-gap surrogate (log-sum-exp) for pair separation.
    n = pos.shape[0]
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    req = r[:, None] + r[None, :]
    deficit = req - d  # >0 means overlap
    m = deficit.max()
    if m > 0:
        return m + np.logaddexp.reduce(np.logaddexp.reduce(deficit, axis=1), axis=0)
    return 0.0


def _objective_and_grad(pos, r, W, H):
    # Objective: penalize perimeter overflow + pair overlap.
    n = pos.shape[0]
    x, y = pos[:, 0], pos[:, 1]
    Lx, Rx = x - r, x + r
    Ly, Ry = y - r, y + r
    Wcur = Rx.max() - Lx.min()
    Hcur = Ry.max() - Ly.min()
    over = max(0.0, (Wcur + Hcur - 2.0))

    # pair overlap (soft)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 1e12)
    req = r[:, None] + r[None, :]
    gap = d - req
    # soft min of gap: smaller = worse
    gmin = gap.min()
    # smooth sum of negative gaps (overlap amount)
    K = 40.0
    logsum = np.logaddexp.reduce(np.logaddexp.reduce(-gap, axis=1) / K, axis=0) * K
    val = 1e4 * over + logsum + 1e-6 * (gmin ** 2 if gmin < 0 else 0.0)

    # gradient w.r.t. positions
    gx = np.zeros_like(pos[:, 0])
    gy = np.zeros_like(pos[:, 1])
    # perimeter gradient (active if over>0)
    if over > 0:
        iLx = np.argmin(Lx); iRx = np.argmax(Rx)
        iLy = np.argmin(Ly); iRy = np.argmax(Ry)
        gx[iLx] -= 1e4; gx[iRx] += 1e4
        gy[iLy] -= 1e4; gy[iRy] += 1e4
    # pair overlap gradient
    invd = 1.0 / np.where(d > 1e-12, d, 1e-12)
    w = np.exp(-gap / K)
    np.fill_diagonal(w, 0.0)
    wsum = w.sum(axis=1)
    diffx = pos[:, None, 0] - pos[None, :, 0]
    diffy = pos[:, None, 1] - pos[None, :, 1]
    gx += (w * diffx * invd).sum(axis=1)
    gy += (w * diffy * invd).sum(axis=1)
    return val, np.column_stack([gx, gy])


def _polish(pos, r, W, H, t_end):
    # Local refinement: scale positions to fill rectangle, then local optimize
    # the positions to maximize min gap subject to perimeter <= 2.
    x, y = pos[:, 0], pos[:, 1]
    Lx, Rx = x - r, x + r
    Ly, Ry = y - r, y + r
    cx0, cx1 = Lx.min(), Rx.max()
    cy0, cy1 = Ly.min(), Ry.max()
    # center in [0,W]x[0,H]
    pos = pos.copy()
    pos[:, 0] += (W - (cx1 - cx0)) / 2.0 - cx0
    pos[:, 1] += (H - (cy1 - cy0)) / 2.0 - cy0

    def f(v):
        p = v.reshape(-1, 2)
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        req = r[:, None] + r[None, :]
        return -(d - req).min()

    res = minimize(f, pos.reshape(-1), method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
                   jac=None)
    if res.success or res.fun < f(pos.reshape(-1)):
        pos = res.x.reshape(-1, 2)
    return pos


def construct():
    t0 = time.time()
    rng = np.random.default_rng(0)

    best_pos = None
    best_r = None
    best_val = -np.inf

    # base grid radius
    side = int(np.ceil(np.sqrt(N)))
    base_r = 0.5 / side - 1e-3

    for restart in range(6):
        if time.time() > DEADLINE:
            break
        # initial positions: jittered grid
        pos = np.array([((i + 0.5) / side, (j + 0.5) / side)
                        for i in range(side) for j in range(side)][:N], dtype=float)
        pos = pos * 1.0
        if restart > 0:
            pos = pos + rng.normal(0, 0.03, size=pos.shape)
        r = np.full(N, base_r)

        # Force-directed iterations: maximize r while keeping perimeter <= 2
        W = H = None
        it = 0
        while time.time() < DEADLINE - 2.0:
            it += 1
            # grow radii uniformly a bit
            r *= 1.004
            # check perimeter
            x, y = pos[:, 0], pos[:, 1]
            Wcur = (x + r).max() - (x - r).min()
            Hcur = (y + r).max() - (y - r).min()
            if Wcur + Hcur > 2.0:
                # scale down radii to fit
                scale = 2.0 / (Wcur + Hcur)
                r *= scale
                break
            # force-directed relaxation: repel overlaps, spring to keep contact
            for _ in range(3):
                d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
                np.fill_diagonal(d, np.inf)
                req = r[:, None] + r[None, :]
                overlap = req - d  # >0 overlap
                force = np.zeros_like(pos)
                for i in range(N):
                    for j in range(i + 1, N):
                        if overlap[i, j] > 0:
                            dirv = (pos[j] - pos[i]) / max(d[i, j], 1e-9)
                            f = overlap[i, j]
                            force[i] -= dirv * f
                            force[j] += dirv * f
                pos += 0.5 * force
            # keep inside bounding box (recenter)
            x, y = pos[:, 0], pos[:, 1]
            cx0 = (x - r).min(); cx1 = (x + r).max()
            cy0 = (y - r).min(); cy1 = (y + r).max()
            pos[:, 0] += (cx0 + cx1) / 2.0 - (cx0 + cx1) / 2.0  # no-op; recenter to 0
            pos = pos - pos.mean(axis=0)

            # periodic polish
            if it % 5 == 0 and time.time() < DEADLINE - 5.0:
                Wcur = (x + r).max() - (x - r).min()
                Hcur = (y + r).max() - (y - r).min()
                if Wcur + Hcur <= 2.0:
                    pos = _polish(pos, r, Wcur, Hcur, DEADLINE)
                    x, y = pos[:, 0], pos[:, 1]

        # final fit: scale r to exactly perimeter 2
        x, y = pos[:, 0], pos[:, 1]
        Wcur = (x + r).max() - (x - r).min()
        Hcur = (y + r).max() - (y - r).min()
        if Wcur + Hcur > 2.0:
            r *= 2.0 / (Wcur + Hcur)
        # recenter to [0,W]x[0,H]
        Wcur = (x + r).max() - (x - r).min()
        Hcur = (y + r).max() - (y - r).min()
        pos[:, 0] += (2.0 - (Wcur + Hcur)) / 2.0 - (x - r).min()
        pos[:, 1] += (2.0 - (Wcur + Hcur)) / 2.0 - (y - r).min()

        val = r.sum()
        if val > best_val:
            best_val = val
            best_pos = pos.copy()
            best_r = r.copy()

    # final polish on best
    x, y = best_pos[:, 0], best_pos[:, 1]
    Wcur = (x + best_r).max() - (x - best_r).min()
    Hcur = (y + best_r).max() - (y - best_r).min()
    if time.time() < DEADLINE - 2.0:
        best_pos = _polish(best_pos, best_r, Wcur, Hcur, DEADLINE)

    # build output
    out = np.column_stack([best_pos, best_r])
    return out
# EVOLVE-BLOCK-END
