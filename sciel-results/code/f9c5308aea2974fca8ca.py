"""Fresh design: SA + fast GD + high-precision polish for log-energy on the sphere."""
import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
SAFETY = 8.0  # seconds margin before the kill
RNG_SEED = 12345


def _energy_grad(pts):
    """Return (E, G) where E = sum_{i<j} log(1/|xi-xj|), G = dE/dpts (on R^3).
    E = - sum_{i<j} log(d_ij).  dE/dx_i = sum_{j != i} (x_i - x_j)/|x_i-x_j|^2."""
    # pairwise differences
    d = pts[:, None, :] - pts[None, :, :]          # (N,N,3)
    dist2 = np.einsum('ijk,ijk->ij', d, d)         # (N,N)
    # avoid diagonal / self
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    # gradient rows: G_i = sum_j (x_i - x_j)/dist2_ij  (j != i)
    # (x_i - x_j)/dist2 = d_ij / dist2_ij
    inv_d2 = 1.0 / dist2
    np.fill_diagonal(inv_d2, 0.0)
    G = np.einsum('ij,ijk->ik', inv_d2, d)         # (N,3)
    # energy: -sum_{i<j} log(dist)  = -0.5 * sum_{i != j} log(dist)
    # use a masked log over upper triangle
    iu = np.triu_indices(N, 1)
    E = -np.sum(np.log(dist[iu]))
    return E, G


def _project(pts):
    """Re-normalize each row to the unit sphere (chordal constraint)."""
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return pts / n


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gd_step(pts, step, iters, deadline, best_E):
    """Plain gradient descent on the sphere with adaptive step + decay.
    Returns (pts, E, best_pts, best_E)."""
    E, G = _energy_grad(pts)
    best_pts = pts.copy()
    best_E = E
    lr = step
    for _ in range(iters):
        if time.time() > deadline:
            break
        # project gradient to tangent (remove radial component)
        Gt = G - np.sum(G * pts, axis=1, keepdims=True) * pts
        pts = _project(pts - lr * Gt)
        E, G = _energy_grad(pts)
        if E < best_E:
            best_E = E
            best_pts = pts.copy()
        # adaptive: shrink if energy rose, else keep; decay
        lr *= 0.999
    return pts, E, best_pts, best_E


def _sa(pts, start_temp, end_temp, iters, deadline, best_pts, best_E, rng):
    """Simulated annealing: perturb + short GD re-optimization.
    Returns (best_pts, best_E)."""
    cur = pts.copy()
    cur_E, _ = _energy_grad(cur)
    T = start_temp
    decay = (end_temp / start_temp) ** (1.0 / max(iters, 1))
    for k in range(iters):
        if time.time() > deadline:
            break
        # random tangent perturbation
        noise = rng.standard_normal((N, 3)) * math.sqrt(T)
        noise = noise - np.sum(noise * cur, axis=1, keepdims=True) * cur
        cand = _project(cur + noise)
        cand_E, _ = _energy_grad(cand)
        if cand_E < cur_E or rng.random() < math.exp(-(cand_E - cur_E) / max(T, 1e-12)):
            cur, cur_E = cand, cand_E
        # short GD polish from current
        cur, cur_E, local_pts, local_E = _gd_step(cur, 1e-3, 30, deadline, best_E)
        if local_E < best_E:
            best_E = local_E
            best_pts = local_pts.copy()
        T *= decay
    return best_pts, best_E


def _polish(pts, deadline):
    """High-precision final polish: many iterations, tight tolerance."""
    E, G = _energy_grad(pts)
    best_pts = pts.copy()
    best_E = E
    lr = 1e-3
    for it in range(200000):
        if time.time() > deadline:
            break
        Gt = G - np.sum(G * pts, axis=1, keepdims=True) * pts
        gnorm = np.linalg.norm(Gt)
        pts = _project(pts - lr * Gt)
        E, G = _energy_grad(pts)
        if E < best_E:
            best_E = E
            best_pts = pts.copy()
        # adaptive lr
        if gnorm < 1e-10:
            break
        # gentle decay
        lr *= 0.9995
    return best_pts, best_E


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY
    rng = np.random.default_rng(RNG_SEED)

    # best-so-far across all phases
    best_pts = None
    best_E = math.inf

    # Phase 0: Fibonacci start, then GD
    pts = _fibonacci_start()
    pts, E, bpts, bE = _gd_step(pts, 1e-2, 4000, deadline, best_E)
    if bE < best_E:
        best_E, best_pts = bE, bpts

    # Phase 1: Simulated annealing from the GD result
    if time.time() < deadline:
        bpts, bE = _sa(best_pts, start_temp=0.5, end_temp=1e-4,
                       iters=3000, deadline=deadline,
                       best_pts=best_pts, best_E=best_E, rng=rng)
        if bE < best_E:
            best_E, best_pts = bE, bpts

    # Phase 2: a few more random restarts if time allows (cheap)
    remaining = deadline - time.time()
    if remaining > 20.0:
        for restart in range(3):
            if time.time() > deadline:
                break
            rpts = _fibonacci_start()
            # jitter the start
            rpts = _project(rpts + rng.standard_normal((N, 3)) * 0.1)
            rpts, RE, rbpts, rbE = _gd_step(rpts, 1e-2, 2000, deadline, best_E)
            if rbE < best_E:
                best_E, best_pts = rbE, rbpts
            # short SA from this restart
            if time.time() < deadline:
                sbpts, sbE = _sa(rbpts, start_temp=0.3, end_temp=1e-4,
                                 iters=1000, deadline=deadline,
                                 best_pts=best_pts, best_E=best_E, rng=rng)
                if sbE < best_E:
                    best_E, best_pts = sbE, sbpts

    # Phase 3: high-precision polish
    if time.time() < deadline:
        bpts, bE = _polish(best_pts, deadline)
        if bE < best_E:
            best_E, best_pts = bE, bpts

    # Final safety: ensure finite, nonzero, unit-normalized
    best_pts = _project(best_pts)
    if not np.all(np.isfinite(best_pts)):
        # fallback to a valid Fibonacci configuration
        best_pts = _fibonacci_start()
    return best_pts
