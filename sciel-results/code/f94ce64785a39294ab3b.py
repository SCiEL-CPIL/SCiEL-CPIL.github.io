"""Basin-hopping solver for Morse cluster global minima.

Strategy:
  1. Generate diverse initial seeds (lattice-based + random compact).
  2. Use basin hopping with adaptive cooling to escape shallow local minima.
  3. Parallelize across multiple independent basin-hopping chains.
  4. Return the best configuration found within the time budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _morse_E(P, rho):
    """Exact energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    r = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
    """Gradient of exact Morse energy w.r.t. coordinates (n,3)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    """Robustly nudge so every pair is >= min_d apart."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(500):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        if len(idx) == 0:
            break
        i, j = idx[0]
        gap = min_d - d[i, j] + 0.01
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0]); nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=300, ftol=1e-10):
    """L-BFGS-B on exact Morse energy; returns (E, P)."""
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        P = x.reshape(n, 3)
        return _morse_E(P, rho)

    def gfun(x):
        P = x.reshape(n, 3)
        return _morse_grad(P, rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 10000,
                                "ftol": ftol, "gtol": 1e-7, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _fcc_seeds(n, rho):
    """Generate FCC lattice seeds with varying lattice constants."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1, 1.2]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _bcc_seeds(n, rho):
    """Generate BCC lattice seeds."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _random_compact(n, seed=None):
    """Random compact cluster."""
    rng = np.random.RandomState(seed)
    R = n ** (1/3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _basin_hopping(P0, rho, n, max_iters=200, T0=0.5, T_end=0.01, 
                   step_scale=0.5, deadline=None, rng_seed=0):
    """
    Basin hopping: local minimization + perturbation + Metropolis acceptance.
    Returns (best_E, best_P).
    """
    rng = np.random.RandomState(rng_seed)
    
    # Initial local optimization
    P0 = _enforce_min_dist(P0)
    E, P = _local_opt(P0, rho, max_iter=200)
    if E is None:
        E, P = 1e10, P0
    
    best_E = E
    best_P = P.copy()
    
    # Adaptive perturbation scale
    step = step_scale
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Perturb
        noise = rng.normal(0, step, size=P.shape)
        P_new = P + noise
        P_new = _enforce_min_dist(P_new)
        
        # Local optimization
        E_new, P_new = _local_opt(P_new, rho, max_iter=150)
        if E_new is None:
            E_new, P_new = 1e10, P
        
        # Metropolis acceptance
        if E_new < best_E:
            best_E = E_new
            best_P = P_new.copy()
        
        T = T0 * (T_end / T0) ** (it / max(1, max_iters - 1))
        if E_new <= E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-12)):
            E, P = E_new, P_new
        else:
            # Rejected: reduce step size slightly
            step *= 0.95
        
        # Occasionally increase step to explore
        if it % 50 == 0 and it > 0:
            step = min(step * 1.3, 1.0)
    
    return best_E, best_P


def _run_chain(P0, rho, n, deadline, rng_seed):
    """Run one basin-hopping chain."""
    try:
        E, P = _basin_hopping(P0, rho, n, max_iters=150, 
                              T0=0.3, T_end=0.005,
                              step_scale=0.4, deadline=deadline,
                              rng_seed=rng_seed)
        return E, P
    except Exception:
        return None, None


def construct(instance):
    _init_timer()
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 42.0)
    
    best_E = float('inf')
    best_P = None
    
    # Generate diverse seeds
    seeds = []
    seeds.extend(_fcc_seeds(n, rho))
    seeds.extend(_bcc_seeds(n, rho))
    for s in range(3):
        seeds.append(_random_compact(n, seed=100 + s))
    
    # Ensure we have at least some seeds
    if not seeds:
        seeds = [_random_compact(n, seed=0)]
    
    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
        E, P = _local_opt(_enforce_min_dist(s), rho, max_iter=100)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P
    
    # Sort seeds by energy, use best ones for basin hopping
    seed_results.sort(key=lambda x: x[0])
    top_seeds = seed_results[:6] if len(seed_results) >= 6 else seed_results
    
    # Run basin hopping chains in parallel (up to 4 threads)
    n_chains = min(4, len(top_seeds))
    if n_chains > 0 and time.time() < deadline - 8.0:
        with _cf.ThreadPoolExecutor(max_workers=n_chains) as pool:
            futures = []
            for i in range(n_chains):
                P0 = top_seeds[i][1]
                futures.append(pool.submit(_run_chain, P0, rho, n, deadline, rng_seed=1000 + i))
            
            for f in futures:
                try:
                    E, P = f.result(timeout=1.0)
                    if E is not None and E < best_E:
                        best_E = E
                        best_P = P
                except Exception:
                    pass
    
    # If we still have time, do a final polish on the best configuration
    if best_P is not None and time.time() < deadline - 3.0:
        E, P = _local_opt(_enforce_min_dist(best_P), rho, max_iter=500, ftol=1e-12)
        if E is not None and E < best_E:
            best_E = E
            best_P = P
    
    # Fallback if nothing worked
    if best_P is None:
        best_P = _random_compact(n, seed=42)
        best_E = _morse_E(best_P, rho)
    
    # Final validity check
    best_P = _enforce_min_dist(best_P)
    
    # One last quick relaxation to ensure validity
    if time.time() < deadline - 1.0:
        E, P = _local_opt(best_P, rho, max_iter=50)
        if E is not None:
            best_P = P
    
    best_P = _enforce_min_dist(best_P)
    
    return best_P

# EVOLVE-BLOCK-END
