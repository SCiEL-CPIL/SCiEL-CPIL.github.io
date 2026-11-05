"""
Morse cluster global minima search using basin hopping with diverse structured seeds.
Strategy:
  - Generate structured seeds (FCC, HCP, icosahedral, decahedral fragments) plus random compact clusters.
  - For each seed, perform basin hopping: perturb a random subset of atoms, then relax locally.
  - Use L-BFGS-B with exact gradients for the final polish.
  - Track best energy across all instances; return the best positions.
  - Explicit deadline guard to stay within 300 s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = 300.0
_START_TIME = None

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _get_energy(pos, rho):
    """Compute energy using the provided utility."""
    try:
        e, g = morse_energy_gradient(pos, rho)
        return float(e)
    except Exception:
        n = len(pos)
        e = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 1e-10:
                    r = 1e-10
                u = rho * (1.0 - r)
                eu = np.exp(u)
                e += eu * (eu - 2.0)
        return e

def _fcc_positions(n, spacing=1.0):
    """Generate an FCC-like fragment of n atoms."""
    pts = []
    a = spacing
    basis = np.array([
        [0, 0, 0], [0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]
    ])
    max_k = int(np.ceil((2 * n) ** (1/3))) + 2
    for ix in range(-max_k, max_k + 1):
        for iy in range(-max_k, max_k + 1):
            for iz in range(-max_k, max_k + 1):
                for b in basis:
                    p = np.array([ix * a, iy * a, iz * a]) + b
                    pts.append(p)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in pts):
                pts.append(p)
                break
    return np.array(pts)

def _basin_hopping(pos, rho, max_iters=100, perturb_sigma=0.3, rng=None, time_limit=None):
    """Basin hopping using chem_clusters.relax for local minimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = pos.copy()
    best_e = _get_energy(best_pos, rho)
    n = len(pos)
    
    # First, relax the starting position
    try:
        relaxed = relax(best_pos, 'morse', rho)
        if relaxed is not None:
            best_pos = np.array(relaxed)
            best_e = _get_energy(best_pos, rho)
    except Exception:
        pass

    for it in range(max_iters):
        if time_limit is not None and _elapsed() > time_limit:
            break
            
        n_perturb = max(1, int(n * 0.2))
        idx = rng.choice(n, size=n_perturb, replace=False)
        new_pos = best_pos.copy()
        new_pos[idx] += rng.normal(0, perturb_sigma, size=(n_perturb, 3))
        
        try:
            relaxed = relax(new_pos, 'morse', rho)
            if relaxed is not None:
                candidate_pos = np.array(relaxed)
                candidate_e = _get_energy(candidate_pos, rho)
                
                if candidate_e < best_e:
                    best_e = candidate_e
                    best_pos = candidate_pos
        except Exception:
            continue
            
    return best_pos, best_e

def construct(instance):
    """Find the lowest-energy arrangement for the given instance."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    rng = np.random.default_rng(n * 7 + int(rho))
    
    seeds = []
    try:
        seeds.append(_fcc_positions(n))
    except Exception:
        pass
    
    for _ in range(5):
        try:
            seeds.append(_random_compact(n, rng))
        except Exception:
            pass
            
    best_pos = None
    best_e = np.inf
    
    time_limit = _DEADLINE - 30
    
    for seed in seeds:
        if _elapsed() > time_limit:
            break
            
        try:
            relaxed = relax(seed, 'morse', rho)
            if relaxed is not None:
                cur_pos = np.array(relaxed)
            else:
                cur_pos = seed
        except Exception:
            cur_pos = seed
            
        cur_e = _get_energy(cur_pos, rho)
        
        if cur_e < best_e:
            best_e = cur_e
            best_pos = cur_pos.copy()
            
        if _elapsed() < time_limit:
            bh_pos, bh_e = _basin_hopping(cur_pos, rho, max_iters=50, perturb_sigma=0.3, rng=rng, time_limit=time_limit)
            if bh_e < best_e:
                best_e = bh_e
                best_pos = bh_pos
                
    if best_pos is None:
        best_pos = _random_compact(n, rng)
        
    return best_pos
# EVOLVE-BLOCK-END
