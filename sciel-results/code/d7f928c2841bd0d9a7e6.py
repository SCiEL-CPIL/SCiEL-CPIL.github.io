"""New program: Basin Hopping with FCC fragment initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _get_kind_params(kind):
    """Map kind string to (n_exp, m_exp, c)."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _compute_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(positions)
    if n == 0:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_r_n = np.power(dists, -n_exp)
    inv_r_m = np.power(dists, -m_exp)
    
    sum_inv_r_n = np.sum(inv_r_n, axis=1)
    sum_inv_r_m = np.sum(inv_r_m, axis=1)
    
    e_rep = 0.5 * np.sum(sum_inv_r_n)
    
    sum_inv_r_m_no_self = sum_inv_r_m - 1.0
    sum_inv_r_m_no_self = np.maximum(sum_inv_r_m_no_self, 0.0)
    
    e_att = -c * np.sum(np.sqrt(sum_inv_r_m_no_self))
    
    return e_rep + e_att

def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and min distance)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    # Efficient min distance check
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist

def _relax(positions, potential, kind):
    """Local optimization using L-BFGS-B. Returns (energy, positions)."""
    try:
        e, p = relax(positions, potential, kind)
        if np.isfinite(e) and _check_validity(p):
            return e, p
    except Exception:
        pass
    # Fallback: try minimize manually if relax fails or returns invalid
    try:
        n_exp, m_exp, c = _get_kind_params(kind)
        def fun(x):
            p = x.reshape(-1, 3)
            if not np.all(np.isfinite(p)):
                return 1e10
            # Add penalty for close contacts
            if len(p) > 1:
                dists = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, 1.0)
                min_d = np.min(dists)
                if min_d < 0.35:
                    return 1e10 + (0.35 - min_d) ** 2 * 1000
            e = _compute_energy(p, n_exp, m_exp, c)
            return e if np.isfinite(e) else 1e10
        
        def grad(x):
            p = x.reshape(-1, 3)
            if not np.all(np.isfinite(p)):
                return np.zeros_like(x)
            # Simple gradient if needed, but relax() should handle it. 
            # For fallback, we might just use finite diff or skip gradient.
            # L-BFGS-B without gradient is slow, but better than nothing.
            return None # Let minimize use numerical gradient if grad is None? No, it errors.
            # Actually, let's just return the energy and let minimize handle it? 
            # scipy.optimize.minimize(method='L-BFGS-B') requires jac. 
            # Let's use sutton_chen_energy_gradient if available.
            e, g = sutton_chen_energy_gradient(p, kind)
            if not np.isfinite(e):
                return np.zeros_like(x)
            if len(p) > 1:
                dists = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, 1.0)
                min_d = np.min(dists)
                if min_d < 0.35:
                    return np.zeros_like(x) # Gradient undefined/penalty zone, just stop
            return g.flatten()

        x0 = positions.flatten()
        res = minimize(fun, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
        if res.success or np.isfinite(res.fun):
            p = res.x.reshape(-1, 3)
            e = _compute_energy(p, n_exp, m_exp, c)
            if np.isfinite(e) and _check_validity(p):
                return e, p
    except Exception:
        pass
    return np.inf, positions

def _fcc_fragment(n, spacing=1.0):
    """Generate n points from an FCC lattice."""
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    fcc_pts = []
    for n1 in range(-max_coord, max_coord + 1):
        for n2 in range(-max_coord, max_coord + 1):
            for n3 in range(-max_coord, max_coord + 1):
                x = (n1 + n2) * 0.5
                y = (n2 + n3) * 0.5
                z = (n3 + n1) * 0.5
                fcc_pts.append((x, y, z))
    fcc_pts = np.array(fcc_pts) * spacing
    fcc_pts -= np.mean(fcc_pts, axis=0)
    radii = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(radii)
    return fcc_pts[idx[:n]]

def _random_compact(n, spacing=0.8, seed=42):
    """Random points with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0)
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        min_dist = 0.35
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 2000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _basin_hopping(starts, n_exp, m_exp, c, potential, kind, time_limit, seed=42):
    """
    Basin Hopping algorithm.
    starts: list of initial positions
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    best_E = np.inf
    best_P = None
    
    # Relax all starts first
    relaxed_starts = []
    for s in starts:
        if time.time() - start_time > time_limit:
            break
        e, p = _relax(s, potential, kind)
        if np.isfinite(e) and _check_validity(p):
            relaxed_starts.append((e, p))
            if e < best_E:
                best_E = e
                best_P = p.copy()
        
    if not relaxed_starts:
        return best_E, best_P

    # Sort by energy
    relaxed_starts.sort(key=lambda x: x[0])
    
    # Basin Hopping Loop
    current_E, current_P = relaxed_starts[0]
    if current_E < best_E:
        best_E = current_E
        best_P = current_P.copy()
        
    T = 1.0
    T0 = 1.0
    Tend = 0.01
    alpha = 0.99
    
    step_size = 0.5
    
    while time.time() - start_time < time_limit - 5:
        # Generate a random perturbation
        # Move a random subset of atoms or all atoms?
        # Standard BH: move all atoms by a random vector.
        # For large N, moving all atoms might be too drastic. 
        # Let's move a random fraction (e.g., 20-50%) or just all with small step.
        # Let's try moving all atoms with a Gaussian noise.
        
        noise = rng.normal(0, step_size, current_P.shape)
        new_P = current_P + noise
        
        # Relax the new structure
        new_E, new_P_relaxed = _relax(new_P, potential, kind)
        
        if not np.isfinite(new_E) or not _check_validity(new_P_relaxed):
            # If relaxation failed, try a smaller step or different move?
            # For now, just reject.
            T *= alpha
            continue
            
        delta_E = new_E - current_E
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_E = new_E
            current_P = new_P_relaxed
            if current_E < best_E:
                best_E = current_E
                best_P = current_P.copy()
        
        # Cooling
        T *= alpha
        if T < Tend:
            break
            
        # Adapt step size? 
        # If we've been stuck (no improvement) for a while, increase step size?
        # Simple heuristic: reduce step size slowly to refine
        step_size *= 0.999
        
    return best_E, best_P

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    n_exp, m_exp, c = _get_kind_params(kind)
    
    start_time = time.time()
    time_limit = 280.0 # Leave 20s buffer
    
    # Generate diverse starts
    starts = []
    
    # 1. FCC fragments with different spacings
    for spacing in [0.8, 1.0, 1.2]:
        try:
            fcc = _fcc_fragment(size, spacing=spacing)
            if _check_validity(fcc):
                starts.append(fcc)
        except Exception:
            pass
            
    # 2. Random compact structures
    for seed in [42, 123, 456, 789, 1000, 2000]:
        try:
            rnd = _random_compact(size, spacing=0.8, seed=seed)
            if _check_validity(rnd):
                starts.append(rnd)
        except Exception:
            pass
            
    # 3. Icosahedral-like (if size is small enough or just use as seed)
    # For larger sizes, FCC is better. Let's add a "snowball" or "layered" structure?
    # Let's stick to FCC and Random for robustness.
    
    if not starts:
        # Fallback
        starts.append(_random_compact(size, spacing=0.8, seed=42))
        
    # Run Basin Hopping
    best_E, best_P = _basin_hopping(starts, n_exp, m_exp, c, potential, kind, time_limit, seed=42)
    
    # If BH didn't find anything or time ran out, try a few more direct relaxations
    if best_P is None:
        for s in starts:
            e, p = _relax(s, potential, kind)
            if np.isfinite(e) and _check_validity(p):
                best_E = e
                best_P = p
                break
                
    if best_P is None:
        # Last resort
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _relax(best_P, potential, kind)
        
    # Final validation
    if not np.all(np.isfinite(best_P)):
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _relax(best_P, potential, kind)
        
    if not _check_validity(best_P):
        # Try to fix by perturbation and relaxation
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            perturbed = best_P + np.random.normal(0, 0.2, best_P.shape)
            e, p = _relax(perturbed, potential, kind)
            if np.isfinite(e) and _check_validity(p):
                best_E = e
                best_P = p
                break
                
    return best_P
# EVOLVE-BLOCK-END
