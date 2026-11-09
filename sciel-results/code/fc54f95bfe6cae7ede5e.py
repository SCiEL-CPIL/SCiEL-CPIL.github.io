"""Morse cluster global minima: Parallel Adaptive Basin Hopping with Vectorized Gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 315.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0

def _morse_energy_and_grad(rho, coords):
    """
    Morse energy and gradient.
    Uses vectorized computation for speed.
    Returns (energy, gradient).
    """
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    
    min_d = np.min(d)
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    
    # u = rho * (1 - d)
    u = rho * (1.0 - d)
    # Clip u to prevent overflow in exp. 
    # If r is small, u is large positive. exp(large) -> inf.
    # However, if d >= 0.5, u <= rho * 0.5.
    # For rho=10, u <= 5. exp(5) ~ 150. Safe.
    # For rho=3, u <= 1.5. Safe.
    # So clipping isn't strictly necessary for valid inputs, but good for safety.
    e_u = np.exp(u)
    
    # V_ij = e_u * (e_u - 2)
    v_ij = e_u * (e_u - 2.0)
    
    # Energy is sum over i<j. 
    # Sum of all i!=j is 2 * E.
    # But diagonal is inf, so v_ij diagonal is inf? 
    # Wait, u diagonal is rho * (1 - inf) = -inf. exp(-inf) = 0.
    # So v_ij diagonal is 0 * (0 - 2) = 0.
    # So sum(v_ij) is sum over i!=j.
    energy = 0.5 * np.sum(v_ij)
    
    # Gradient
    # dV/dr = rho * e_u * (e_u - 1)
    dV_dr = rho * e_u * (e_u - 1.0)
    
    # diff[i, j] = coords[i] - coords[j]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # grad_i = sum_j (dV_dr[i, j] / d[i, j]) * diff[i, j]
    # d[i, i] is inf, dV_dr[i, i] is 0 (since e_u=0). 0/inf = 0.
    # But to be safe with numpy warnings or NaNs from 0/0 if d was 0 (which it isn't),
    # we can mask.
    mask = d > 0
    d_safe = np.where(mask, d, 1.0)
    dV_dr_safe = np.where(mask, dV_dr, 0.0)
    
    # factor[i, j] = dV_dr[i, j] / d[i, j]
    factor = dV_dr_safe / d_safe
    
    # grad = sum_j factor[i, j] * diff[i, j]
    # factor is (n, n, 1) broadcast? No, factor is (n, n).
    # We need factor[:, :, np.newaxis] * diff
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _local_minimize(rho, x0):
    """
    Perform local minimization using L-BFGS-B.
    Returns (energy, coords).
    """
    n = len(x0)
    x0_flat = x0.flatten()
    
    def func_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        e, g = _morse_energy_and_grad(rho, x)
        return e, g.flatten()
        
    # Check for overlaps first
    d = cdist(x0, x0)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        # Push apart
        for _ in range(50):
            d = cdist(x0, x0)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            if min_d >= 0.5: break
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            if i == j: continue
            diff = x0[i] - x0[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6: 
                x0[i] += 0.01
                continue
            diff /= dist
            x0[i] += 0.05 * diff
            x0[j] -= 0.05 * diff
            
    try:
        res = minimize(func_and_grad, x0_flat, method='L-BFGS-B', 
                       options={'maxiter': 100, 'gtol': 1e-4})
        x_opt = res.x.reshape(n, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt < 1e8:
            return e_opt, x_opt
    except Exception:
        pass
    
    # Fallback to original relax if minimize fails
    e_rel, x_rel = relax(x0, 'morse', rho)
    return e_rel, x_rel

def _generate_fcc_seeds(n, rng, count=5):
    """Generate multiple FCC-like seeds with slight perturbations and rotations."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_fcc = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_fcc @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _generate_hcp_seeds(n, rng, count=2):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    c = a * np.sqrt(8/3)
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # A and B layers
                x = i * a + 0.5 * (j % 2) * a
                y = j * a * np.sqrt(3)/2
                z = k * c
                pts.append([x, y, z])
                # B layer offset
                x2 = i * a + 0.5 * a + 0.5 * (j % 2) * a
                y2 = j * a * np.sqrt(3)/2 + a * np.sqrt(3)/6
                z2 = k * c + c/3
                pts.append([x2, y2, z2])
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_hcp = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_hcp @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _generate_random_compact_seeds(n, rng, count=5):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        box_size = (n * 0.5) ** (1/3) * 2
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        seeds.append(pts)
    return seeds

def _basin_hopping_chain(rho, x0, rng, duration, seed_id):
    """
    Run a single basin hopping chain for a specified duration.
    """
    best_x = x0.copy()
    cur_x = x0.copy()
    
    # Initial relax
    e_rel, x_rel = _local_minimize(rho, x0)
    if e_rel > 1e8:
        return best_x, 1e10

    cur_e = e_rel
    best_e = e_rel
    best_x = x_rel.copy()
    
    temp = 0.5
    t_start = time.time()
    n = len(x0)
    
    while _remaining() > 1.0 and (time.time() - t_start) < duration:
        shake_scale = 0.2 + 0.3 * np.sqrt(temp)
        
        idx = rng.choice(n, size=max(1, int(n*0.2)), replace=False)
        noise = rng.normal(0, shake_scale, size=(len(idx), 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        # Relax
        e_rel, x_rel = _local_minimize(rho, x_new)
        
        if e_rel > 1e8:
            continue

        dE = e_rel - cur_e
        if dE < 0 or (temp > 0 and np.random.rand() < np.exp(-dE / temp)):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        
        temp *= 0.95
        
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    fcc_seeds = _generate_fcc_seeds(n, rng, count=4)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=2)
    random_seeds = _generate_random_compact_seeds(n, rng, count=4)
    
    # Select 4 starts for parallel chains
    # Mix FCC, HCP, Random
    chains_starts = [
        fcc_seeds[0],
        fcc_seeds[1],
        hcp_seeds[0] if len(hcp_seeds) > 0 else random_seeds[0],
        random_seeds[0]
    ]
    
    total_time = _remaining()
    polish_time = max(5.0, total_time * 0.1)
    search_time = total_time - polish_time
    time_per_chain = search_time / 4.0
    
    best_x_global = None
    best_e_global = np.inf
    
    def run_chain(args):
        x0, t_dur, seed_id = args
        x_res, e_res = _basin_hopping_chain(rho, x0, np.random.default_rng(seed_id), t_dur, seed_id)
        return x_res, e_res

    args = []
    for i, start in enumerate(chains_starts):
        args.append((start, time_per_chain, n * 100 + i))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_chain, arg) for arg in args]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                x_res, e_res = future.result()
                if e_res < best_e_global:
                    best_e_global = e_res
                    best_x_global = x_res
            except Exception as e:
                pass
                
    if best_x_global is None:
        return chains_starts[0]
        
    if _remaining() > polish_time:
        rng_polish = np.random.default_rng(42)
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time, 999)
        
    e_final = _morse_energy(rho, best_x_global)
    if e_final < 1e8:
        return best_x_global
    else:
        d = cdist(best_x_global, best_x_global)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.5:
             for _ in range(100):
                 d = cdist(best_x_global, best_x_global)
                 np.fill_diagonal(d, np.inf)
                 min_d = np.min(d)
                 if min_d >= 0.5: break
                 idx = np.unravel_index(np.argmin(d), d.shape)
                 i, j = idx
                 if i == j: continue
                 diff = best_x_global[i] - best_x_global[j]
                 dist = np.linalg.norm(diff)
                 if dist < 1e-6: continue
                 diff /= dist
                 best_x_global[i] += 0.01 * diff
                 best_x_global[j] -= 0.01 * diff
        return best_x_global

# EVOLVE-BLOCK-END
