"""Morse cluster global minima: Multi-start with diverse structural prototypes, L-BFGS-B, and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 310.0  # Safety margin before 330s kill
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
    """Morse energy and gradient with overlap penalty."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    
    min_d = np.min(d)
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    
    e = np.exp(rho * (1.0 - d))
    energy = np.sum(e * (e - 2.0)) / 2.0
    
    dV_dr = rho * e * (e - 1.0)
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    mask = d > 0
    d_safe = np.where(mask, d, 1.0)
    
    grad = np.sum(dV_dr[:, :, np.newaxis] * diff / d_safe[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _local_minimize(rho, x0, maxiter=200):
    """
    Perform local minimization using L-BFGS-B with tight tolerance.
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
                       options={'maxiter': maxiter, 'gtol': 1e-6, 'ftol': 1e-10})
        x_opt = res.x.reshape(n, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt < 1e8:
            return e_opt, x_opt
    except Exception:
        pass
    
    # Fallback to original relax if minimize fails
    e_rel, x_rel = relax(x0, 'morse', rho)
    return e_rel, x_rel

def _generate_fcc_seeds(n, rng, count=3):
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
                x = i * a + 0.5 * (j % 2) * a
                y = j * a * np.sqrt(3)/2
                z = k * c
                pts.append([x, y, z])
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

def _generate_icosahedral_seeds(n, rng, count=2):
    """Generate Icosahedral-like seeds (nested shells)."""
    seeds = []
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    shells = [1.0, 2.0, 3.0, 4.0, 5.0]
    all_pts = []
    for r in shells:
        for v in verts:
            all_pts.append(v * r)
    
    while len(all_pts) < n:
        all_pts.append(rng.uniform(-2, 2, size=3))
    
    all_pts = np.array(all_pts[:n])
    all_pts -= all_pts.mean(axis=0)
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = all_pts @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _generate_random_compact_seeds(n, rng, count=3):
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
    Uses adaptive shake scale based on progress.
    """
    best_x = x0.copy()
    cur_x = x0.copy()
    
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
        # Adaptive shake: start moderate, decrease over time
        shake_scale = 0.15 + 0.25 * np.sqrt(temp)
        
        # Shake a random subset of atoms
        n_shake = max(1, int(n * 0.15))
        idx = rng.choice(n, size=n_shake, replace=False)
        noise = rng.normal(0, shake_scale, size=(n_shake, 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        e_rel, x_rel = _local_minimize(rho, x_new)
        
        if e_rel > 1e8:
            continue

        dE = e_rel - cur_e
        if dE < 0 or (temp > 0 and rng.random() < np.exp(-dE / temp)):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        
        temp *= 0.97
    
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    fcc_seeds = _generate_fcc_seeds(n, rng, count=2)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=1)
    ico_seeds = _generate_icosahedral_seeds(n, rng, count=1)
    random_seeds = _generate_random_compact_seeds(n, rng, count=2)
    
    # Select initial points for 4 parallel chains
    chains_starts = []
    chains_starts.append(fcc_seeds[0])
    chains_starts.append(hcp_seeds[0])
    chains_starts.append(ico_seeds[0])
    chains_starts.append(random_seeds[0])
    
    # Allocate time to each chain
    total_time = _remaining()
    polish_time = max(5.0, total_time * 0.12)
    search_time = total_time - polish_time
    time_per_chain = search_time / 4.0
    
    # Run parallel chains
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
            except Exception:
                pass
                
    if best_x_global is None:
        return chains_starts[0]
        
    # Final Polish with remaining time: tighter local minimization
    if _remaining() > polish_time:
        # Do a final high-precision polish on the best structure
        rng_polish = np.random.default_rng(42)
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time, 999)
        
        # Extra tight local minimization
        e_tight, x_tight = _local_minimize(rho, best_x_global, maxiter=500)
        if e_tight < best_e_global and e_tight < 1e8:
            best_x_global = x_tight
            best_e_global = e_tight
        
    # Final validation
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
