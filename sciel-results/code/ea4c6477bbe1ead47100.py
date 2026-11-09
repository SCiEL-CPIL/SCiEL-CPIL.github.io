"""Morse cluster global minima: Multi-start with diverse structural prototypes."""
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
    """Morse energy and gradient with overlap penalty."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    
    # Add penalty for overlaps
    min_d = np.min(d)
    if min_d < 0.5:
        penalty = 1e6 * (0.5 - min_d) ** 2
        # Gradient of penalty
        grad = np.zeros_like(coords)
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i != j:
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-8:
                diff /= dist
                grad[i] += 2e6 * (0.5 - min_d) * diff
                grad[j] -= 2e6 * (0.5 - min_d) * diff
        return penalty + 1e10, grad
    
    e = np.exp(rho * (1.0 - d))
    energy = np.sum(e * (e - 2.0)) / 2.0
    
    # Gradient: dV/dr = rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    # dV/dx_i = sum_{j!=i} dV/dr_ij * (r_i - r_j)/r_ij
    grad = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            rij = coords[i] - coords[j]
            r = d[i, j]
            if r > 1e-8:
                dVdr = rho * e[i, j] * (e[i, j] - 1.0)
                grad[i] += dVdr * rij / r
                grad[j] -= dVdr * rij / r
    
    return energy, grad

def _local_minimize(rho, x0, max_iter=200):
    """Aggressive local minimization using L-BFGS-B."""
    x_flat = x0.flatten()
    n = len(x0)
    
    def func_and_grad(x):
        coords = x.reshape(n, 3)
        e, g = _morse_energy_and_grad(rho, coords)
        return e, g
    
    try:
        res = minimize(func_and_grad, x_flat, method='L-BFGS-B', 
                       jac=True, options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 20})
        x_opt = res.x.reshape(n, 3)
        e_opt = res.fun
        if e_opt > 1e8:
            # Try to fix overlaps
            x_opt = _fix_overlaps(rho, x_opt)
            e_opt = _morse_energy(rho, x_opt)
        return x_opt, e_opt
    except Exception:
        return x0, _morse_energy(rho, x0)

def _fix_overlaps(rho, coords, max_iter=100):
    """Fix overlapping atoms by pushing them apart."""
    coords = coords.copy()
    for _ in range(max_iter):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= 0.5:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            # Random direction
            diff = np.random.randn(3)
            dist = np.linalg.norm(diff)
        diff /= dist
        push = 0.05
        coords[i] += push * diff
        coords[j] -= push * diff
    return coords

def _generate_fcc_seeds(n, rng, count=3):
    """Generate FCC-like seeds with rotations and noise."""
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
        noisy = rotated + rng.normal(0, 0.15, rotated.shape)
        seeds.append(noisy)
    return seeds

def _generate_hcp_seeds(n, rng, count=2):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    c = np.sqrt(8/3) * a
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
    
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # AB stacking
                if k % 2 == 0:
                    pts.append([i*a, j*a, k*c/2])
                    pts.append([(i+0.5)*a, (j+0.5)*a, k*c/2])
                else:
                    pts.append([(i+1/3)*a, (j+2/3)*a, k*c/2])
                    pts.append([(i+2/3)*a, (j+1/3)*a, k*c/2])
    
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
        noisy = rotated + rng.normal(0, 0.15, rotated.shape)
        seeds.append(noisy)
    return seeds

def _generate_icosahedral_seeds(n, rng, count=2):
    """Generate icosahedral-like seeds (nested shells)."""
    seeds = []
    
    for _ in range(count):
        # Generate points on nested spherical shells
        pts = []
        # Center
        pts.append([0, 0, 0])
        
        # Icosahedral vertices for first shell
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        
        # First shell at r=1
        for v in ico_verts:
            if len(pts) < n:
                pts.append(v * 1.0)
        
        # Second shell at r=2 (more points)
        if len(pts) < n:
            n_second = min(n - len(pts), 30)
            angles_theta = rng.uniform(0, np.pi, n_second)
            angles_phi = rng.uniform(0, 2*np.pi, n_second)
            for theta, phi_a in zip(angles_theta, angles_phi):
                pts.append([
                    2.0 * np.sin(theta) * np.cos(phi_a),
                    2.0 * np.sin(theta) * np.sin(phi_a),
                    2.0 * np.cos(theta)
                ])
        
        # Third shell at r=3 if needed
        if len(pts) < n:
            n_third = n - len(pts)
            angles_theta = rng.uniform(0, np.pi, n_third)
            angles_phi = rng.uniform(0, 2*np.pi, n_third)
            for theta, phi_a in zip(angles_theta, angles_phi):
                pts.append([
                    3.0 * np.sin(theta) * np.cos(phi_a),
                    3.0 * np.sin(theta) * np.sin(phi_a),
                    3.0 * np.cos(theta)
                ])
        
        pts = np.array(pts[:n])
        pts += rng.normal(0, 0.1, pts.shape)
        seeds.append(pts)
    
    return seeds

def _generate_spherical_shells(n, rng, count=2):
    """Generate points on spherical shells."""
    seeds = []
    
    for _ in range(count):
        pts = []
        # Distribute points across multiple shells
        n_shells = max(2, int(np.cbrt(n)) + 1)
        pts_per_shell = n // n_shells
        
        for shell_idx in range(n_shells):
            r = 1.0 + shell_idx * 0.8
            n_pts = min(pts_per_shell, n - len(pts))
            if n_pts <= 0:
                break
            
            # Fibonacci sphere for even distribution
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_pts):
                y = 1 - 2 * (i + 0.5) / n_pts
                radius = np.sqrt(1 - y*y)
                theta = golden_angle * i
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                pts.append([x * r, y * r, z * r])
        
        pts = np.array(pts[:n])
        pts += rng.normal(0, 0.1, pts.shape)
        seeds.append(pts)
    
    return seeds

def _generate_random_compact_seeds(n, rng, count=2):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        box_size = (n * 0.5) ** (1/3) * 2
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        seeds.append(pts)
    return seeds

def _basin_hopping_chain(rho, x0, rng, duration):
    """
    Run a single basin hopping chain with L-BFGS-B local minimization.
    """
    n = len(x0)
    cur_x = x0.copy()
    
    # Initial local minimization
    cur_x, cur_e = _local_minimize(rho, cur_x)
    
    if cur_e > 1e8:
        cur_x = _fix_overlaps(rho, cur_x)
        cur_e = _morse_energy(rho, cur_x)
        if cur_e > 1e8:
            return cur_x, 1e10
    
    best_x = cur_x.copy()
    best_e = cur_e
    
    temp = 0.3
    t_start = time.time()
    
    while _remaining() > 1.0 and (time.time() - t_start) < duration:
        # Adaptive shake
        shake_scale = 0.15 + 0.25 * np.sqrt(temp)
        
        # Perturb subset of atoms
        n_perturb = max(1, int(n * 0.15))
        idx = rng.choice(n, size=n_perturb, replace=False)
        noise = rng.normal(0, shake_scale, size=(n_perturb, 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        # Local minimize
        x_new, e_new = _local_minimize(rho, x_new)
        
        if e_new > 1e8:
            x_new = _fix_overlaps(rho, x_new)
            e_new = _morse_energy(rho, x_new)
            if e_new > 1e8:
                continue
        
        # Metropolis acceptance
        dE = e_new - cur_e
        if dE < 0 or (temp > 0 and rng.random() < np.exp(-dE / temp)):
            cur_x = x_new
            cur_e = e_new
            if cur_e < best_e:
                best_e = cur_e
                best_x = cur_x.copy()
        
        # Cool down
        temp *= 0.97
    
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds from different structural families
    fcc_seeds = _generate_fcc_seeds(n, rng, count=2)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=1)
    ico_seeds = _generate_icosahedral_seeds(n, rng, count=1)
    shell_seeds = _generate_spherical_shells(n, rng, count=1)
    random_seeds = _generate_random_compact_seeds(n, rng, count=1)
    
    # Select 4 diverse starting points
    chains_starts = [
        fcc_seeds[0],
        hcp_seeds[0] if hcp_seeds else fcc_seeds[1],
        ico_seeds[0] if ico_seeds else shell_seeds[0],
        random_seeds[0]
    ]
    
    # Allocate time
    total_time = _remaining()
    polish_time = max(3.0, total_time * 0.15)
    search_time = total_time - polish_time
    time_per_chain = search_time / 4.0
    
    best_x_global = None
    best_e_global = np.inf
    
    def run_chain(args):
        x0, t_dur, seed_id = args
        x_res, e_res = _basin_hopping_chain(rho, x0, np.random.default_rng(seed_id), t_dur)
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
    
    # Final polish
    if _remaining() > polish_time:
        rng_polish = np.random.default_rng(42)
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time)
    
    # Final validation and fix
    e_final = _morse_energy(rho, best_x_global)
    if e_final > 1e8:
        best_x_global = _fix_overlaps(rho, best_x_global)
        e_final = _morse_energy(rho, best_x_global)
    
    return best_x_global

# EVOLVE-BLOCK-END
