"""
Dzugutov cluster global minima search.
Combines high-quality polytetrahedral seeding with the robust relax API.
Uses multiprocessing to explore multiple basins in parallel within the time budget.
"""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Global deadline guard
TOTAL_TIME_LIMIT = 310.0  # Leave 20s safety margin for 330s kill
_start_time = None
_best_results = {}  # Cache best positions per n

# Dzugutov Potential Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())

def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient using the provided helper."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass
    
    # Fallback manual computation (vectorized)
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-8:
                continue
            
            r_vec = diff[i, j]
            
            if r < a:
                V = (A * (r ** (-m) - B)) * np.exp(c / (r - a))
                dVdr = A * (-m * r ** (-m - 1) - c / (r - a) ** 2 * (r ** (-m) - B)) * np.exp(c / (r - a))
            elif r < b:
                V = B * np.exp(d / (r - b))
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec
    
    return energy, grad

def _make_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron-like structure.
    For n=55, this is the exact Mackay structure.
    For other n, we trim or add atoms to the shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # Shell 0: Center
    positions = [np.zeros(3)]
    remaining = n - 1
    
    # Shell 1: 12 vertices (radius ~ 1.0)
    r1 = 1.0
    shell1 = v * r1
    take1 = min(12, remaining)
    positions.extend(list(shell1[:take1]))
    remaining -= take1
    
    # Shell 2: 30 edge midpoints (radius ~ 1.7)
    if remaining > 0:
        # Generate edge midpoints
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                # Check if they are connected (distance ~ 1.05 in unit ico)
                d = np.linalg.norm(v[i] - v[j])
                if d < 1.1: # Approximate edge length in unit ico
                    mid = (v[i] + v[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edges.append(mid)
        
        # There should be 30 unique edge midpoints
        # Remove duplicates
        unique_edges = []
        for e in edges:
            if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
                unique_edges.append(e)
        
        r2 = np.sqrt(2 * (1 + 1/phi)) # Geometric radius for edge midpoints
        shell2 = np.array(unique_edges) * r2
        take2 = min(len(shell2), remaining)
        positions.extend(list(shell2[:take2]))
        remaining -= take2
    
    # Shell 3: 20 face centers (radius ~ 2.2)
    if remaining > 0:
        faces = []
        # Hardcode or generate faces. For simplicity, use a known set of 20 faces for icosahedron
        # Indices of vertices for faces
        face_indices = [
            [0, 1, 8], [0, 1, 11], [0, 2, 8], [0, 2, 9],
            [0, 3, 9], [0, 3, 11], [1, 4, 8], [1, 4, 10],
            [1, 5, 10], [1, 5, 11], [2, 4, 8], [2, 4, 9],
            [2, 5, 9], [2, 5, 10], [3, 6, 9], [3, 6, 11],
            [3, 7, 11], [3, 7, 10], [6, 4, 8], [6, 7, 10]
        ]
        
        face_centers = []
        for f in face_indices:
            fc = (v[f[0]] + v[f[1]] + v[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
            
        r3 = np.sqrt(3 * (1 + 1/phi)) # Approx radius for face centers
        shell3 = np.array(face_centers) * r3
        take3 = min(len(shell3), remaining)
        positions.extend(list(shell3[:take3]))
        remaining -= take3
        
    # If still remaining, add random atoms on a larger shell
    if remaining > 0:
        r4 = 2.8
        for _ in range(remaining):
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * r4
            positions.append(p)
            
    # If too many atoms, remove some from the outermost shell
    if len(positions) > n:
        # Sort by radius and keep closest n
        radii = np.linalg.norm(np.array(positions), axis=1)
        indices = np.argsort(radii)[:n]
        positions = [positions[i] for i in indices]
        
    return np.array(positions)

def _make_fcc_cluster(n, spacing, rng):
    """Generate a compact FCC-like cluster."""
    positions = [np.zeros(3)]
    remaining = n - 1
    
    a = spacing * 0.9
    lattice_pts = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pc = np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a
                    lattice_pts.append(pc)
    
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]
    
    # Remove origin
    nonzero_mask = dists > 0.1
    lattice_pts = lattice_pts[nonzero_mask]
    
    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])
        
    while len(positions) < n:
        p = rng.uniform(-4 * spacing, 4 * spacing, 3)
        positions.append(p)
        
    return np.array(positions[:n])

def _make_random_compact(n, spacing, rng):
    """Generate a random compact cluster."""
    positions = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    max_attempts = 5000
    
    for _ in range(max_attempts):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)
                
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
        
    return np.array(positions[:n])

def _enforce_min_distance(positions, rng, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(20):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = rng.normal(0, 1, 3)
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos

def _worker(args):
    """Worker function for parallel relaxation."""
    positions, n, seed, time_limit = args
    rng = np.random.default_rng(seed)
    
    # Enforce min distance
    positions = _enforce_min_distance(positions, rng)
    
    try:
        # Primary strategy: Use the provided relax API
        # relax typically returns (energy, positions)
        energy, relaxed_pos = relax(positions, 'dzugutov')
        
        # If relax fails or returns None, try L-BFGS
        if relaxed_pos is None:
            def objective(x):
                pos = x.reshape(-1, 3)
                e, g = _dzugutov_energy_gradient(pos)
                return e, g.flatten()
            
            result = minimize(objective, positions.flatten(), method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-10})
            energy = result.fun
            relaxed_pos = result.x.reshape(-1, 3)
            
        return energy, relaxed_pos
    except Exception as e:
        return np.inf, positions

def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    
    if n in _best_results:
        return _best_results[n]
        
    # Determine time budget for this instance
    # We have 6 instances total.
    # We want to spend roughly equal time on each, but prioritize if we are behind.
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    
    # Rough allocation: 40s per instance, but dynamic based on remaining
    base_budget = 45.0
    remaining = _remaining()
    instances_left = len(instances_to_do) - idx
    # Ensure we don't spend more than remaining
    time_budget = min(base_budget, remaining / max(1, instances_left) * 1.2) 
    time_budget = max(time_budget, 10.0) # Minimum 10s
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Mackay Icosahedron (best for n=55, good for others)
    mackay = _make_mackay_icosahedron(n, rng)
    seeds.append(mackay)
    
    # 2. FCC-like
    fcc = _make_fcc_cluster(n, 1.1, rng)
    seeds.append(fcc)
    
    # 3. Random Compact
    rand1 = _make_random_compact(n, 1.1, rng)
    seeds.append(rand1)
    
    # 4. Another Random with different spacing
    rand2 = _make_random_compact(n, 1.0, rng)
    seeds.append(rand2)
    
    # 5. Perturbed Mackay
    mackay_pert = mackay + rng.normal(0, 0.1, (n, 3))
    seeds.append(mackay_pert)
    
    # 6. Another Random
    rand3 = _make_random_compact(n, 1.2, rng)
    seeds.append(rand3)
    
    # Parallelize relaxation of these seeds
    # Use up to 4 threads
    num_workers = min(4, len(seeds))
    
    # Create args for workers
    worker_args = []
    for i, seed in enumerate(seeds[:num_workers]):
        # Allocate time per worker
        worker_time = time_budget / num_workers
        worker_args.append((seed, n, 42 + n + i, worker_time))
        
    best_energy = np.inf
    best_pos = None
    
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker, worker_args)
            
            for energy, pos in results:
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
    except Exception as e:
        # Fallback to sequential if multiprocessing fails
        for seed in seeds[:num_workers]:
            if _remaining() < 5:
                break
            worker_args = (seed, n, 42 + n, time_budget / len(seeds))
            energy, pos = _worker(worker_args)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
                
    # If no valid result, use the first seed relaxed sequentially
    if best_pos is None:
        best_pos = _enforce_min_distance(seeds[0], rng)
        try:
            energy, relaxed_pos = relax(best_pos, 'dzugutov')
            if relaxed_pos is not None:
                best_pos = relaxed_pos
        except:
            pass
            
    # Final check: ensure min distance
    best_pos = _enforce_min_distance(best_pos, rng)
    
    _best_results[n] = best_pos
    return best_pos
