"""Symmetry-constrained Basin Hopping for Sutton-Chen TM clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 285.0  # Safety margin: evaluator kills at 330s

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.all(dists >= min_dist)

def _icosahedral_generators():
    """Generators for I_h symmetry: 1 center + 12 vertices + 30 edge midpoints + 20 face centers."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    pts.extend(v)
    
    # 30 edge midpoints
    edges = [
        (0,1),(0,2),(0,4),(0,5),
        (1,3),(1,4),(1,7),(1,8),
        (2,3),(2,6),(2,9),(2,10),
        (3,7),(3,8),
        (4,5),(4,8),(4,11),
        (5,9),(5,10),(5,11),
        (6,7),(6,9),(6,10),
        (7,11),
        (8,11),
        (9,10)
    ]
    for a, b in edges:
        mid = (v[a] + v[b]) / 2.0
        pts.append(mid / np.linalg.norm(mid))
        
    # 20 face centers
    faces = [
        (0,1,4),(0,1,8),(0,2,5),(0,2,9),(0,4,11),(0,5,11),
        (1,3,4),(1,3,7),(1,4,8),(1,7,8),
        (2,3,5),(2,3,6),(2,5,9),(2,6,10),
        (3,6,7),(3,7,11),
        (4,5,11),(4,8,11),
        (5,9,11),(5,10,11),
        (6,9,10),(6,10,11),
        (7,9,10),(7,10,11)
    ]
    for a, b, c in faces:
        fc = (v[a] + v[b] + v[c]) / 3.0
        pts.append(fc / np.linalg.norm(fc))
        
    return np.array(pts[:55])

def _octahedral_generators():
    """Generators for O_h symmetry: 1 center + 6 vertices + 12 edge midpoints + 8 face centers."""
    pts = [np.zeros(3)]
    # 6 vertices (cubic)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                if abs(i)+abs(j)+abs(k) == 3:
                    pts.append(np.array([i, j, k]) / np.sqrt(3.0))
    # 12 edge midpoints
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [0]:
                pts.append(np.array([i, j, k]) / np.sqrt(2.0))
            for j in [0]:
                for k in [-1, 1]:
                    pts.append(np.array([i, j, k]) / np.sqrt(2.0))
    # 8 face centers
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                pts.append(np.array([i, j, k]) / 3.0)
                
    return np.array(pts[:27])

def _decahedral_generators():
    """Generators for D_5h symmetry: 1 center + 10 vertices (2 pentagons) + 20 edge midpoints."""
    pts = [np.zeros(3)]
    # 10 vertices: two pentagons
    for sign in [-1, 1]:
        for i in range(5):
            angle = 2 * np.pi * i / 5
            r = 1.0
            z = sign * 0.5
            p = np.array([r * np.cos(angle), r * np.sin(angle), z])
            pts.append(p / np.linalg.norm(p))
            
    # 20 edge midpoints: 10 radial + 10 circumferential
    # Radial: between top and bottom pentagons
    for i in range(5):
        angle = 2 * np.pi * i / 5
        p_top = np.array([np.cos(angle), np.sin(angle), 0.5])
        p_bot = np.array([np.cos(angle), np.sin(angle), -0.5])
        mid = (p_top + p_bot) / 2.0
        pts.append(mid / np.linalg.norm(mid))
        
    # Circumferential: within each pentagon
    for sign in [-1, 1]:
        for i in range(5):
            angle1 = 2 * np.pi * i / 5
            angle2 = 2 * np.pi * (i+1) / 5
            p1 = np.array([np.cos(angle1), np.sin(angle1), sign * 0.5])
            p2 = np.array([np.cos(angle2), np.sin(angle2), sign * 0.5])
            mid = (p1 + p2) / 2.0
            pts.append(mid / np.linalg.norm(mid))
            
    return np.array(pts[:31])

def _build_cluster(n, sym_type, radii, kind):
    """
    Build a cluster of size n using symmetry generators and radial parameters.
    radii: array of shell radii for the distinct orbits.
    """
    if sym_type == 'Ih':
        gens = _icosahedral_generators()
        orbits = [1, 12, 30, 20] # Sizes of orbits: center, vertices, edges, faces
        # If n > 55, we need more shells. We'll approximate by adding more shells of 120 (or similar)
        # For now, handle up to 55 precisely, then add random-ish shells
        if n <= 55:
            pts = []
            idx = 0
            for i, size in enumerate(orbits):
                if idx >= len(radii):
                    break
                r = radii[i]
                for j in range(size):
                    if len(pts) < n:
                        pts.append(gens[idx] * r)
                    idx += 1
            return np.array(pts[:n])
        else:
            # Fallback for larger n: use first 55 and add more
            pts = []
            idx = 0
            for i, size in enumerate(orbits):
                if idx >= len(radii):
                    break
                r = radii[i]
                for j in range(size):
                    if len(pts) < n:
                        pts.append(gens[idx] * r)
                    idx += 1
            # Add remaining atoms as a spherical shell
            remaining = n - len(pts)
            if remaining > 0 and len(radii) > 4:
                r = radii[4]
                for i in range(remaining):
                    y = 1.0 - 2.0 * (i + 0.5) / remaining
                    rad = np.sqrt(1.0 - y*y)
                    theta = 2.399963229728653 * i
                    p = np.array([rad*np.cos(theta), rad*np.sin(theta), y]) * r
                    pts.append(p)
            return np.array(pts[:n])
            
    elif sym_type == 'Oh':
        gens = _octahedral_generators()
        orbits = [1, 6, 12, 8]
        if n <= 27:
            pts = []
            idx = 0
            for i, size in enumerate(orbits):
                if idx >= len(radii):
                    break
                r = radii[i]
                for j in range(size):
                    if len(pts) < n:
                        pts.append(gens[idx] * r)
                    idx += 1
            return np.array(pts[:n])
        else:
            # Fallback
            return _build_cluster(n, 'Ih', radii, kind) # Use Ih as better default for larger
            
    elif sym_type == 'D5h':
        gens = _decahedral_generators()
        orbits = [1, 10, 20]
        if n <= 31:
            pts = []
            idx = 0
            for i, size in enumerate(orbits):
                if idx >= len(radii):
                    break
                r = radii[i]
                for j in range(size):
                    if len(pts) < n:
                        pts.append(gens[idx] * r)
                    idx += 1
            return np.array(pts[:n])
        else:
            # Fallback
            return _build_cluster(n, 'Ih', radii, kind)
            
    else:
        return None

def _sym_energy_gradient(params, n, sym_type, kind):
    """
    Objective function for symmetry-constrained optimization.
    params: array of shell radii.
    """
    try:
        positions = _build_cluster(n, sym_type, params, kind)
        if positions is None or len(positions) != n:
            return 1e10, np.zeros_like(params)
        
        if not _is_valid(positions):
            return 1e10, np.zeros_like(params)
            
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        
        if not np.isfinite(energy):
            return 1e10, np.zeros_like(params)
            
        # Project gradient back to radial parameters
        # dE/dr_i = sum_{j in orbit i} dot(gradient_j, direction_j)
        # Since direction_j = position_j / r_i, and position_j = generator_j * r_i
        # dot(grad_j, pos_j/r_i) = dot(grad_j, generator_j) * r_i / r_i = dot(grad_j, generator_j) ?
        # Actually: pos = gen * r. dpos/dr = gen.
        # dE/dr = sum dot(grad, dpos/dr) = sum dot(grad, gen)
        
        # We need to map gradients back to radii
        # This requires knowing which generator corresponds to which atom.
        # For simplicity and speed, we will use finite differences for the gradient in the reduced space
        # if the analytic projection is complex. Given the small dimensionality, FD is acceptable for the outer loop.
        
        return energy, None # Use FD for gradient in minimize if we don't implement projection carefully
        
    except Exception:
        return 1e10, np.zeros_like(params)

def _optimize_symmetry(n, sym_type, init_radii, kind, deadline):
    """
    Optimize the shell radii for a given symmetry type.
    """
    # Determine number of shells needed
    if sym_type == 'Ih':
        n_shells = 5 if n > 55 else 4
    elif sym_type == 'Oh':
        n_shells = 5 if n > 27 else 4
    else:
        n_shells = 5 if n > 31 else 3
        
    # Initial radii guess
    if len(init_radii) < n_shells:
        init_radii = np.array(init_radii)
        while len(init_radii) < n_shells:
            init_radii = np.append(init_radii, init_radii[-1] * 1.2)
    else:
        init_radii = init_radii[:n_shells]
        
    # Bounds: radii must be positive
    bounds = [(0.5, 5.0) for _ in range(n_shells)]
    
    # Use Nelder-Mead or Powell as they don't require gradients and are robust to the non-smoothness
    # caused by the discrete nature of adding atoms in shells
    def objective(radii):
        e, _ = _sym_energy_gradient(radii, n, sym_type, kind)
        return e
        
    try:
        res = minimize(objective, init_radii, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200, 'adaptive': True},
                       bounds=bounds)
        if time.time() > deadline:
            return res.fun, res.x
            
        best_radii = res.x
        best_energy = res.fun
        
        # Polish with L-BFGS-B on full coordinates
        positions = _build_cluster(n, sym_type, best_radii, kind)
        if positions is not None and _is_valid(positions):
            try:
                polished_energy, polished_pos = relax(positions, 'sutton_chen', kind)
                if polished_energy < best_energy and _is_valid(polished_pos):
                    best_energy = polished_energy
                    # We return the polished positions, but we need to map back to radii?
                    # No, the caller will handle the final positions.
                    # We return energy and the positions directly.
                    return best_energy, polished_pos
            except Exception:
                pass
                
        return best_energy, positions
        
    except Exception:
        return 1e10, None

def _worker_task(args):
    """Worker function for parallel symmetry-optimized searches."""
    n, sym_type, init_radii, kind, deadline = args
    
    try:
        energy, positions = _optimize_symmetry(n, sym_type, init_radii, kind, deadline)
        if positions is not None and _is_valid(positions):
            return (energy, positions)
        return (np.inf, None)
    except Exception:
        return (np.inf, None)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    deadline = _START_TIME + _DEADLINE
    
    # Define initial radii guesses based on typical shell spacings
    # Typical nearest neighbor distance is ~1.4-1.5 in reduced units for SC
    base_r = 1.4
    
    initial_configs = [
        ('Ih', np.array([0.0, base_r, base_r*1.8, base_r*2.6, base_r*3.4])),
        ('Oh', np.array([0.0, base_r, base_r*1.7, base_r*2.4, base_r*3.1])),
        ('D5h', np.array([0.0, base_r, base_r*1.6, base_r*2.2, base_r*2.8]))
    ]
    
    # Add variations
    for i in range(3):
        sym, radii = initial_configs[i]
        # Perturb radii
        rng = np.random.default_rng(42 + i)
        perturbed_radii = radii * rng.uniform(0.8, 1.2, len(radii))
        initial_configs.append((sym, perturbed_radii))
        
    # Run parallel optimization
    tasks = []
    for sym, radii in initial_configs:
        tasks.append((n, sym, radii, kind, deadline))
        
    n_workers = min(4, len(tasks))
    
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_task, tasks)
            
        for energy, pos in results:
            if energy < best_energy and pos is not None and _is_valid(pos):
                best_energy = energy
                best_pos = pos
    except Exception:
        # Fallback: sequential
        for sym, radii in initial_configs[:3]:
            if _time_left() < 5.0:
                break
            try:
                energy, pos = _optimize_symmetry(n, sym, radii, kind, deadline)
                if energy < best_energy and pos is not None and _is_valid(pos):
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass

    # If symmetry-constrained search failed or didn't find a good result,
    # do a final quick BH from the best symmetry result or a generic FCC start
    if best_pos is None or best_energy > -1000: # Threshold for "bad"
        # Generate a generic FCC start
        max_r = int(np.ceil(n ** (1/3))) + 3
        candidates = []
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                for k in range(-max_r, max_r + 1):
                    if (i + j + k) % 2 == 0:
                        p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                        candidates.append((np.linalg.norm(p), p))
        candidates.sort(key=lambda x: x[0])
        pts = [p for _, p in candidates[:n]]
        fcc_pos = np.array(pts)
        
        if _is_valid(fcc_pos):
            try:
                e, pos = relax(fcc_pos, 'sutton_chen', kind)
                if e < best_energy and _is_valid(pos):
                    best_energy = e
                    best_pos = pos
            except Exception:
                pass
                
    if best_pos is None:
        # Last resort
        best_pos = np.zeros((n, 3))
        rng = np.random.default_rng(0)
        for i in range(n):
            while True:
                p = rng.uniform(-2, 2, 3)
                if i == 0 or np.all(np.linalg.norm(best_pos[:i] - p, axis=1) > 0.35):
                    best_pos[i] = p
                    break
                    
    return best_pos

# EVOLVE-BLOCK-END
