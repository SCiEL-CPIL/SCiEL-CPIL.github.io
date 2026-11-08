"""
Strategy: Structure-Guided Discrete Topology Search.
Instead of random continuous displacement (standard Basin Hopping), we use:
1. Diverse physics-informed templates (FCC, HCP, BCC, Icosahedral).
2. Rigid-body rotations of the cluster to explore orientation-dependent minima.
3. Topology-preserving perturbations: small displacements constrained to maintain
   local neighbor topology (Voronoi cell integrity), avoiding high-energy stretched
   configurations that trap standard BH.
4. Parallel execution across 4 threads with adaptive kick scaling.
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    L = n ** (1.0/3.0) * spacing * 2.0
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    points.append(p)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        return _fcc_fragment(n, spacing, rng)

    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 2
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift = np.array([0, 0])
        else:
            shift = np.array([a/2, a*np.sqrt(3)/6])
            
        n_side = int(np.ceil(np.sqrt(n))) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift[0]
                y = j * a * np.sqrt(3) / 2.0 + shift[1]
                points.append([x, y, z])
                
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        return _hcp_fragment(n, spacing, rng)
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _bcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    points = []
    n_side = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                p1 = base
                p2 = base + np.array([0.5, 0.5, 0.5]) * a
                points.append(p1)
                points.append(p2)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        return _bcc_fragment(n, spacing, rng)
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Generate an icosahedral-like cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0]) * spacing * 1.5
    
    points = [np.zeros(3)]  # center
    points.extend([v for v in verts])
    
    # Add second shell if needed
    if n > 13:
        # Add atoms at face centers of icosahedron
        face_centers = []
        # Approximate by sampling points on sphere
        for _ in range(n - 13):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * spacing * 2.2
            face_centers.append(p)
        points.extend(face_centers)
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster (liquid-like)."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * n ** (1.0/3.0) * 1.0
    points = []
    max_attempts = n * 200
    attempts = 0
    min_dist = 0.5 * spacing
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
            
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _rotate_cluster(pos, rng):
    """Apply a random rigid-body rotation to the cluster."""
    # Random rotation matrix
    theta = rng.uniform(0, 2 * np.pi)
    axis = rng.normal(0, 1, 3)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    return R @ pos.T

def _topology_preserving_perturb(pos, scale, rng):
    """
    Perturb the cluster while trying to preserve local topology.
    We use smaller displacements and check that no bonds stretch too much.
    """
    n = pos.shape[0]
    new_pos = pos.copy()
    
    # Compute current neighbor distances
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Find nearest neighbors (topology)
    nn_idx = np.argmin(dists, axis=1)
    nn_dists = dists[np.arange(n), nn_idx]
    
    # Perturb with small displacements
    for i in range(n):
        # Displacement constrained by local environment
        local_scale = scale * 0.5  # Smaller than standard BH
        disp = rng.normal(0, local_scale, 3)
        
        # Check if displacement would stretch the nearest neighbor bond too much
        nn_i = nn_idx[i]
        current_d = np.linalg.norm(pos[i] - pos[nn_i])
        new_d = np.linalg.norm(pos[i] + disp - pos[nn_i])
        
        # Allow up to 20% stretch
        if new_d < current_d * 1.2:
            new_pos[i] = pos[i] + disp
        else:
            # Try smaller displacement
            new_pos[i] = pos[i] + disp * 0.5
    
    return new_pos

def _worker(task_id, initial_pos, potential, kind, rng_seed):
    """
    Worker function for parallel structure-guided search.
    Returns (energy, positions).
    """
    global _START_TIME
    rng = np.random.default_rng(rng_seed)
    current_pos = initial_pos.copy()
    current_energy = np.inf
    
    try:
        current_energy, current_pos = relax(current_pos, potential, kind)
    except Exception:
        return (np.inf, initial_pos)
    
    max_iter = 300
    for i in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Adaptive kick size: start larger, decrease over time
        base_scale = 0.3 + 0.5 * (1.0 - i / max_iter)
        
        # Alternate between rotation and topology-preserving perturbation
        if i % 3 == 0:
            # Rigid-body rotation
            new_pos = _rotate_cluster(current_pos, rng)
        elif i % 3 == 1:
            # Topology-preserving perturbation
            new_pos = _topology_preserving_perturb(current_pos, base_scale, rng)
        else:
            # Standard small perturbation
            noise = rng.normal(0, base_scale * 0.3, current_pos.shape)
            new_pos = current_pos + noise
        
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, kind)
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos_relaxed
        except Exception:
            continue
            
    # Final polish
    if _time_left() > 5.0:
        try:
            def fun(x):
                pos = x.reshape(-1, 3)
                e, g = sutton_chen_energy_gradient(pos, kind)
                return e, g.flatten()
            
            res = minimize(
                fun, 
                current_pos.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            if res.success and res.fun < current_energy:
                current_energy = res.fun
                current_pos = res.x.reshape(-1, 3)
        except Exception:
            pass
            
    return (current_energy, current_pos)

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    spacing = 1.0
    
    # Generate diverse initial templates
    rng_master = np.random.default_rng(size)
    
    templates = []
    
    # FCC
    try:
        fcc = _fcc_fragment(size, spacing, rng_master)
        if len(fcc) == size:
            templates.append(fcc)
    except Exception:
        pass
        
    # HCP
    try:
        hcp = _hcp_fragment(size, spacing, rng_master)
        if len(hcp) == size:
            templates.append(hcp)
    except Exception:
        pass
        
    # BCC
    try:
        bcc = _bcc_fragment(size, spacing, rng_master)
        if len(bcc) == size:
            templates.append(bcc)
    except Exception:
        pass
        
    # Icosahedral
    try:
        ico = _icosahedral_shell(size, spacing, rng_master)
        if len(ico) == size:
            templates.append(ico)
    except Exception:
        pass
        
    # Random (Multiple diverse starts)
    for _ in range(3):
        rand = _random_compact(size, spacing, rng_master)
        templates.append(rand)
        
    if not templates:
        # Fallback if all template generators fail
        templates.append(np.random.default_rng(size).uniform(-1, 1, (size, 3)))
        
    # Launch parallel workers
    # We have 4 threads available.
    num_workers = 4
    best_energy = np.inf
    best_positions = templates[0]
    
    # Create tasks: prioritize diverse templates
    tasks = []
    for i, t in enumerate(templates):
        seed = int(rng_master.integers(0, 2**31))
        tasks.append((i, t, seed))
    
    # Take first 4 most diverse templates
    active_tasks = tasks[:num_workers]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for task_id, pos, seed in active_tasks:
            future = executor.submit(_worker, task_id, pos, potential, kind, seed)
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception as e:
                pass
                
    # If time remains and we had more templates, try the next one
    if len(tasks) > num_workers and _time_left() > 30.0:
        task_id, pos, seed = tasks[num_workers]
        energy, positions = _worker(task_id, pos, potential, kind, seed)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
            
    return best_positions
