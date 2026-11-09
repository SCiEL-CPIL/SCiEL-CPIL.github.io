"""
Sutton-Chen cluster optimization using simulated annealing + local relaxation.
Structural change: replace pure portfolio approach with SA-guided global search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard: 330s limit, aim for 280s to be safe
DEADLINE = time.time() + 280

def _generate_icosahedron_vertices(radius=1.0):
    """Generate coordinates for a regular icosahedron with 'radius'. Returns 12 vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norm = np.linalg.norm(verts, axis=1)
    scale = radius / norm
    verts = verts * scale[:, np.newaxis]
    return verts

def _generate_mackay_icosahedron(n_atoms, spacing=0.72):
    """Generate a Mackay icosahedral cluster."""
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    points = [np.zeros(3)]
    
    if n_atoms > 1:
        verts1 = _generate_icosahedron_vertices(radius=1.0)
        points.extend(verts1.tolist())
        
    if n_atoms > 13:
        scale2 = 1.701
        verts2 = _generate_icosahedron_vertices(radius=scale2)
        points.extend(verts2.tolist())
        
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3:
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale2
                    edges.append(mid)
        points.extend(edges)
        
    if len(points) < n_atoms:
        scale4 = 2.571
        verts4 = _generate_icosahedron_vertices(radius=scale4)
        points.extend(verts4.tolist())
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3:
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale4
                    points.append(mid)
        if len(points) < n_atoms:
            scale5 = 3.0
            verts5 = _generate_icosahedron_vertices(radius=scale5)
            points.extend(verts5.tolist())
            
    points = np.array(points)
    points = points - points[:min(len(points), n_atoms)].mean(axis=0)
    points = points * spacing
    return points[:n_atoms]

def _generate_fcc_shells(n_atoms, spacing=0.72):
    """Generate a compact FCC-like cluster."""
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0))) + 3
    coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    coords.append((i, j, k))
    
    if not coords:
        coords = [(0,0,0)]
        
    coords = np.array(coords, dtype=float)
    dists = np.sum(coords**2, axis=1)
    sorted_idx = np.argsort(dists)
    selected = coords[sorted_idx[:n_atoms]]
    
    center = np.mean(selected, axis=0)
    selected = selected - center
    selected = selected * spacing
    return selected

def _generate_hcp_shells(n_atoms, spacing=0.72):
    """Generate a compact HCP-like cluster."""
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    sites = []
    a = 1.0
    c = 1.633 * a
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 2
    
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3) / 2.0
                sites.append([x, y, z])
    
    sites = np.array(sites)
    sites = np.unique(np.round(sites, 10), axis=0)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    sites = sites[idx]
    sites = sites - sites[:n_atoms].mean(axis=0)
    sites = sites[:n_atoms] * spacing
    return sites

def _random_cluster(n, seed, min_dist=0.6, spacing=0.72):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = spacing * n ** (1.0 / 3.0) * 1.5
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        points.append(p)
    return np.array(points)

def _simulated_annealing(start_pos, n, kind, potential, t_start, max_time=60.0):
    """
    Simulated annealing with local relaxation.
    Returns (energy, positions) of best found.
    """
    rng = np.random.default_rng(42)
    
    # Initial temperature and cooling schedule
    T0 = 5.0
    Tend = 0.01
    n_iter = 200
    
    current_pos = start_pos.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for it in range(n_iter):
        # Check time
        if time.time() - t_start > max_time:
            break
        
        # Generate perturbation
        step_size = T0 * 0.1
        perturbation = rng.normal(0, step_size, size=(n, 3))
        new_pos = current_pos + perturbation
        
        # Check validity
        if not np.all(np.isfinite(new_pos)):
            continue
        
        from scipy.spatial.distance import pdist
        min_d = pdist(new_pos, metric='euclidean').min()
        if min_d < 0.35:
            continue
        
        new_energy, _ = sutton_chen_energy_gradient(new_pos, kind)
        
        # Accept/reject
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            delta_E = new_energy - current_energy
            T = T0 * (Tend / T0) ** (it / n_iter)
            if rng.random() < np.exp(-delta_E / T):
                current_pos = new_pos
                current_energy = new_energy
        
        # Periodic local relaxation
        if it % 20 == 19:
            try:
                e_relax, pos_relax = relax(current_pos, potential, kind)
                if np.all(np.isfinite(pos_relax)):
                    min_d = pdist(pos_relax, metric='euclidean').min()
                    if min_d >= 0.35:
                        current_pos = pos_relax
                        current_energy = e_relax
                        if e_relax < best_energy:
                            best_energy = e_relax
                            best_pos = pos_relax.copy()
            except Exception:
                pass
    
    return best_energy, best_pos

def construct(instance):
    """
    SA-guided global search with local relaxation.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    t_start = time.time()
    spacing = 0.72
    
    # Generate diverse starting points
    start_configs = []
    
    # Structured starts
    for gen_func in [_generate_mackay_icosahedron, _generate_fcc_shells, _generate_hcp_shells]:
        try:
            cfg = gen_func(size, spacing)
            if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
                from scipy.spatial.distance import pdist
                min_d = pdist(cfg, metric='euclidean').min()
                if min_d >= 0.35:
                    start_configs.append(cfg)
        except Exception:
            pass
    
    # Random starts
    for seed in [size, size + 100, size + 200]:
        try:
            cfg = _random_cluster(size, seed, min_dist=0.6, spacing=spacing)
            if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
                from scipy.spatial.distance import pdist
                min_d = pdist(cfg, metric='euclidean').min()
                if min_d >= 0.35:
                    start_configs.append(cfg)
        except Exception:
            pass
    
    # Remove duplicates
    unique_configs = []
    for cfg in start_configs:
        is_dup = False
        for ucfg in unique_configs:
            if np.allclose(cfg, ucfg, atol=0.1):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
    
    start_configs = unique_configs
    
    best_energy = np.inf
    best_positions = None
    
    # For each start, do SA + local relaxation
    for i, cfg in enumerate(start_configs):
        elapsed = time.time() - t_start
        remaining = 270.0 - elapsed
        if remaining < 10.0:
            break
        
        # Allocate time to each SA run
        time_per_sa = min(60.0, remaining / max(1, len(start_configs) - i))
        
        try:
            e_sa, pos_sa = _simulated_annealing(cfg, size, kind, potential, t_start, max_time=time_per_sa)
            
            if np.all(np.isfinite(pos_sa)):
                from scipy.spatial.distance import pdist
                min_d = pdist(pos_sa, metric='euclidean').min()
                if min_d >= 0.35:
                    # Final local relaxation
                    try:
                        e_final, pos_final = relax(pos_sa, potential, kind)
                        if np.all(np.isfinite(pos_final)):
                            min_d2 = pdist(pos_final, metric='euclidean').min()
                            if min_d2 >= 0.35 and e_final < best_energy:
                                best_energy = e_final
                                best_positions = pos_final.copy()
                    except Exception:
                        if e_sa < best_energy:
                            best_energy = e_sa
                            best_positions = pos_sa.copy()
        except Exception:
            continue
    
    # If no valid result, return a simple valid configuration
    if best_positions is None:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
        
    return best_positions
# EVOLVE-BLOCK-END
