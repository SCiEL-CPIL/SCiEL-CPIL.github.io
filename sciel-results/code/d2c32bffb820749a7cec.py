import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges(verts):
    """
    Get the 30 edges of the icosahedron defined by the 12 vertices.
    Returns list of (i, j) index pairs.
    """
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist):
                edges.append((i, j))
    
    return edges

def _get_icosahedron_faces(verts, edges):
    """
    Get the 20 faces of the icosahedron.
    Returns list of (i, j, k) index triples.
    """
    faces = []
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    # For each pair of edges sharing a vertex, check if they form a face
    # An icosahedron face is a triangle where all three edges exist
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edge_set:
                continue
            for k in range(j+1, 12):
                if (i, k) in edge_set and (j, k) in edge_set:
                    faces.append((i, j, k))
    
    return faces

def _build_mackay_55():
    """
    Build the 55-atom Mackay icosahedron:
    - 1 center atom
    - 12 atoms on shell 1 (icosahedron vertices at r=1)
    - 42 atoms on shell 2 (12 vertices + 30 edge midpoints at r=1.87)
    """
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    v1 = _get_icosahedron_vertices(1.0)
    for v in v1:
        points.append(v)
    
    # Shell 2: 42 atoms
    base_verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges(base_verts)
    
    shell2_pts = []
    for v in base_verts:
        shell2_pts.append(v)
    
    for i, j in edges:
        mid = (base_verts[i] + base_verts[j]) / 2.0
        shell2_pts.append(mid)
    
    shell2_pts = np.array(shell2_pts)
    norms = np.linalg.norm(shell2_pts, axis=1)
    shell2_pts = shell2_pts / norms[:, np.newaxis] * 1.87
    
    for v in shell2_pts:
        points.append(v)
    
    return np.array(points)

def _relax_fragment(frag, potential='dzugutov'):
    """
    Relax a small fragment to its local minimum.
    """
    try:
        e, pos = relax(frag, potential, None)
        return pos
    except Exception:
        return frag

def _build_icosahedral_fragment(n_atoms, seed=0):
    """
    Build a relaxed icosahedral fragment with n_atoms atoms.
    Uses Mackay icosahedron as template, then relaxes.
    """
    if n_atoms == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    if n_atoms <= 13:
        # Center + icosahedron vertices
        points = [np.array([0.0, 0.0, 0.0])]
        v1 = _get_icosahedron_vertices(1.0)
        for v in v1[:n_atoms-1]:
            points.append(v)
        return np.array(points)
    
    if n_atoms <= 55:
        # Use Mackay 55 and take first n_atoms
        mackay = _build_mackay_55()
        return mackay[:n_atoms]
    
    # For n > 55, add atoms on a third shell
    mackay = _build_mackay_55()
    points = list(mackay)
    
    # Add atoms on a sphere of radius ~2.6
    rng = np.random.default_rng(seed)
    max_r = np.max(np.linalg.norm(mackay, axis=1))
    
    for _ in range(n_atoms - len(points)):
        added = False
        for _attempt in range(50):
            u, v = rng.uniform(0, 1), rng.uniform(0, 1)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            dir_vec = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            r_new = max_r + rng.uniform(0.3, 0.8)
            p_new = dir_vec * r_new
            
            dists = np.linalg.norm(np.array(points) - p_new, axis=1)
            if np.min(dists) > 1.2:
                points.append(p_new)
                max_r = np.max(np.linalg.norm(np.array(points), axis=1))
                added = True
                break
        if not added:
            points.append(mackay[0] + rng.normal(0, 0.5, 3))
    
    return np.array(points[:n_atoms])

def _merge_fragments(fragments, potential='dzugutov', deadline=None):
    """
    Merge pre-relaxed fragments into a single cluster using gradient-based optimization.
    """
    # Concatenate all fragments
    all_atoms = np.vstack(fragments)
    n_atoms = len(all_atoms)
    
    # Initial coarse relaxation
    try:
        e, pos = relax(all_atoms, potential, None)
        return pos
    except Exception:
        return all_atoms

def _build_cluster_from_fragments(n_atoms, potential='dzugutov', seed=0, deadline=None):
    """
    Build a cluster by decomposing n_atoms into icosahedral fragments and merging them.
    """
    rng = np.random.default_rng(seed)
    
    # Strategy: decompose n_atoms into fragments of size 13, 12, or smaller
    # 13 = center + icosahedron (12 vertices)
    # 12 = icosahedron only
    # 1 = single atom
    
    # Try different decompositions
    decompositions = []
    
    # All 13s
    n_13 = n_atoms // 13
    rem = n_atoms % 13
    if rem == 0:
        decompositions.append([13] * n_13)
    elif rem == 1:
        decompositions.append([13] * n_13 + [1])
    elif rem == 12:
        decompositions.append([13] * n_13 + [12])
    elif rem <= 12:
        decompositions.append([13] * n_13 + [rem])
    
    # Mix of 13s and 12s
    for n_13 in range(max(0, n_atoms // 13 - 2), min(n_atoms // 13 + 1, n_atoms // 12 + 1)):
        rem = n_atoms - n_13 * 13
        if rem < 0:
            continue
        n_12 = rem // 12
        rem2 = rem % 12
        if rem2 == 0:
            decompositions.append([13] * n_13 + [12] * n_12)
        elif rem2 == 1:
            decompositions.append([13] * n_13 + [12] * n_12 + [1])
        elif rem2 <= 12:
            decompositions.append([13] * n_13 + [12] * n_12 + [rem2])
    
    # Remove duplicates and keep unique
    unique_decomps = []
    for d in decompositions:
        d_sorted = tuple(sorted(d))
        if d_sorted not in unique_decomps:
            unique_decomps.append(d_sorted)
    
    # Try each decomposition
    best_energy = float('inf')
    best_positions = None
    
    for decomp in unique_decomps[:5]:  # Limit to 5 decompositions
        if deadline is not None and time.time() > deadline:
            break
        
        # Build fragments
        fragments = []
        for i, size in enumerate(decomp):
            if size <= 0:
                continue
            frag = _build_icosahedral_fragment(size, seed=seed + i)
            frag = _relax_fragment(frag, potential)
            fragments.append(frag)
        
        if not fragments:
            continue
        
        # Merge fragments
        merged = _merge_fragments(fragments, potential, deadline)
        
        # Final relaxation
        try:
            e, pos = relax(merged, potential, None)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            pass
    
    if best_positions is not None:
        return best_positions
    
    # Fallback: single fragment
    return _build_icosahedral_fragment(n_atoms, seed=seed)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # 10s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            # Use the provided relax function
            e, pos = relax(init_positions, potential, None)
            if e < best_energy and np.all(np.isfinite(pos)):
                best_energy = e
                best_positions = pos
        except Exception:
            pass

    # Strategy 1: Fragment-merger approach (primary)
    for seed in range(3):
        if time.time() > deadline:
            break
        try:
            cluster = _build_cluster_from_fragments(size, potential, seed=seed, deadline=deadline)
            if cluster is not None and len(cluster) == size:
                try_relax(cluster, label=f"fragment_{seed}")
        except Exception:
            pass
    
    # Strategy 2: Direct Mackay icosahedron
    try:
        mackay = _build_mackay_55()
        if size <= 55:
            try_relax(mackay[:size], label="mackay_direct")
        else:
            # Add extra atoms
            points = list(mackay)
            rng = np.random.default_rng(42)
            max_r = np.max(np.linalg.norm(mackay, axis=1))
            for _ in range(size - len(points)):
                added = False
                for _attempt in range(30):
                    u, v = rng.uniform(0, 1), rng.uniform(0, 1)
                    theta = 2 * np.pi * u
                    phi_angle = np.arccos(2 * v - 1)
                    dir_vec = np.array([
                        np.sin(phi_angle) * np.cos(theta),
                        np.sin(phi_angle) * np.sin(theta),
                        np.cos(phi_angle)
                    ])
                    r_new = max_r + rng.uniform(0.3, 0.7)
                    p_new = dir_vec * r_new
                    dists = np.linalg.norm(np.array(points) - p_new, axis=1)
                    if np.min(dists) > 1.0:
                        points.append(p_new)
                        max_r = np.max(np.linalg.norm(np.array(points), axis=1))
                        added = True
                        break
                if not added:
                    points.append(mackay[0] + rng.normal(0, 0.4, 3))
            try_relax(np.array(points[:size]), label="mackay_extended")
    except Exception:
        pass
    
    # Strategy 3: Perturbation on best found
    if best_positions is not None and time.time() < deadline - 30:
        for attempt in range(8):
            if time.time() > deadline:
                break
            try:
                rng = np.random.default_rng(1000 + attempt)
                amp = 0.08 + 0.03 * attempt
                perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                try_relax(perturbed, label=f"perturb_{attempt}")
            except Exception:
                pass
    
    # Final fallback
    if best_positions is None:
        try:
            best_positions = _build_icosahedral_fragment(size, seed=0)
        except Exception:
            # Absolute fallback: random compact
            rng = np.random.default_rng(3)
            radius = 1.5 * size ** (1.0/3.0)
            points = []
            for _ in range(size):
                while True:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) > radius:
                        continue
                    if points:
                        dists = np.linalg.norm(np.array(points) - p, axis=1)
                        if np.min(dists) < 0.8:
                            continue
                    points.append(p)
                    break
            best_positions = np.array(points)

    return best_positions
