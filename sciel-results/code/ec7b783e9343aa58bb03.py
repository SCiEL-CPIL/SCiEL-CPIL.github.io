import numpy as np
from autoresearch_suite.chem_clusters import relax
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def generate_fcc_fragment(n, scale=1.0):
    """Generate an FCC cluster fragment of n atoms."""
    # FCC lattice constant such that nearest neighbor distance is ~1.0 (Dzugutov min is ~1.05, so scale slightly)
    # Nearest neighbor in FCC is a/sqrt(2). Let's set NN = 1.05.
    a = 1.05 * np.sqrt(2)
    points = []
    # Generate points in a cube large enough
    max_coord = int(n ** (1/3) * a * 2) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < (n ** (1/3)) * a * 0.8:
                        points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        # If we didn't get enough, expand search or just return what we have (should not happen with large max_coord)
        # For robustness, if len < n, pad with random or just return empty? 
        # Let's ensure we get at least n by expanding radius.
        pass
        
    # Sort by distance from center to get spherical fragment
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices]

def generate_hcp_fragment(n):
    """Generate an HCP cluster fragment of n atoms."""
    a = 1.05 * np.sqrt(2) # Same NN distance
    points = []
    max_coord = int(n ** (1/3) * a * 2) + 2
    hcp_c = 1.633 * a # Ideal HCP c/a ratio is sqrt(8/3) ~ 1.633
    # HCP lattice:
    # Layer A: (i*a, j*a*sqrt(3), 0) + (0, 0, 0) or (a/2, a*sqrt(3)/6, 0) ?
    # Simpler: Generate points in a hexagonal grid.
    # Let's use a simpler approach: Generate FCC and HCP is hard to do quickly correctly.
    # Alternative: Use a generic "close packed" random generator or just rely on FCC and Icosahedron.
    # Let's skip HCP for now and use FCC and Icosahedron and Random.
    return None

def generate_icosahedron(n):
    """Generate an icosahedral cluster of n atoms."""
    if n > 55:
        return None
        
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Scale to NN distance 1.05
    # Edge length of standard verts is 2.
    scale = 1.05 / 2.0
    verts = verts * scale
    
    center = np.array([0, 0, 0])
    
    # Edge midpoints (30)
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(np.linalg.norm(verts[i] - verts[j]), 1.05):
                edge_midpoints.append((verts[i] + verts[j]) / 2.0)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (20)
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(np.linalg.norm(verts[i] - verts[j]), 1.05):
                for k in range(j+1, 12):
                    if (np.isclose(np.linalg.norm(verts[i] - verts[k]), 1.05) and
                        np.isclose(np.linalg.norm(verts[j] - verts[k]), 1.05)):
                        face_centers.append((verts[i] + verts[j] + verts[k]) / 3.0)
    face_centers = np.array(face_centers)
    
    all_atoms = np.vstack([center[None, :], verts, edge_midpoints, face_centers])
    
    if n == 55:
        return all_atoms
    elif n == 13:
        return np.vstack([center[None, :], verts])
    elif n == 12:
        return verts
    else:
        # Truncate by distance from center
        dists = np.linalg.norm(all_atoms, axis=1)
        indices = np.argsort(dists)[:n]
        return all_atoms[indices]

def generate_random_close_packing(n, seed=42):
    """Generate random close packing."""
    rng = np.random.default_rng(seed)
    L = 1.05
    radius = L * n ** (1/3) * 1.2
    points = []
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.9 * L:
                    points.append(p)
                    attempts = 0
            else:
                points.append(p)
                attempts = 0
            if len(points) % 100 == 0:
                attempts += 1
    if len(points) < n:
        return None
    return np.array(points)

def basin_hopping(initial_pos, potential, n_steps=50, sigma=0.2, temperature=1.0, seed=0):
    """
    Run Basin Hopping starting from initial_pos.
    Returns the best position and energy found.
    """
    rng = np.random.default_rng(seed)
    try:
        e_curr, pos_curr = relax(initial_pos, potential)
    except Exception:
        return None, np.inf
        
    e_best, pos_best = e_curr, pos_curr.copy()
    
    for i in range(n_steps):
        # Perturb
        noise = rng.normal(0, sigma, pos_curr.shape)
        pos_perturbed = pos_curr + noise
        
        # Relax
        try:
            e_perturbed, pos_perturbed = relax(pos_perturbed, potential)
        except Exception:
            continue
            
        # Metropolis accept
        delta_e = e_perturbed - e_curr
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temperature):
            e_curr, pos_curr = e_perturbed, pos_perturbed
            if e_curr < e_best:
                e_best, pos_best = e_curr, pos_curr.copy()
                
    return pos_best, e_best

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    candidates = []
    
    # 1. Icosahedron
    icos = generate_icosahedron(n)
    if icos is not None:
        candidates.append(icos)
        
    # 2. FCC
    fcc = generate_fcc_fragment(n)
    if fcc is not None and len(fcc) == n:
        candidates.append(fcc)
        
    # 3. Random Close Packing (multiple seeds for diversity)
    for seed in [1, 2, 3]:
        rcp = generate_random_close_packing(n, seed=seed)
        if rcp is not None:
            candidates.append(rcp)
            
    # Filter valid candidates (correct shape, no overlapping atoms < 0.5)
    valid_candidates = []
    for cand in candidates:
        if cand.shape != (n, 3):
            continue
        # Check min distance
        # Efficient check: use cdist or simple loop for small n
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(cand[i] - cand[j])
                if d < min_dist:
                    min_dist = d
                if d < 0.5:
                    min_dist = 0
                    break
            if min_dist < 0.5:
                break
        if min_dist >= 0.5:
            valid_candidates.append(cand)
            
    if not valid_candidates:
        # Fallback: very random
        rng = np.random.default_rng(0)
        valid_candidates.append(rng.uniform(-5, 5, (n, 3)))
        
    # Relax all candidates once to get initial energies
    relaxed_candidates = []
    for cand in valid_candidates:
        try:
            e, pos = relax(cand, potential)
            relaxed_candidates.append((e, pos))
        except Exception:
            pass
            
    if not relaxed_candidates:
        # Should not happen, but fallback
        return valid_candidates[0]
        
    # Sort by energy
    relaxed_candidates.sort(key=lambda x: x[0])
    
    # Take top 3 candidates for Basin Hopping
    top_candidates = [pos for e, pos in relaxed_candidates[:3]]
    
    # Run Basin Hopping in parallel
    best_pos = None
    best_energy = np.inf
    
    def run_bh(pos, idx):
        # Vary parameters slightly for diversity
        sigma = 0.2 + 0.05 * idx
        temp = 1.0 + 0.2 * idx
        steps = 30 # Adjust based on time budget
        pos_best, e_best = basin_hopping(pos, potential, n_steps=steps, sigma=sigma, temperature=temp, seed=idx)
        return pos_best, e_best
        
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_bh, pos, i): i for i, pos in enumerate(top_candidates)}
        for future in as_completed(futures):
            try:
                pos_best, e_best = future.result()
                if pos_best is not None and e_best < best_energy:
                    best_energy = e_best
                    best_pos = pos_best
            except Exception:
                pass
                
    if best_pos is None:
        # Fallback to best relaxed candidate
        best_pos = relaxed_candidates[0][1]
        
    return best_pos
