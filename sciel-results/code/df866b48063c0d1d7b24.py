"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_icosahedron(n):
    """Generate a cluster starting with a central icosahedron and adding atoms."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (scaled to unit sphere roughly, or specific distance)
    # Standard icosahedron vertices:
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)] # Center
    points.extend(verts.tolist()) # 12 vertices, total 13
    
    if n <= 13:
        # If n < 13, we just return a subset? No, task n >= 38.
        # But for robustness, if n < 13, just take first n.
        return np.array(points[:n])

    # We need n-13 more atoms.
    # The second shell of an icosahedron has 30 positions (centers of edges/faces).
    # Let's generate candidate positions for the 2nd shell.
    # Edges of icosahedron: 30 edges.
    # Faces: 20 faces.
    # A common structure is the Mackay icosahedron (13+30+55).
    # We will try to place atoms near the midpoints of edges or face centers.
    
    # Generate edge midpoints
    # Get all pairs of vertices that are connected.
    # Distances between vertices in unit icosahedron:
    # Edge length = sqrt( (1-0)^2 + (phi-1)^2 )? No.
    # Let's just compute distances.
    d_min = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < d_min - 1e-5:
                d_min = d
    # Actually, in a unit icosahedron, the shortest distance is the edge.
    # Let's just use a simple radial expansion.
    # Place new atoms at radius R2.
    R1 = 1.0
    R2 = np.sqrt(2) # Rough estimate for 2nd shell
    
    candidates = []
    # Sample points on a sphere of radius R2
    rng = np.random.default_rng(42 + n)
    for _ in range(200):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * R2
        candidates.append(p)
        
    # Also try to place atoms in specific "gap" locations if we can identify them easily.
    # For simplicity, we will just do a greedy packing from the current 13 atoms.
    
    current_points = np.array(points)
    min_dist = 0.8 # Slightly larger than 0.5 to allow relax
    
    while len(current_points) < n:
        # Generate a batch of random candidates in a shell
        batch_size = 50
        r_min = R2 * 0.9
        r_max = R2 * 1.5
        batch = []
        for _ in range(batch_size):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            r = rng.uniform(r_min, r_max)
            p = v * r
            batch.append(p)
        
        # Try to add as many as possible
        added = 0
        for p in batch:
            if len(current_points) >= n:
                break
            # Check distance against all current points
            dists = np.linalg.norm(current_points - p, axis=1)
            if np.min(dists) >= min_dist:
                current_points = np.vstack([current_points, [p]])
                added += 1
                
        if added == 0:
            # If stuck, expand search radius
            r_min = r_max
            r_max *= 1.1
            # Add some noise to current points to help? No, keep it simple.
            # Just increase radius.
            if r_max > 10:
                # Fallback: just add random points in a large sphere
                for _ in range(n - len(current_points)):
                    v = rng.normal(size=3)
                    v /= np.linalg.norm(v)
                    r = rng.uniform(0, 5)
                    p = v * r
                    current_points = np.vstack([current_points, [p]])
                break

    return current_points[:n]

def _generate_fcc(n):
    """Generate a truncated FCC cluster."""
    rng = np.random.default_rng(123 + n)
    points = []
    # FCC lattice basis vectors
    a = 1.0
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]
    
    # Generate points in a grid
    N = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + np.array(b)
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    # Select closest n points
    points = candidates[:n]
    
    # Center it
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points

def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.1 * n ** (1.0 / 3.0)
    min_dist = 0.8
    attempts = 0
    max_attempts = n * 50
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    # If we didn't get enough, just fill with random points (might overlap, relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    """
    Direct approach: Generate diverse structural templates (Icosahedral, FCC, Random)
    and relax each. Use basin hopping for refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    timeout = 295.0 # Safety margin before 330s kill
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedral template
    try:
        p_ico = _generate_icosahedron(size)
        candidates.append(p_ico)
    except Exception:
        pass
        
    # 2. FCC template
    try:
        p_fcc = _generate_fcc(size)
        candidates.append(p_fcc)
    except Exception:
        pass
        
    # 3. Random packings (a few seeds)
    for seed in [1, 2, 3]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append(p_rand)
        except Exception:
            pass

    best_energy = np.inf
    best_pos = None
    
    # Relax all candidates
    for i, pos in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            continue
            
    # Refinement: Basin Hopping style
    # Take the best structure, perturb, and relax again.
    if best_pos is not None:
        rng = np.random.default_rng(size + 100)
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        # Number of refinement steps depends on time left
        # Estimate time per relax: ~2s? Let's assume we have ~40s left.
        # Do 10-15 steps.
        max_refine_steps = 15
        
        for step in range(max_refine_steps):
            if time.time() - start_time > timeout - 5:
                break
            
            # Perturbation
            sigma = 0.2 * np.exp(-step * 0.1) # Cooling
            noise = rng.normal(0, sigma, size=current_pos.shape)
            perturbed_pos = current_pos + noise
            
            try:
                energy, new_pos = relax(perturbed_pos, potential, parameter)
                # Accept if better (or always accept for exploration? 
                # Since we want global min, and relax is deterministic local min,
                # we should keep the best found so far).
                if energy < current_energy:
                    current_energy = energy
                    current_pos = new_pos
                    best_energy = current_energy
                    best_pos = current_pos
            except Exception:
                continue
                
    if best_pos is None:
        # Fallback: Return one of the generated candidates if relax failed entirely
        if candidates:
            return candidates[0]
        else:
            # Last resort: linear chain (invalid for energy, but valid geometry?)
            # Task says "atoms must be at least 0.5 apart".
            return np.array([np.array([i, 0, 0]) for i in range(size)])

    return best_pos
# EVOLVE-BLOCK-END
