# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _convex_hull_sites(points):
    """
    Generate candidate surface sites for adding new atoms.
    Returns a list of candidate positions (vertices, edge midpoints, face centroids).
    """
    from scipy.spatial import ConvexHull
    
    if len(points) < 4:
        # Degenerate cases
        if len(points) == 3:
            # Triangle: vertices and edge midpoints
            sites = list(points)
            for i in range(3):
                for j in range(i+1, 3):
                    mid = 0.5 * (points[i] + points[j])
                    sites.append(mid)
            return sites
        elif len(points) == 2:
            return list(points) + [0.5 * (points[0] + points[1])]
        else:
            return list(points)
        
    try:
        hull = ConvexHull(points)
        sites = []
        
        # 1. Vertices
        for v in hull.vertices:
            sites.append(points[v])
            
        # 2. Edge midpoints (from simplicial facets)
        seen_edges = set()
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    v1 = simplex[i]
                    v2 = simplex[j]
                    edge_key = tuple(sorted([v1, v2]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        mid = 0.5 * (points[v1] + points[v2])
                        sites.append(mid)
                        
        # 3. Face centroids (from simplicial facets)
        for simplex in hull.simplices:
            centroid = np.mean(points[simplex], axis=0)
            sites.append(centroid)
            
        return sites
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        return list(points)

def _add_atom_to_surface(core, rng, target_n):
    """
    Add one atom to the core at a promising surface site.
    """
    if len(core) >= target_n:
        return core
    
    sites = _convex_hull_sites(core)
    
    if not sites:
        # Fallback: random point near surface
        center = np.mean(core, axis=0)
        radius = np.max(np.linalg.norm(core - center, axis=1))
        new_pt = center + radius * 1.1 * rng.normal(0, 0.2, size=3)
        return np.vstack([core, new_pt])
    
    sites = np.array(sites)
    
    # Calculate distance from center to rank sites (prefer outer sites)
    center = np.mean(core, axis=0)
    dists_to_center = np.linalg.norm(sites - center, axis=1)
    
    # Calculate "crowding": min distance to existing atoms
    # We want sites that are not too close to existing atoms
    min_dists_to_core = []
    for s in sites:
        d = np.linalg.norm(core - s, axis=1)
        min_dists_to_core.append(d.min() if len(d) > 0 else np.inf)
    min_dists_to_core = np.array(min_dists_to_core)
    
    # Score: We want high distance from center (surface) and high min distance to core (not crowded)
    # Normalize
    max_center_dist = np.max(dists_to_center) if len(dists_to_center) > 0 else 1.0
    max_core_dist = np.max(min_dists_to_core) if len(min_dists_to_core) > 0 else 1.0
    
    scores = (dists_to_center / max_center_dist) * 0.5 + (min_dists_to_core / max_core_dist) * 0.5
    
    # Add some randomness to break symmetry
    scores += rng.normal(0, 0.1, size=len(scores))
    
    best_idx = np.argmax(scores)
    new_pt = sites[best_idx]
    
    return np.vstack([core, new_pt])

def _generate_hierarchical(n, rng, base_seed=0):
    """
    Generate a cluster by growing from a small core.
    """
    # Determine starting core size
    # Use known good small clusters as seeds if possible, or just start small
    # For n >= 13, start with 13 (Icosahedron-like) or 12 (Cuboctahedron-like)
    # We will generate a few variants
    
    # Variant 1: Start with 13 (Icosahedral shell)
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Add center
    ico_pts = np.vstack([ico_pts, np.array([0, 0, 0])])
    # Normalize to roughly unit radius
    ico_pts /= np.max(np.linalg.norm(ico_pts, axis=1))
    
    if n <= 13:
        # If n is small, we can't just use 13. 
        # For n < 13, we'll just use random or simple structures.
        # But the task asks for n >= 38. So we assume n > 13.
        pass
        
    core = ico_pts.copy()
    
    # If n is less than len(core), we need to trim. 
    # But n is at least 38, so core (13) is smaller.
    
    while len(core) < n:
        core = _add_atom_to_surface(core, rng, n)
        
    # Add small noise to break perfect symmetry
    core = core + rng.normal(0, 0.02, size=core.shape)
    
    return core

def _generate_hierarchical_fcc(n, rng):
    """
    Alternative: Start with an FCC-like core and grow.
    """
    # Small FCC cluster (e.g., 12 atoms: cuboctahedron)
    # Vertices of cuboctahedron
    pts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if abs(i) + abs(j) + abs(k) == 1: # This is not cuboctahedron
                    pass
    # Let's just use a standard FCC truncation for n=12 or 13 as a seed
    # Or simpler: use the 13-atom icosahedron but perturb it more towards FCC?
    # Actually, let's just use a different seed: Random packing of 10 atoms, then grow.
    
    # Generate 10 random atoms
    r_est = 1.5
    core = []
    attempts = 0
    while len(core) < 10 and attempts < 1000:
        attempts += 1
        p = rng.uniform(-r_est, r_est, 3)
        if np.linalg.norm(p) > r_est:
            continue
        if core:
            d = np.linalg.norm(np.array(core) - p, axis=1)
            if d.min() < 0.4:
                continue
        core.append(p)
    core = np.array(core)
    
    while len(core) < n:
        core = _add_atom_to_surface(core, rng, n)
        
    core = core + rng.normal(0, 0.05, size=core.shape)
    return core

def _relax_worker(args):
    """
    Worker function for parallel relaxation.
    """
    x, kind, n, seed_val = args
    
    try:
        e, p = relax(x, 'sutton_chen', kind)
        
        # Validate
        if not np.all(np.isfinite(p)) or not np.isfinite(e):
            return (np.inf, None)
            
        # Check min distance constraint
        if n > 1:
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return (np.inf, None)
                
        return (e, p)
    except Exception:
        return (np.inf, None)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    base_seed = 42 ^ (n * 101)
    
    tasks = []
    
    # Strategy 1: Icosahedral Growth (High symmetry, dense)
    rng1 = np.random.default_rng(base_seed)
    pos1 = _generate_hierarchical(n, rng1, base_seed)
    tasks.append((pos1, kind, n, base_seed))
    
    # Strategy 2: Random Core Growth (More diverse surface)
    rng2 = np.random.default_rng(base_seed + 1)
    pos2 = _generate_hierarchical_fcc(n, rng2)
    tasks.append((pos2, kind, n, base_seed + 1))
    
    # Strategy 3: Icosahedral Growth with higher noise
    rng3 = np.random.default_rng(base_seed + 2)
    pos3 = _generate_hierarchical(n, rng3, base_seed + 2)
    pos3 += rng3.normal(0, 0.1, size=pos3.shape)
    tasks.append((pos3, kind, n, base_seed + 2))
    
    # Strategy 4: Random Core Growth with higher noise
    rng4 = np.random.default_rng(base_seed + 3)
    pos4 = _generate_hierarchical_fcc(n, rng4)
    pos4 += rng4.normal(0, 0.1, size=pos4.shape)
    tasks.append((pos4, kind, n, base_seed + 3))
    
    # Strategy 5: BCC Truncation (Fallback/Comparison)
    # Reusing a simple BCC generator from previous logic, just in case
    def _simple_bcc(n, rng):
        r_est = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
        max_idx = int(np.ceil(r_est)) + 2
        points = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                for k in range(-max_idx, max_idx + 1):
                    p1 = np.array([i, j, k])
                    if np.linalg.norm(p1) <= r_est:
                        points.append(p1)
                    p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                    if np.linalg.norm(p2) <= r_est:
                        points.append(p2)
        if not points:
            return rng.uniform(-1, 1, size=(n, 3))
        points = np.array(points)
        points -= points.mean(axis=0)
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        if len(sorted_idx) < n:
            extra = n - len(sorted_idx)
            rand_pts = rng.uniform(-r_est, r_est, size=(extra, 3))
            points = np.vstack([points, rand_pts])
            dists = np.linalg.norm(points, axis=1)
            sorted_idx = np.argsort(dists)
        return points[sorted_idx[:n]]
        
    rng5 = np.random.default_rng(base_seed + 4)
    pos5 = _simple_bcc(n, rng5)
    pos5 += rng5.normal(0, 0.05, size=pos5.shape)
    tasks.append((pos5, kind, n, base_seed + 4))
    
    best_e = np.inf
    best_p = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for t in tasks:
            if time.time() > deadline - 10.0:
                break
            futures.append(executor.submit(_relax_worker, t))
            
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 5.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
                
    # Refine best solution if time permits
    if best_p is not None and time.time() < deadline - 20.0:
        new_tasks = []
        for i in range(4):
            rng_ref = np.random.default_rng(base_seed + 100 + i)
            noise = 0.05 + 0.05 * i
            perturbed = best_p + rng_ref.normal(0, noise, size=best_p.shape)
            new_tasks.append((perturbed, kind, n, base_seed + 100 + i))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_relax_worker, t) for t in new_tasks]
            for future in concurrent.futures.as_completed(futures):
                if time.time() > deadline - 5.0:
                    break
                try:
                    e, p = future.result()
                    if e < best_e:
                        best_e = e
                        best_p = p
                except Exception:
                    pass
                    
    if best_p is None:
        # Fallback
        best_p = np.zeros((n, 3))
        for i in range(n):
            best_p[i] = [ (i % 10) * 0.5, ((i // 10) % 10) * 0.5, (i // 100) * 0.5 ]
            
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
