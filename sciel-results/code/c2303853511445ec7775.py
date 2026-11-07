"""Structure prediction via growth from known small-cluster minima."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

def _get_small_minima(n, kind):
    """
    Return approximate coordinates for known small-cluster minima.
    Uses icosahedral/Mackay-like structures as starting points.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n == 2:
        return np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    if n == 3:
        # Triangle
        r = 0.7
        return np.array([
            [r, 0, 0],
            [-r/2, r*np.sqrt(3)/2, 0],
            [-r/2, -r*np.sqrt(3)/2, 0]
        ])
    if n == 4:
        # Tetrahedron
        r = 0.7
        return np.array([
            [r, r, r],
            [r, -r, -r],
            [-r, r, -r],
            [-r, -r, r]
        ]) / np.linalg.norm([r, r, r]) * r
    if n == 5:
        # Square pyramid
        r = 0.7
        return np.array([
            [r, 0, 0], [0, r, 0], [-r, 0, 0], [0, -r, 0],
            [0, 0, r*0.8]
        ])
    if n == 6:
        # Octahedron
        r = 0.7
        return np.array([
            [r, 0, 0], [-r, 0, 0],
            [0, r, 0], [0, -r, 0],
            [0, 0, r], [0, 0, -r]
        ])
    if n == 7:
        # Octahedron + center
        r = 0.7
        pts = _get_small_minima(6, kind).tolist()
        pts.append([0.0, 0.0, 0.0])
        return np.array(pts)
    if n == 8:
        # Cube
        r = 0.6
        return np.array([
            [r,r,r], [r,r,-r], [r,-r,r], [r,-r,-r],
            [-r,r,r], [-r,r,-r], [-r,-r,r], [-r,-r,-r]
        ])
    if n == 9:
        # Square antiprism or cube+center
        r = 0.65
        pts = _get_small_minima(8, kind).tolist()
        pts.append([0.0, 0.0, 0.0])
        return np.array(pts)
    if n == 10:
        # Two squares offset
        r = 0.65
        return np.array([
            [r,0,0], [0,r,0], [-r,0,0], [0,-r,0],
            [r,0,0.5], [0,r,0.5], [-r,0,0.5], [0,-r,0.5],
            [0,0,0], [0,0,0.5]
        ])
    if n == 11:
        # Icosahedron subset
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts[0]) * 0.7
        return verts[:11]
    if n == 12:
        # Icosahedron
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts[0]) * 0.7
        return verts
    if n == 13:
        # Icosahedron + center
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts[0]) * 0.7
        pts = verts.tolist()
        pts.append([0.0, 0.0, 0.0])
        return np.array(pts)
    
    # For larger n, use growth from n=13 core
    return None

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points roughly uniformly on a sphere of given radius."""
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = golden_angle * i
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(points) * radius

def _find_surface_sites(core_positions, n_new, kind):
    """
    Find good positions for n_new new atoms on the surface of the core.
    Uses a combination of:
    1. Fibonacci sphere distribution
    2. Edge/facet centers of the core's convex hull (approximated)
    3. Random perturbations
    """
    if len(core_positions) == 0:
        return _fibonacci_sphere(n_new, radius=0.8)
    
    # Estimate core radius
    core_center = np.mean(core_positions, axis=0)
    core_radii = np.linalg.norm(core_positions - core_center, axis=1)
    r_core = np.mean(core_radii) * 1.1  # Slightly larger than mean radius
    
    new_positions = []
    
    # Strategy 1: Fibonacci sphere at appropriate radius
    fib_pts = _fibonacci_sphere(n_new, radius=r_core)
    for p in fib_pts:
        new_positions.append(p + core_center)
    
    # Strategy 2: Add points at edge midpoints (for icosahedral cores)
    if len(core_positions) >= 12:
        # Find pairs of core atoms that are close (likely bonded)
        core_n = len(core_positions)
        min_dist = np.inf
        for i in range(core_n):
            for j in range(i + 1, core_n):
                d = np.linalg.norm(core_positions[i] - core_positions[j])
                min_dist = min(min_dist, d)
        
        # Edge midpoints
        edge_mids = []
        for i in range(core_n):
            for j in range(i + 1, core_n):
                d = np.linalg.norm(core_positions[i] - core_positions[j])
                if d < min_dist * 1.3:
                    mid = (core_positions[i] + core_positions[j]) / 2
                    # Push outward
                    direction = mid - core_center
                    if np.linalg.norm(direction) > 0.01:
                        direction = direction / np.linalg.norm(direction)
                        edge_mids.append(mid + direction * 0.15)
        
        # Use some edge midpoints
        if len(edge_mids) > 0 and n_new > 0:
            # Take up to n_new//3 edge midpoints
            n_edge = min(len(edge_mids), max(1, n_new // 3))
            rng_edge = np.random.default_rng(42)
            edge_indices = rng_edge.choice(len(edge_mids), size=n_edge, replace=False)
            for idx in edge_indices:
                new_positions.append(edge_mids[idx])
    
    # Strategy 3: Add some random surface points
    n_random = max(1, n_new // 4)
    rng_rand = np.random.default_rng(123)
    for _ in range(n_random):
        # Random direction
        dir_vec = rng_rand.normal(size=3)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        # Random radius near surface
        r = r_core * (0.95 + 0.1 * rng_rand.random())
        new_positions.append(core_center + dir_vec * r)
    
    # Trim or pad to n_new
    if len(new_positions) > n_new:
        # Keep first n_new
        new_positions = new_positions[:n_new]
    elif len(new_positions) < n_new:
        # Pad with more Fibonacci points
        extra_needed = n_new - len(new_positions)
        extra_fib = _fibonacci_sphere(extra_needed, radius=r_core * 1.05)
        for p in extra_fib:
            new_positions.append(p + core_center)
    
    return np.array(new_positions[:n_new])

def _grow_cluster(core_positions, n_total, kind, rng):
    """
    Grow a cluster from a core to n_total atoms by adding atoms one at a time
    at energetically favorable surface sites.
    """
    if n_total <= len(core_positions):
        return core_positions[:n_total]
    
    positions = core_positions.tolist()
    n_core = len(positions)
    
    for i in range(n_total - n_core):
        # Find where to add the next atom
        # Use a simple heuristic: try several candidate positions and pick the one
        # that minimizes local energy contribution
        
        # Candidate 1: Fibonacci sphere position
        fib_pts = _fibonacci_sphere(1, radius=0.8 * (len(positions) ** (1/3.0)) * 0.7)
        candidates = [fib_pts[0]]
        
        # Candidate 2: Near existing atoms (surface sites)
        if len(positions) > 0:
            center = np.mean(positions, axis=0)
            radii = np.linalg.norm(np.array(positions) - center, axis=1)
            r_avg = np.mean(radii)
            
            # Try positions near each of the last few atoms
            for j in range(max(0, len(positions) - 5), len(positions)):
                atom_pos = np.array(positions[j])
                # Try several directions from this atom
                for angle in np.linspace(0, 2*np.pi, 4):
                    for elevation in [0.3, 0.6, 0.9]:
                        dir_vec = np.array([np.cos(angle), np.sin(angle) * np.sin(elevation), np.cos(elevation)])
                        dir_vec = dir_vec / np.linalg.norm(dir_vec)
                        candidates.append(atom_pos + dir_vec * 0.65)
        
        # Candidate 3: Random
        if len(positions) > 0:
            center = np.mean(positions, axis=0)
            r_avg = np.mean(np.linalg.norm(np.array(positions) - center, axis=1))
            for _ in range(5):
                dir_vec = rng.normal(size=3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                r = r_avg * (0.9 + 0.2 * rng.random())
                candidates.append(center + dir_vec * r)
        
        # Evaluate candidates by simple energy heuristic
        best_candidate = None
        best_score = np.inf
        
        for cand in candidates:
            cand = np.array(cand)
            # Check minimum distance to existing atoms
            if len(positions) > 0:
                dists = np.linalg.norm(np.array(positions) - cand, axis=1)
                min_d = np.min(dists)
                if min_d < 0.4:
                    continue  # Too close
            
            # Simple score: prefer positions that are not too far and have good spacing
            if len(positions) > 0:
                center = np.mean(positions, axis=0)
                dist_from_center = np.linalg.norm(cand - center)
                avg_radius = np.mean(np.linalg.norm(np.array(positions) - center, axis=1))
                # Prefer being near the average radius
                score = abs(dist_from_center - avg_radius) * 2 + 0.1 * dist_from_center
            else:
                score = 0.0
            
            if score < best_score:
                best_score = score
                best_candidate = cand
        
        if best_candidate is None:
            # Fallback: add at a random surface position
            if len(positions) > 0:
                center = np.mean(positions, axis=0)
                r_avg = np.mean(np.linalg.norm(np.array(positions) - center, axis=1))
                dir_vec = rng.normal(size=3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                best_candidate = center + dir_vec * r_avg * 1.05
            else:
                best_candidate = np.array([0.0, 0.0, 0.0])
        
        positions.append(best_candidate)
    
    return np.array(positions)

def _generate_initial_positions(n, kind, rng):
    """Generate multiple diverse initial configurations using growth strategy."""
    configs = []
    
    # Strategy 1: Grow from n=13 icosahedral core
    if n >= 13:
        try:
            core13 = _get_small_minima(13, kind)
            if core13 is not None:
                grown = _grow_cluster(core13, n, kind, rng)
                if len(grown) == n and np.all(np.isfinite(grown)):
                    configs.append(grown)
        except:
            pass
    
    # Strategy 2: Grow from n=12 icosahedral core
    if n >= 12:
        try:
            core12 = _get_small_minima(12, kind)
            if core12 is not None:
                grown = _grow_cluster(core12, n, kind, rng)
                if len(grown) == n and np.all(np.isfinite(grown)):
                    configs.append(grown)
        except:
            pass
    
    # Strategy 3: Grow from n=10 core
    if n >= 10:
        try:
            core10 = _get_small_minima(10, kind)
            if core10 is not None:
                grown = _grow_cluster(core10, n, kind, rng)
                if len(grown) == n and np.all(np.isfinite(grown)):
                    configs.append(grown)
        except:
            pass
    
    # Strategy 4: Two-layer icosahedral (Mackay-like)
    if n >= 13:
        try:
            phi = (1 + np.sqrt(5)) / 2
            # Inner shell: 13 atoms (icosahedron + center)
            inner = _get_small_minima(13, kind)
            if inner is not None:
                # Outer shell: add atoms on a larger sphere
                n_outer = n - 13
                if n_outer > 0:
                    outer_r = 0.7 * 1.8  # Larger radius
                    outer_pts = _fibonacci_sphere(n_outer, radius=outer_r)
                    config = np.vstack([inner, outer_pts])
                    if len(config) == n and np.all(np.isfinite(config)):
                        configs.append(config)
        except:
            pass
    
    # Strategy 5: Random compact (fallback diversity)
    radius = 0.75 * n ** (1.0 / 3.0) * 1.1
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 80:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.38 for q in points):
                points.append(p)
        attempts += 1
    if len(points) == n:
        configs.append(np.array(points))
    
    # Strategy 6: Another random with different parameters
    radius2 = 0.72 * n ** (1.0 / 3.0) * 1.0
    points2 = []
    attempts2 = 0
    while len(points2) < n and attempts2 < n * 80:
        p = rng.uniform(-radius2, radius2, 3)
        if np.linalg.norm(p) <= radius2:
            if all(np.linalg.norm(p - q) >= 0.36 for q in points2):
                points2.append(p)
        attempts2 += 1
    if len(points2) == n:
        configs.append(np.array(points2))
    
    return configs

def _validate_positions(positions, min_dist=0.35):
    """Check if positions are valid (finite, no close pairs)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True

def construct(instance):
    """Grow clusters from known small-cluster minima, relax, and return best."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    rng = np.random.default_rng(size * 7 + 123)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    configs = _generate_initial_positions(size, kind, rng)
    
    # Add a few more random configurations for diversity
    for trial in range(2):
        radius = 0.73 * size ** (1.0 / 3.0) * (1.0 + 0.15 * trial)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 100:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            configs.append(np.array(points))
    
    # Relax each configuration
    for config in configs:
        if config is None or len(config) != size:
            continue
        if not _validate_positions(config):
            continue
        
        try:
            energy, positions = relax(config, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                if _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    # If no valid result, try a simpler approach
    if best_positions is None:
        # Use a simple spherical arrangement
        fib_pts = _fibonacci_sphere(size, radius=0.8 * size ** (1/3.0) * 0.7)
        try:
            energy, positions = relax(fib_pts, potential, kind)
            if np.all(np.isfinite(positions)) and _validate_positions(positions):
                best_positions = positions
        except:
            pass
    
    # Final fallback: linear arrangement
    if best_positions is None:
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.5, 0, 0]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
