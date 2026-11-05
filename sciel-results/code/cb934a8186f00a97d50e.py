"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly and efficiently (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    # positions shape: (n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r = dist[mask]
    
    if len(r) == 0:
        return 0.0
        
    energy = 0.0
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        # Avoid division by zero if any r is extremely small
        r1 = np.maximum(r1, 1e-12)
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
        
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r[mask2]
        energy += np.sum(B * np.exp(d / (r2 - b)))
        
    return float(energy)


def _ensure_min_distance(positions, min_dist=0.5, max_iters=50):
    """Ensure all pairs are at least min_dist apart by simple iterative pushing."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iters):
        violated = False
        # Vectorized distance check for efficiency
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Get indices of violations
        i_idx, j_idx = np.where(np.triu(np.ones((n, n), dtype=bool), k=1) & (dist < min_dist))
        
        if len(i_idx) == 0:
            break
            
        for i, j in zip(i_idx, j_idx):
            d = dist[i, j]
            if d < 1e-6:
                # Random nudge
                positions[i] += 0.1 * np.random.standard_normal(3)
            else:
                direction = (positions[i] - positions[j]) / d
                push = (min_dist - d) / 2 + 0.01
                positions[i] += push * direction
                positions[j] -= push * direction
            violated = True
            
        if not violated:
            break
            
    return positions


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _generate_dodecahedron(radius=1.0):
    """Generate 20 vertices of a regular dodecahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    # 8 cube vertices
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([sx, sy, sz])
    # 4 vertices in xy-plane
    verts.append([0, 1/phi, phi])
    verts.append([0, -1/phi, phi])
    verts.append([0, 1/phi, -phi])
    verts.append([0, -1/phi, -phi])
    # 4 vertices in xz-plane
    verts.append([1/phi, phi, 0])
    verts.append([-1/phi, phi, 0])
    verts.append([1/phi, -phi, 0])
    verts.append([-1/phi, -phi, 0])
    # 4 vertices in yz-plane
    verts.append([phi, 0, 1/phi])
    verts.append([phi, 0, -1/phi])
    verts.append([-phi, 0, 1/phi])
    verts.append([-phi, 0, -1/phi])
    
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_icosahedral_cluster(n, rng):
    """Build a Mackay-like icosahedral cluster."""
    if n <= 12:
        return _generate_icosahedron(1.0)[:n]
    
    # Start with 13-atom icosahedron (center + 12)
    positions = [np.zeros(3)] + list(_generate_icosahedron(1.0))
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Add 2nd shell (30 atoms) at edge midpoints scaled out
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    # Approximate edge midpoints by sampling directions
    # For simplicity and robustness, we add atoms on a spherical shell at radius ~1.55
    # This mimics the 2nd shell of a Mackay icosahedron
    for i in range(n - len(positions)):
        # Sample direction
        u = rng.uniform(-1, 1)
        v = rng.uniform(-1, 1)
        theta = np.arccos(u)
        phi_angle = 2 * np.pi * v
        r = 1.55 + rng.uniform(-0.1, 0.1)
        p = np.array([
            r * np.sin(theta) * np.cos(phi_angle),
            r * np.sin(theta) * np.sin(phi_angle),
            r * np.cos(theta)
        ])
        positions.append(p)
        
    return np.array(positions[:n])


def _build_dodecahedral_cluster(n, rng):
    """Build a dodecahedral cluster."""
    if n <= 20:
        return _generate_dodecahedron(1.0)[:n]
    
    positions = list(_generate_dodecahedron(1.0))
    
    # Add atoms in outer shells
    for i in range(n - len(positions)):
        u = rng.uniform(-1, 1)
        v = rng.uniform(-1, 1)
        theta = np.arccos(u)
        phi_angle = 2 * np.pi * v
        r = 1.8 + rng.uniform(-0.2, 0.2)
        p = np.array([
            r * np.sin(theta) * np.cos(phi_angle),
            r * np.sin(theta) * np.sin(phi_angle),
            r * np.cos(theta)
        ])
        positions.append(p)
        
    return np.array(positions[:n])


def _build_random_compact(n, rng):
    """Generate n random points in a compact sphere."""
    R = 0.8 * (n / (4/3 * np.pi)) ** (1/3) * 2.0
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        # Simple check against existing points
        if all(np.linalg.norm(p - np.array(q)) >= 0.55 for q in points):
            points.append(p)
            
    # Fill remaining if needed
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
        
    return np.array(points)


def _build_fibonacci_sphere(n, rng):
    """Generate points on a Fibonacci sphere."""
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0,0,0]])
    
    points = []
    # Center atom
    points.append(np.zeros(3))
    n_remaining = n - 1
    
    # Golden ratio
    gr = (1 + np.sqrt(5)) / 2
    offset = 2.0 / n_remaining
    increment = np.pi * (3 - np.sqrt(5))
    
    for i in range(n_remaining):
        y = ((i * offset) - 1) + (offset / 2)
        radius = np.sqrt(1 - y * y)
        phi = i * increment
        x = np.cos(phi) * radius
        z = np.sin(phi) * radius
        # Scale to appropriate radius for Dzugutov (approx 1.0-1.5)
        p = np.array([x, y, z]) * 1.2
        points.append(p)
        
    return np.array(points[:n])


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using multiple
    polytetrahedral/icosahedral initial guesses and local relaxation.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    rng = np.random.default_rng(n * 7919 + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Candidate generators
    candidates = []
    
    # 1. Icosahedral (Mackay-like)
    candidates.append(_build_icosahedral_cluster(n, rng))
    # 2. Dodecahedral
    candidates.append(_build_dodecahedral_cluster(n, rng))
    # 3. Random Compact
    candidates.append(_build_random_compact(n, rng))
    # 4. Fibonacci Sphere
    candidates.append(_build_fibonacci_sphere(n, rng))
    
    # 5. Perturbed Icosahedral
    ico = _build_icosahedral_cluster(n, rng)
    ico_pert = ico + 0.1 * rng.standard_normal(ico.shape)
    candidates.append(ico_pert)
    
    # 6. Perturbed Random
    rand = _build_random_compact(n, rng)
    rand_pert = rand + 0.1 * rng.standard_normal(rand.shape)
    candidates.append(rand_pert)
    
    # Process candidates
    for i, pos in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        # Ensure minimum distance
        pos = _ensure_min_distance(pos, min_dist=0.5)
        
        # Quick energy check to filter out obviously bad structures
        e_init = _dzugutov_energy(pos)
        if e_init > 1e10:
            continue
            
        # Relax using the suite's optimized function
        try:
            energy, relaxed = relax(pos, potential, instance.get("kind", instance.get("rho")))
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
            
        # If we have a very good result, we can stop early
        if best_energy < -100 * n: # Heuristic: very low energy per atom
            break

    # If no valid result found, return the best initial guess or a fallback
    if best_positions is None:
        # Fallback: use the first candidate
        pos = _ensure_min_distance(candidates[0], min_dist=0.5)
        try:
            energy, relaxed = relax(pos, potential, instance.get("kind", instance.get("rho")))
            best_positions = relaxed
        except Exception:
            best_positions = pos
            
    return best_positions
# EVOLVE-BLOCK-END
