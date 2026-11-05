"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron with given radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    # Normalize to unit radius then scale
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts


def _octahedron_vertices(radius=1.0):
    """Generate 6 vertices of a regular octahedron with given radius."""
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float) * radius
    return verts


def _tetrahedron_vertices(radius=1.0):
    """Generate 4 vertices of a regular tetrahedron with given radius."""
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts


def _build_polytetrahedral_cluster(n, rng, scale=1.1):
    """
    Build a cluster by growing an icosahedral core and adding atoms
    in positions that favor polytetrahedral coordination.
    """
    if n <= 12:
        # Just use icosahedron vertices (or subset)
        verts = _icosahedron_vertices(scale)
        if n <= 12:
            return verts[:n].copy()
    
    # Start with icosahedral core (12 atoms)
    positions = _icosahedron_vertices(scale).tolist()
    
    # For atoms beyond 12, add them in second shell positions
    # The second shell of an icosahedron has 30 positions (face centers)
    # and then more in outer shells
    
    # Generate candidate positions for additional atoms
    # Use the face centers of the icosahedron for the next layer
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unit = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1, keepdims=True)
    verts_unit = verts_unit / norms
    
    # Find face centers of icosahedron (20 faces)
    # Each face is a triangle of 3 vertices
    # We can compute face centers by averaging 3 vertices that form a face
    # For simplicity, generate positions on a sphere at larger radius
    # and pick those that are well-separated from existing atoms
    
    target_count = n
    max_radius = scale * 2.5
    min_dist = 0.55  # Minimum distance between atoms
    
    attempts = 0
    max_attempts = 10000
    
    while len(positions) < target_count and attempts < max_attempts:
        attempts += 1
        # Try to add an atom at a random position on a sphere shell
        # Bias toward positions that are well-coordinated
        r = scale * (1.0 + 0.3 * np.sqrt(attempts / max_attempts))
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        
        # Check minimum distance
        if all(np.linalg.norm(np.array(pos) - p) >= min_dist for pos in positions):
            positions.append(p)
    
    # If we still don't have enough atoms, add them more aggressively
    while len(positions) < target_count:
        r = scale * (1.0 + 0.2 * rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        # Just add it, relax will fix issues
        positions.append(p)
    
    return np.array(positions)


def _build_icosahedral_cluster(n, rng, scale=1.1):
    """
    Build a cluster based on icosahedral symmetry.
    For n=13: 12+1 (central atom)
    For larger n: add shells
    """
    if n == 13:
        positions = _icosahedron_vertices(scale)
        positions = np.vstack([positions, [[0, 0, 0]]])
        return positions
    
    # Start with icosahedral core
    positions = _icosahedron_vertices(scale).tolist()
    positions.append([0.0, 0.0, 0.0])  # Central atom (13 total)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Add second shell atoms
    # The 30 edge midpoints of the icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unit = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1, keepdims=True)
    verts_unit = verts_unit / norms
    
    # Compute edge midpoints (30 edges)
    edges = [
        (0,1),(0,2),(0,4),(0,5),(0,8),(0,11),
        (1,3),(1,4),(1,5),(1,9),(1,10),
        (2,3),(2,6),(2,7),(2,8),(2,11),
        (3,6),(3,7),(3,9),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,11),
        (5,6),(5,7),(5,8),(5,9),
        (6,8),(6,10),
        (7,8),(7,10),
        (8,11),(9,11)
    ]
    
    second_shell = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * scale * 1.4  # Scale outward
        second_shell.append(mid)
    
    # Add second shell atoms until we have enough
    for pos in second_shell:
        if len(positions) >= n:
            break
        positions.append(pos)
    
    # If still not enough, add more
    while len(positions) < n:
        r = scale * (1.5 + 0.3 * rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        positions.append(p)
    
    return np.array(positions[:n])


def _build_random_compact(n, rng, scale=1.1):
    """Generate a random compact cluster."""
    positions = []
    radius = scale * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.55
    
    attempts = 0
    max_attempts = 50000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(pos)) >= min_dist for pos in positions):
                positions.append(p)
    
    # If we couldn't place all atoms, add remaining randomly
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions)


def _relax_with_timeout(positions, potential, parameter, time_budget=10.0):
    """Relax positions with a time budget."""
    start = time.time()
    try:
        energy, relaxed = relax(positions, potential, parameter)
        elapsed = time.time() - start
        return energy, relaxed, elapsed
    except Exception as e:
        return float('inf'), positions, time.time() - start


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using multiple
    polytetrahedral/icosahedral initial guesses and local relaxation.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(n * 7919)  # Different seed for each n
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate multiple diverse initial guesses
    n_restarts = 12
    
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Choose initialization strategy
        strategy = i % 4
        
        if strategy == 0:
            # Icosahedral-based
            positions = _build_icosahedral_cluster(n, rng, scale=1.1 + 0.05 * rng.random())
        elif strategy == 1:
            # Polytetrahedral growth
            positions = _build_polytetrahedral_cluster(n, rng, scale=1.1 + 0.05 * rng.random())
        elif strategy == 2:
            # Random compact
            positions = _build_random_compact(n, rng, scale=1.1 + 0.1 * rng.random())
        else:
            # Icosahedral with perturbation
            positions = _build_icosahedral_cluster(n, rng, scale=1.1)
            noise = 0.1 * rng.standard_normal(positions.shape)
            positions = positions + noise
        
        # Ensure minimum distance constraint
        # Simple check: if any pair is too close, nudge them apart
        for _ in range(50):
            min_dist_violation = False
            for k in range(len(positions)):
                for l in range(k + 1, len(positions)):
                    dist = np.linalg.norm(positions[k] - positions[l])
                    if dist < 0.5:
                        # Nudge apart
                        if dist < 1e-6:
                            positions[k] += 0.1 * rng.standard_normal(3)
                        else:
                            direction = (positions[k] - positions[l]) / dist
                            positions[k] += 0.05 * direction
                            positions[l] -= 0.05 * direction
                        min_dist_violation = True
            if not min_dist_violation:
                break
        
        # Relax
        energy, relaxed, elapsed = _relax_with_timeout(positions, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
    
    # If we have time, do a few more restarts with better initial guesses
    if best_positions is not None and time.time() < deadline - 30:
        # Use the best position as a template and add noise
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            noise = 0.15 * rng.standard_normal(best_positions.shape)
            positions = best_positions + noise
            
            # Ensure minimum distance
            for _ in range(50):
                min_dist_violation = False
                for k in range(len(positions)):
                    for l in range(k + 1, len(positions)):
                        dist = np.linalg.norm(positions[k] - positions[l])
                        if dist < 0.5:
                            if dist < 1e-6:
                                positions[k] += 0.1 * rng.standard_normal(3)
                            else:
                                direction = (positions[k] - positions[l]) / dist
                                positions[k] += 0.05 * direction
                                positions[l] -= 0.05 * direction
                            min_dist_violation = True
                if not min_dist_violation:
                    break
            
            energy, relaxed, elapsed = _relax_with_timeout(positions, potential, parameter)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
    
    if best_positions is None:
        # Fallback: return a simple compact arrangement
        positions = _build_random_compact(n, rng, scale=1.1)
        energy, relaxed, elapsed = _relax_with_timeout(positions, potential, parameter)
        best_positions = relaxed
    
    return best_positions
# EVOLVE-BLOCK-END
