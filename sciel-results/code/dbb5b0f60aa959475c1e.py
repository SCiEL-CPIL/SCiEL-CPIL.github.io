"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _macky_icosahedron(n):
    """
    Generate a Mackay icosahedral structure with up to 63 atoms.
    If n < 63, it returns a subset.
    """
    # 1. Generate icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Identify edges and faces
    # Edge length in unit icosahedron
    # Distance between (0,1,phi) and (0,-1,phi) is 2.
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 0.382 + 2.618) = sqrt(4) = 2.
    # So edge length is 2.
    
    edges = []
    faces = []
    n_verts = len(verts)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, 2.0, atol=1e-6):
                edges.append((i, j))
    
    # Find faces: triplets of vertices that are all connected
    # For each pair (i,j) in edges, find k such that (i,k) and (j,k) are also edges
    edge_set = set(edges)
    for i, j in edges:
        for k in range(n_verts):
            if k == i or k == j:
                continue
            if (min(i, k), max(i, k)) in edge_set and (min(j, k), max(j, k)) in edge_set:
                # Sort triplet to avoid duplicates
                triplet = tuple(sorted([i, j, k]))
                if triplet not in faces:
                    faces.append(triplet)
    
    # 3. Generate shells
    # Shell 1: Vertices (12 atoms)
    shell1 = verts
    
    # Shell 2: Face centers (20 atoms)
    # Centroid of 3 vertices
    shell2 = []
    for f in faces:
        centroid = np.mean(verts[list(f)], axis=0)
        shell2.append(centroid)
    shell2 = np.array(shell2)
    
    # Shell 3: Edge midpoints (30 atoms)
    shell3 = []
    for e in edges:
        midpoint = (verts[e[0]] + verts[e[1]]) / 2.0
        shell3.append(midpoint)
    shell3 = np.array(shell3)
    
    # Add center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Combine all
    all_points = np.vstack([center, shell1, shell2, shell3])
    
    # The radii are:
    # Center: 0
    # Shell 1: 1.0
    # Shell 2: norm of centroid
    # Shell 3: norm of midpoint
    
    # Scale the structure so that the nearest neighbor distance is ~1.9
    # In unit icosahedron, NN distance is 2.0 (edge length).
    # So scale factor = 1.9 / 2.0 = 0.95
    scale = 0.95
    all_points = all_points * scale
    
    # If n is larger than 63, we need to add more shells or expand.
    # But the task only goes up to 60.
    # So we just trim.
    
    if n >= len(all_points):
        return all_points.copy()
    
    # Trimming strategy:
    # Prioritize keeping: Center, Shell 1, Shell 2.
    # Trim from Shell 3 first.
    # Indices:
    # 0: Center
    # 1-12: Shell 1
    # 13-32: Shell 2
    # 33-62: Shell 3
    
    # We want to keep the first `n` atoms if we order them by importance.
    # Let's reorder: Center, Shell 1, Shell 2, Shell 3
    # Current order is already Center, Shell 1, Shell 2, Shell 3.
    # So just take the first n.
    
    return all_points[:n].copy()


def _random_compact(n, seed, spacing=1.1, radius_factor=0.75):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.8 * spacing
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms with strict constraint, relax it
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    # Last resort: just add random points
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Construct low-energy Dzugutov cluster using structured initializations and relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Mackay Icosahedron (Primary)
    # Generate 63-atom Mackay, trim to size
    macky_points = _macky_icosahedron(63)
    if len(macky_points) >= size:
        init_points = macky_points[:size].copy()
    else:
        # Fallback if size > 63 (not expected here)
        init_points = _random_compact(size, seed=size)
        
    # Relax
    energy, positions = relax(init_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 2: Perturbed Mackay
    rng = np.random.default_rng(size + 1)
    perturbation = rng.normal(0, 0.05, init_points.shape)
    perturbed_points = init_points + perturbation
    energy, positions = relax(perturbed_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 3: Different trimming of Mackay
    # Maybe remove atoms from the outside in a different way?
    # For now, try random compact as a diverse start
    rc_points = _random_compact(size, seed=size + 100, spacing=1.1, radius_factor=0.75)
    energy, positions = relax(rc_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
        
    # Strategy 4: Scaled Mackay (slightly larger)
    # Sometimes the optimal bond length is slightly different
    macky_scaled = _macky_icosahedron(63) * 1.05
    if len(macky_scaled) >= size:
        init_points2 = macky_scaled[:size].copy()
    else:
        init_points2 = macky_scaled
    energy, positions = relax(init_points2, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions

    if _time_remaining() < 40:
        return best_positions

    # Strategy 5: Scaled Mackay (slightly smaller)
    macky_scaled2 = _macky_icosahedron(63) * 0.95
    if len(macky_scaled2) >= size:
        init_points3 = macky_scaled2[:size].copy()
    else:
        init_points3 = macky_scaled2
    energy, positions = relax(init_points3, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
