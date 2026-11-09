"""Dzugutov cluster global minima via polytetrahedral geometric initialization with Langevin refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 275.0  # safety margin before 300s kill

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _dzugutov_gradient(positions):
    """Compute Dzugutov potential energy gradient."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                dr = -m * r ** (-m - 1) * A * np.exp(c / (r - a)) + \
                     A * (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2)
                force = dr * rij / r
                grad[i] += force
                grad[j] -= force
            elif r < b:
                dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
                force = dr * rij / r
                grad[i] += force
                grad[j] -= force
    return grad


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts


def _build_polytetrahedral(n, rng):
    """Build a polytetrahedral cluster of n atoms using icosahedral shells."""
    # Start with a central atom
    points = [np.zeros(3)]
    
    # Icosahedral first shell (12 atoms)
    ico_scale = 1.15  # typical nearest-neighbor distance
    ico_verts = _icosahedron_vertices(ico_scale)
    for v in ico_verts:
        points.append(v.copy())
    
    # Second shell: positions on faces and edges of the icosahedron
    # Mackay icosahedron has 55 atoms: 1 + 12 + 42
    # We need to add atoms to reach n
    
    if n <= 13:
        return np.array(points[:n])
    
    # Add second shell atoms
    # For a Mackay icosahedron, the second shell has 42 atoms
    # These are located at:
    # - 30 edge midpoints (between adjacent icosahedron vertices)
    # - 12 face centers
    
    # Get icosahedron faces and edges
    # Define faces of the icosahedron (20 triangular faces)
    faces = [
        [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 2], [0, 2, 4],
        [1, 5, 9], [1, 9, 11], [1, 11, 7], [1, 7, 3], [1, 3, 5],
        [2, 6, 10], [2, 10, 8], [2, 8, 4], [2, 4, 0], [2, 0, 6],
        [3, 7, 11], [3, 11, 9], [3, 9, 5], [3, 5, 1], [3, 1, 7]
    ]
    
    # Add face centers (12 atoms)
    for face in faces[:12]:
        center = (ico_verts[face[0]] + ico_verts[face[1]] + ico_verts[face[2]]) / 3.0
        # Scale outward slightly
        center = center * 1.3
        points.append(center)
    
    # Add edge midpoints (30 atoms)
    edges = set()
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges.add(e)
    
    edge_list = list(edges)
    for i, (v1, v2) in enumerate(edge_list):
        if len(points) >= n:
            break
        mid = (ico_verts[v1] + ico_verts[v2]) / 2.0
        mid = mid * 1.2
        points.append(mid)
    
    # If we still need more atoms, add them in a third shell or random positions
    while len(points) < n:
        # Add atoms at random positions on a sphere of radius ~2.3
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(-1, 1)
        r = 2.3 * rng.uniform(0.9, 1.1)
        p = np.array([
            r * np.sqrt(1 - phi**2) * np.cos(theta),
            r * np.sqrt(1 - phi**2) * np.sin(theta),
            r * phi
        ])
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Try again with a different position
            continue
    
    return np.array(points[:n])


def _langevin_refine(positions, n_steps=500, dt=0.01, temperature=0.1, rng=None):
    """Refine positions using Langevin dynamics."""
    if rng is None:
        rng = np.random.default_rng()
    
    positions = positions.copy()
    n = len(positions)
    velocities = np.zeros_like(positions)
    
    for step in range(n_steps):
        if time.time() > _DEADLINE:
            break
        
        # Compute forces
        forces = -_dzugutov_gradient(positions)
        
        # Langevin update
        random_forces = rng.normal(0, np.sqrt(2 * temperature / dt), size=positions.shape)
        velocities = velocities * 0.9 + dt * (forces + random_forces)
        positions = positions + dt * velocities
        
        # Ensure minimum distance constraint
        for i in range(n):
            for j in range(i + 1, n):
                rij = positions[i] - positions[j]
                r = np.linalg.norm(rij)
                if r < 0.5 and r > 1e-10:
                    # Push apart
                    correction = (0.5 - r) * rij / r
                    positions[i] += correction / 2
                    positions[j] -= correction / 2
    
    return positions


def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    size = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(size + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Polytetrahedral geometric initialization
    t0 = time.time()
    max_attempts = 8
    attempts = 0
    
    while attempts < max_attempts and time.time() < _DEADLINE:
        attempts += 1
        
        # Build polytetrahedral structure
        positions = _build_polytetrahedral(size, rng)
        
        # Refine with Langevin dynamics
        positions = _langevin_refine(positions, n_steps=300, dt=0.005, temperature=0.05, rng=rng)
        
        # Final relaxation with the provided relax function
        energy, relaxed_positions = relax(positions, potential)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_positions
        
        # Check time
        if time.time() > _DEADLINE - 30:
            break
    
    # Strategy 2: If we have time, try a few random compact clusters
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 30 and best_positions is not None:
        for _ in range(3):
            if time.time() > _DEADLINE - 10:
                break
            
            radius = 1.15 * size ** (1.0 / 3.0) * 0.8
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
            
            energy, positions = relax(np.array(points), potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Final high-precision polish
    if best_positions is not None and time.time() < _DEADLINE - 5:
        # Run Langevin at lower temperature for final refinement
        best_positions = _langevin_refine(best_positions, n_steps=200, dt=0.002, temperature=0.01, rng=rng)
        energy, best_positions = relax(best_positions, potential)
    
    if best_positions is None:
        # Fallback: random cluster
        radius = 1.15 * size ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
        _, best_positions = relax(np.array(points), potential)
    
    return best_positions
# EVOLVE-BLOCK-END
