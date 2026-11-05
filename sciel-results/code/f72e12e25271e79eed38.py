"""Dzugutov cluster global minima search with polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A_const = 5.82
B_const = 1.28
a_const = 1.87
b_const = 1.94
c_const = 1.10
d_const = 0.27
m_const = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_const:
                energy += A_const * (r ** (-m_const) - B_const) * np.exp(c_const / (r - a_const))
            elif r < b_const:
                energy += B_const * np.exp(d_const / (r - b_const))
    return energy


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
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
    """Generate 20 vertices of a regular dodecahedron centered at origin."""
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


def _build_polytetrahedral(n, rng, style='icosahedral'):
    """Build a polytetrahedral cluster structure."""
    if n <= 13:
        # 13-atom icosahedron: center + 12
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        if n < 13:
            # Take first n points
            return core[:n]
        return core
    
    if style == 'icosahedral':
        # Start with 13-atom icosahedron
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        points = list(core)
    else:
        # Start with dodecahedron (20 atoms) for larger clusters
        if n >= 20:
            core = _generate_dodecahedron(1.0)
            points = list(core)
        else:
            core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
            points = list(core[:n])
    
    remaining = n - len(points)
    if remaining <= 0:
        return np.array(points[:n])
    
    # Add atoms in concentric shells
    shell_radius_base = 1.5
    shell_count = 0
    for _ in range(remaining):
        # Determine which shell to add to
        shell_idx = min(shell_count, (remaining // 12) + 1)
        shell_radius = shell_radius_base + shell_idx * 0.4
        
        # Random point on sphere
        u = rng.uniform(-1, 1)
        v = rng.uniform(-1, 1)
        theta = np.arccos(u)
        phi_angle = 2 * np.pi * v
        p = np.array([
            shell_radius * np.sin(theta) * np.cos(phi_angle),
            shell_radius * np.sin(theta) * np.sin(phi_angle),
            shell_radius * np.cos(theta)
        ])
        
        # Check minimum distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        
        if valid:
            points.append(p)
        else:
            # Try a few more times with slightly larger radius
            for attempt in range(5):
                shell_radius_try = shell_radius + attempt * 0.1
                u = rng.uniform(-1, 1)
                v = rng.uniform(-1, 1)
                theta = np.arccos(u)
                phi_angle = 2 * np.pi * v
                p_try = np.array([
                    shell_radius_try * np.sin(theta) * np.cos(phi_angle),
                    shell_radius_try * np.sin(theta) * np.sin(phi_angle),
                    shell_radius_try * np.cos(theta)
                ])
                valid = True
                for q in points:
                    if np.linalg.norm(p_try - q) < 0.55:
                        valid = False
                        break
                if valid:
                    points.append(p_try)
                    break
            else:
                # Just add it; relaxation will fix overlaps
                points.append(p)
    
    return np.array(points[:n])


def _build_compact_random(n, rng):
    """Generate n random points in a compact sphere."""
    R = 0.8 * (n / (4/3 * np.pi)) ** (1/3) * 2.0
    points = []
    max_attempts = 2000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
        
    return np.array(points)


def _build_layered(n, rng):
    """Build a layered (close-packed-like) structure."""
    points = []
    layer_spacing = 0.95
    in_plane_spacing = 0.95
    
    # Estimate how many layers we need
    # Each layer holds roughly sqrt(n) atoms in a hexagonal arrangement
    max_layers = int(np.ceil(np.sqrt(n) * 2)) + 2
    
    layer = 0
    while len(points) < n and layer < max_layers:
        z = layer * layer_spacing
        # Hexagonal packing in the layer
        rows = 0
        while len(points) < n:
            row_offset = (rows % 2) * in_plane_spacing * 0.5
            cols = int(np.ceil(n / 8)) + 2
            for col in range(cols):
                if len(points) >= n:
                    break
                x = col * in_plane_spacing + row_offset
                y = rows * in_plane_spacing * 0.866
                p = np.array([x, y, z])
                
                # Check minimum distance to all existing points
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.55:
                        valid = False
                        break
                if valid:
                    points.append(p)
            rows += 1
        layer += 1
    
    # If we still don't have enough, add random points
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
        
    return np.array(points[:n])


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    rng = np.random.default_rng(size)
    
    # Generate diverse candidate structures
    candidate_generators = [
        ('icosahedral', lambda: _build_polytetrahedral(size, rng, style='icosahedral')),
        ('dodecahedral', lambda: _build_polytetrahedral(size, rng, style='dodecahedral')),
        ('random1', lambda: _build_compact_random(size, rng)),
        ('random2', lambda: _build_compact_random(size, rng)),
        ('layered', lambda: _build_layered(size, rng)),
        ('random3', lambda: _build_compact_random(size, rng)),
    ]
    
    # Try each generator multiple times with different random seeds
    attempts_per_gen = 3
    total_attempts = len(candidate_generators) * attempts_per_gen
    
    for gen_idx in range(len(candidate_generators)):
        for attempt in range(attempts_per_gen):
            if time.time() > deadline:
                break
            
            try:
                gen_name, gen_func = candidate_generators[gen_idx]
                # Use different seed for each attempt
                attempt_rng = np.random.default_rng(size * 1000 + gen_idx * 100 + attempt)
                points = gen_func()
                
                # Relax
                energy, positions = relax(points, potential, None)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
                    
            except Exception as e:
                pass
        
        if time.time() > deadline:
            break
    
    # If we still have time, do more random restarts
    if time.time() < deadline - 30:
        extra_attempts = 5
        for i in range(extra_attempts):
            if time.time() > deadline:
                break
            try:
                attempt_rng = np.random.default_rng(size * 1000 + 1000 + i)
                points = _build_compact_random(size, attempt_rng)
                energy, positions = relax(points, potential, None)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass
    
    if best_positions is None:
        # Fallback: return a random valid structure
        points = _build_compact_random(size, rng)
        try:
            energy, positions = relax(points, potential, None)
            return positions
        except:
            return points
            
    return best_positions
# EVOLVE-BLOCK-END
