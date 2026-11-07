"""Sutton-Chen cluster global minima search using structured templates."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, a=1.0):
    """Generate n positions from an FCC lattice fragment."""
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    # Generate FCC lattice points in a cube
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    candidates = np.array(candidates)
    # Center
    center = candidates.mean(axis=0)
    candidates -= center
    # Sort by distance from center, take n closest
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    return candidates[idx]


def _hcp_positions(n, a=1.0):
    """Generate n positions from an HCP lattice fragment."""
    points = []
    # HCP: two atoms per unit cell in hexagonal arrangement
    # Use a simplified approach: generate hexagonal close-packed layers
    c = a * np.sqrt(8/3)  # c/a ratio for ideal HCP
    a_hex = a * np.sqrt(2/3)
    
    candidates = []
    layers = int(np.ceil((n / 6) ** (1/3))) + 2
    for layer in range(-layers, layers + 1):
        z = layer * c
        for i in range(-layers, layers + 1):
            for j in range(-layers, layers + 1):
                # Hexagonal grid
                x = i * a_hex + (j % 2) * a_hex / 2
                y = j * a_hex * np.sqrt(3) / 2
                # Two atoms per layer (AB stacking)
                offset = 0 if layer % 2 == 0 else a_hex / 2
                candidates.append([x + offset, y, z])
                candidates.append([x + offset + a_hex/2, y + a_hex*np.sqrt(3)/4, z])
    
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    return candidates[idx]


def _icosahedron_positions(n, r=1.0):
    """Generate positions on/inside an icosahedral structure."""
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r
    
    if n <= 13:
        # Center + up to 12 vertices
        pts = [np.zeros(3)] + list(verts[:n-1])
        return np.array(pts)
    
    # For larger n, add concentric shells or use Mackay icosahedron
    # Mackay icosahedron: 1 + 12 + 42 = 55 for n=55
    # 1 + 12 + 42 + 92 = 147 for next shell
    
    # Use a simple approach: place atoms on concentric icosahedral shells
    pts = [np.zeros(3)]
    # Shell 1: 12 atoms
    for v in verts:
        pts.append(v)
    if n <= 13:
        return np.array(pts[:n])
    
    # Shell 2: 42 atoms (Mackay) - use a denser icosahedral surface
    # Approximate with a sphere of 42 points
    from scipy.spatial import ConvexHull
    # Use Fibonacci sphere for 42 points
    def fib_sphere(n_pts, radius):
        pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n_pts):
            z = 1 - 2 * (i + 0.5) / n_pts
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        return np.array(pts) * radius
    
    shell2 = fib_sphere(42, 2.0)
    pts.extend([p for p in shell2])
    if n <= 55:
        return np.array(pts[:n])
    
    # Shell 3: 92 atoms
    shell3 = fib_sphere(92, 3.0)
    pts.extend([p for p in shell3])
    
    return np.array(pts[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster with good initial packing."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    radius = spacing * n ** (1.0/3.0) * 0.9
    
    # Use rejection sampling with good initial packing
    attempts = 0
    while len(points) < n and attempts < n * 200:
        attempts += 1
        # Sample in a sphere
        p = rng.normal(0, radius/2, 3)
        if np.linalg.norm(p) > radius:
            continue
        # Check minimum distance
        min_dist = np.min([np.linalg.norm(p - q) for q in points]) if points else np.inf
        if min_dist >= 0.6 * spacing:
            points.append(p)
    
    # If we didn't get enough, add more with looser constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    return np.array(points)


def construct(instance):
    """Construct Sutton-Chen cluster using multiple structured templates."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Generate multiple diverse starting geometries
    templates = []
    
    # 1. FCC fragment
    try:
        templates.append(_fcc_positions(size, a=0.72))
    except:
        pass
    
    # 2. HCP fragment
    try:
        templates.append(_hcp_positions(size, a=0.72))
    except:
        pass
    
    # 3. Icosahedral
    try:
        templates.append(_icosahedron_positions(size, r=0.72 * size**(1/3)))
    except:
        pass
    
    # 4-7. Random compact with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            templates.append(_random_compact(size, spacing=0.72, seed=seed))
        except:
            pass
    
    # Relax each template and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, template in enumerate(templates):
        if len(template) != size:
            continue
        try:
            energy, positions = relax(template, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
    
    if best_positions is None:
        # Fallback: simple random
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1/3)
        best_positions = rng.uniform(-radius, radius, (size, 3))
        best_positions = relax(best_positions, potential, kind)[1]
    
    return best_positions
# EVOLVE-BLOCK-END
