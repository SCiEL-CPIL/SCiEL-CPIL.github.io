"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize
import threading


# Global deadline guard
_DEADLINE = None
_LOCK = threading.Lock()


def _elapsed():
    if _DEADLINE is None:
        return 0.0
    return time.time() - _DEADLINE


def _time_remaining():
    if _DEADLINE is None:
        return float('inf')
    return _DEADLINE - time.time()


def _generate_icosahedron_shells(n, seed=42):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    points = []
    
    # First shell: icosahedron (12 atoms)
    if n >= 12:
        # Scale icosahedron to appropriate radius
        # For Sutton-Chen, equilibrium distance is around 1.0 in reduced units
        # The icosahedron edge length should be ~1.0
        # Edge length of regular icosahedron with circumradius R is R * sqrt(10 + 2*sqrt(5)) / 3
        # We want edge ~ 1.0, so R = 3 / sqrt(10 + 2*sqrt(5))
        edge_factor = np.sqrt(10 + 2 * np.sqrt(5)) / 3
        R1 = 1.0 / edge_factor
        shell1 = ico_vertices * R1
        points.extend([p.copy() for p in shell1])
    
    # Second shell: 30 vertices (dodecahedron-like)
    if n > 12:
        # Generate 30 points on a sphere of radius ~2*R1
        R2 = 2.0 * R1
        # Use golden ratio spiral
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(30):
            theta = np.arccos(1 - 2 * (i + 0.5) / 30)
            phi = 2 * np.pi * i / golden_ratio
            p = np.array([
                R2 * np.sin(theta) * np.cos(phi),
                R2 * np.sin(theta) * np.sin(phi),
                R2 * np.cos(theta)
            ])
            points.append(p)
    
    # Third shell and beyond: Fibonacci sphere
    remaining = n - len(points)
    if remaining > 0:
        R3 = 3.0 * R1
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(remaining):
            theta = np.arccos(1 - 2 * (i + 0.5) / remaining)
            phi = 2 * np.pi * i / golden_ratio
            p = np.array([
                R3 * np.sin(theta) * np.cos(phi),
                R3 * np.sin(theta) * np.sin(phi),
                R3 * np.cos(theta)
            ])
            points.append(p)
    
    return np.array(points[:n])


def _generate_fcc_shells(n, seed=42):
    """Generate positions based on FCC-like shells."""
    rng = np.random.default_rng(seed)
    
    points = []
    
    # FCC lattice points in a cube
    a = 1.0  # lattice constant
    # Generate FCC points within a sphere of radius ~ (n * a^3 / (4*pi/3))^(1/3)
    max_r = (3 * n * a**3 / (4 * np.pi)) ** (1/3) * 1.2
    
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    n_max = int(np.ceil(max_r / a)) + 1
    for ix in range(-n_max, n_max + 1):
        for iy in range(-n_max, n_max + 1):
            for iz in range(-n_max, n_max + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b
                    if np.linalg.norm(p) <= max_r:
                        points.append(p)
    
    # If we have more points than needed, randomly select
    if len(points) > n:
        indices = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in indices]
    
    # If we have fewer, add random points on a sphere
    while len(points) < n:
        p = rng.uniform(-max_r, max_r, 3)
        if np.linalg.norm(p) <= max_r:
            points.append(p)
    
    return np.array(points[:n])


def _generate_hcp_shells(n, seed=42):
    """Generate positions based on HCP-like shells."""
    rng = np.random.default_rng(seed)
    
    points = []
    
    # HCP lattice
    a = 1.0
    c = np.sqrt(8/3) * a  # ideal HCP
    
    max_r = (3 * n * a**3 / (4 * np.pi)) ** (1/3) * 1.2
    n_max = int(np.ceil(max_r / a)) + 1
    
    # HCP basis
    basis = np.array([
        [0, 0, 0],
        [2/3, 1/3, 0],
        [1/3, 2/3, 0],
        [0, 0, 0.5],
        [2/3, 1/3, 0.5],
        [1/3, 2/3, 0.5]
    ])
    
    for ix in range(-n_max, n_max + 1):
        for iy in range(-n_max, n_max + 1):
            for iz in range(-n_max, n_max + 1):
                for b in basis:
                    # Transform to Cartesian
                    x = a * (ix + b[0])
                    y = a * np.sqrt(3) / 2 * (iy + b[1])
                    z = c * iz + c * b[2]
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= max_r:
                        points.append(p)
    
    if len(points) > n:
        indices = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in indices]
    
    while len(points) < n:
        p = rng.uniform(-max_r, max_r, 3)
        if np.linalg.norm(p) <= max_r:
            points.append(p)
    
    return np.array(points[:n])


def _generate_random_compact(n, seed=42, spacing=0.72):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_with_gradient(positions, kind):
    """Relax using L-BFGS-B with gradient."""
    def objective_and_grad(x):
        positions_reshaped = x.reshape(-1, 3)
        energy, gradient = sutton_chen_energy_gradient(positions_reshaped, kind)
        return energy, gradient
    
    result = minimize(
        objective_and_grad,
        positions.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < np.inf:
        return result.fun, result.x.reshape(-1, 3)
    else:
        return float('inf'), positions


def construct(instance):
    global _DEADLINE
    
    # Set deadline with safety margin
    if _DEADLINE is None:
        _DEADLINE = time.time() + 320  # 10s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    if _time_remaining() < 5:
        # Not enough time, return a simple random structure
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0)
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    # Generate multiple initial structures
    generators = [
        ('ico', lambda: _generate_icosahedron_shells(size, seed=size)),
        ('fcc', lambda: _generate_fcc_shells(size, seed=size)),
        ('hcp', lambda: _generate_hcp_shells(size, seed=size)),
        ('rand1', lambda: _generate_random_compact(size, seed=size + 1)),
        ('rand2', lambda: _generate_random_compact(size, seed=size + 2)),
        ('rand3', lambda: _generate_random_compact(size, seed=size + 3)),
        ('rand4', lambda: _generate_random_compact(size, seed=size + 4)),
    ]
    
    best_energy = float('inf')
    best_positions = None
    
    for gen_name, gen_func in generators:
        if _time_remaining() < 15:
            break
        
        try:
            positions = gen_func()
            
            # Validate initial positions
            if not np.all(np.isfinite(positions)):
                continue
            if positions.shape != (size, 3):
                continue
            
            # Check minimum distance
            min_dist = float('inf')
            for i in range(size):
                for j in range(i + 1, size):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 0.35:
                continue
            
            # Relax
            energy, relaxed_positions = _relax_with_gradient(positions, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_positions.copy()
                
        except Exception:
            continue
    
    # If no valid result, try the built-in relax
    if best_positions is None:
        if _time_remaining() < 10:
            rng = np.random.default_rng(size)
            radius = 0.72 * size ** (1.0 / 3.0)
            points = rng.uniform(-radius, radius, (size, 3))
            return points
        
        try:
            positions = _generate_icosahedron_shells(size, seed=size)
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                return relaxed
        except Exception:
            pass
        
        # Fallback
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0)
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0)
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    min_dist = float('inf')
    for i in range(size):
        for j in range(i + 1, size):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0)
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    return best_positions
# EVOLVE-BLOCK-END
