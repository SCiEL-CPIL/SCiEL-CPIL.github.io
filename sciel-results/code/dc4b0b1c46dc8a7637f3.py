# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_gradient(x, rho):
    """
    Wrapper for morse_energy_gradient that reshapes input/output for scipy.
    x: (n*3,) array
    Returns: (energy, gradient) where gradient is (n*3,)
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _generate_fcc_cluster(n, seed=0):
    """
    Generate an FCC-like cluster of n atoms.
    FCC is the known global minimum structure for Morse potentials in this range.
    We generate a truncated FCC lattice.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius for n atoms in FCC
    # Volume of sphere ~ (4/3) pi R^3. Density of FCC ~ 0.74.
    # n * volume_per_atom ~ 4/3 pi R^3.
    # volume_per_atom ~ 1.0 (since equilibrium distance is 1, unit cell volume is ~ 4 atoms / (2^3/2) ?)
    # Let's just use a radius factor.
    radius = 1.5 * (n ** (1.0/3.0))
    
    lattice_points = []
    r = int(math.ceil(radius)) + 2
    
    # FCC basis vectors
    # Unit cell has atoms at (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # But standard FCC lattice points are at integer coordinates (i,j,k) where i+j+k is even? No.
    # Standard FCC:
    # (0,0,0), (0, 1/2, 1/2), (1/2, 0, 1/2), (1/2, 1/2, 0) + integer translations.
    # To get distance 1 between nearest neighbors:
    # Distance between (0,0,0) and (0, 0.5, 0.5) is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707.
    # We want distance 1. So we scale by 1/sqrt(0.5) = sqrt(2).
    # So basis points are:
    # (0,0,0), (0, 1, 1), (1, 0, 1), (1, 1, 0) scaled by 0.5 * sqrt(2) = 1/sqrt(2).
    # Wait, let's just use the standard construction and scale at the end or during generation.
    
    # Let's generate points in a cubic box and filter.
    # Nearest neighbor distance in FCC is a/sqrt(2) where a is lattice constant.
    # We want d_nn = 1. So a = sqrt(2).
    # Basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0).
    
    a = math.sqrt(2)
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                for b in basis:
                    p = np.array([i*a + b[0], j*a + b[1], k*a + b[2]])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) < n:
        # If we didn't get enough, expand radius or just pad with random
        # This shouldn't happen with r = ceil(radius) + 2
        pass
        
    # Sort by distance from center to get a compact cluster
    lattice_points.sort(key=lambda p: np.linalg.norm(p))
    
    # Take the first n
    selected = lattice_points[:n]
    
    # Add tiny jitter to break symmetry and help optimizer
    jitter = rng.uniform(-0.001, 0.001, (n, 3))
    return np.array(selected) + jitter

def _generate_bcc_cluster(n, seed=0):
    """
    Generate a BCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    r = int(math.ceil(radius)) + 2
    
    # BCC basis: (0,0,0), (0.5, 0.5, 0.5)
    # Nearest neighbor distance: sqrt(3)/2 * a = 1 => a = 2/sqrt(3)
    a = 2.0 / math.sqrt(3.0)
    basis = [
        [0, 0, 0],
        [a/2, a/2, a/2]
    ]
    
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                for b in basis:
                    p = np.array([i*a + b[0], j*a + b[1], k*a + b[2]])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
                        
    lattice_points.sort(key=lambda p: np.linalg.norm(p))
    selected = lattice_points[:n]
    jitter = rng.uniform(-0.001, 0.001, (n, 3))
    return np.array(selected) + jitter

def _generate_icosahedral_cluster(n, seed=0):
    """
    Generate an icosahedral cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.2 * (n ** (1.0/3.0))
    
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    ico_vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    
    # Normalize to unit sphere
    ico_vertices = [np.array(v) / np.linalg.norm(v) for v in ico_vertices]
    
    # Shells
    # Shell 1: 12 vertices
    # Shell 2: 30 points (midpoints of edges)
    # Shell 3: 20 points (face centers)
    # Shell 4: 12 points (vertices of next shell)
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Scale factor for shell 1 to be distance 1 from center?
    # In icosahedral clusters, the distance from center to first shell is usually ~1.
    # Let's assume the first shell radius is 1.0.
    
    # Add Shell 1
    for v in ico_vertices:
        points.append(v)
        
    if len(points) >= n:
        points = points[:n]
        scale = radius / np.max(np.linalg.norm(points, axis=1))
        return np.array(points) * scale + rng.uniform(-0.001, 0.001, (n, 3))
        
    # Add Shell 2 (Edge midpoints)
    # Find adjacent vertices
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_vertices[i] - ico_vertices[j])
            if d < 1.5: # Adjacent
                edges.append((i, j))
                
    for i, j in edges:
        mid = (ico_vertices[i] + ico_vertices[j]) / 2.0
        mid = mid / np.linalg.norm(mid) # Project back to sphere? No, keep radius.
        # Actually, for a proper icosahedral shell, the radius increases.
        # Let's just use a simple radial expansion for simplicity if we need more points.
        # This is a rough approximation.
        
    # For now, if we need more, just add random points on a sphere of radius 2
    remaining = n - len(points)
    for _ in range(remaining):
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * 2.0
        points.append(p)
        
    points = np.array(points[:n])
    scale = radius / np.max(np.linalg.norm(points, axis=1))
    return points * scale + rng.uniform(-0.001, 0.001, (n, 3))

def _refine_structure(positions, rho, max_iter=5000):
    """
    Refine structure using L-BFGS-B.
    """
    n = len(positions)
    x0 = positions.flatten()
    
    try:
        result = minimize(
            _morse_energy_gradient,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success:
            polished = result.x.reshape(n, 3)
            # Check minimum distance
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if np.linalg.norm(polished[i] - polished[j]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result.fun, polished
    except Exception:
        pass
        
    # Fallback to relax if scipy fails or invalid
    try:
        energy, relaxed = relax(positions, 'morse', rho)
        return energy, relaxed
    except:
        return float('inf'), positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse geometric templates
    templates = []
    
    # FCC is the most likely global minimum
    for seed in range(5):
        templates.append(_generate_fcc_cluster(n, seed))
        
    # BCC
    for seed in range(3):
        templates.append(_generate_bcc_cluster(n, seed))
        
    # Icosahedral
    for seed in range(3):
        templates.append(_generate_icosahedral_cluster(n, seed))
        
    # Random (just in case)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        radius = 1.5 * (n ** (1.0/3.0))
        pts = rng.uniform(-radius, radius, (n, 3))
        templates.append(pts)
        
    # Refine each template
    for t_pos in templates:
        if _get_remaining_time() < 10.0:
            break
            
        try:
            energy, refined_pos = _refine_structure(t_pos, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos.copy()
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback
        best_positions = _generate_fcc_cluster(n, 0)
        best_energy = 1e10
        
    # Final polish
    if _get_remaining_time() > 5.0:
        final_energy, final_pos = _refine_structure(best_positions, rho, max_iter=10000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
            
    return best_positions
# EVOLVE-BLOCK-END
