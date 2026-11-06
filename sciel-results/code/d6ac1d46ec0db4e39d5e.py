# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _fcc_lattice_points(n, rng):
    """Generate n points from an FCC lattice, closest to origin."""
    points = []
    a = 1.0  # lattice constant (equilibrium distance)
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    # Generate lattice points in a cube around origin
    max_coord = int(np.ceil(n ** (1.0 / 3.0)) + 2)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    # Remove duplicates (shouldn't be any but just in case)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    # Take n closest
    if len(points) >= n:
        return points[:n].copy()
    else:
        # Pad with random points if needed
        extra = np.random.randn(n - len(points), 3) * 0.5
        return np.vstack([points, extra])


def _bcc_lattice_points(n, rng):
    """Generate n points from a BCC lattice, closest to origin."""
    points = []
    a = 1.0
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    max_coord = int(np.ceil(n ** (1.0 / 3.0)) + 2)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    if len(points) >= n:
        return points[:n].copy()
    else:
        extra = np.random.randn(n - len(points), 3) * 0.5
        return np.vstack([points, extra])


def _hcp_lattice_points(n, rng):
    """Generate n points from an HCP lattice, closest to origin."""
    points = []
    a = 1.0
    c = 1.633 * a  # ideal HCP c/a ratio
    # HCP basis in hexagonal coordinates, convert to Cartesian
    # Layer A: (0,0,0), (1,0,0), (0,1,0), (1,1,0) in hex coords
    # Layer B: (1/3, 2/3, 0.5), (2/3, 1/3, 0.5), etc.
    # Simpler: use a cubic approximation
    # Actually, let's just use a slightly perturbed fcc
    pts = _fcc_lattice_points(n, rng)
    # Add small perturbation
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts


def _random_compact(n, rng, min_dist=0.8):
    """Generate n random points in a sphere with minimum distance constraint."""
    radius = n ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
    if len(points) < n:
        # Pad with more random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _min_dist_ok(pos, min_d=0.5):
    """Check all pairwise distances >= min_d."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    t_start = time.time()
    deadline = t_start + 45  # per-instance budget, 6*45=270 < 300
    
    rng = np.random.default_rng(size * 7 + int(rho))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. FCC lattice
    try:
        init_configs.append(_fcc_lattice_points(size, rng))
    except:
        pass
    
    # 2. BCC lattice
    try:
        init_configs.append(_bcc_lattice_points(size, rng))
    except:
        pass
    
    # 3. HCP-like (perturbed FCC)
    try:
        init_configs.append(_hcp_lattice_points(size, rng))
    except:
        pass
    
    # 4-6. Random compact with different min distances
    for md in [0.8, 0.7, 0.6]:
        try:
            init_configs.append(_random_compact(size, rng, min_dist=md))
        except:
            pass
    
    # 7-8. Perturbed FCC with different noise levels
    for noise in [0.1, 0.2]:
        try:
            pts = _fcc_lattice_points(size, rng)
            pts += rng.normal(0, noise, size=pts.shape)
            init_configs.append(pts)
        except:
            pass
    
    # Relax each initial configuration
    for idx, init_pos in enumerate(init_configs):
        if time.time() > deadline - 5:
            break
        
        # Ensure valid minimum distance before relaxation
        if not _min_dist_ok(init_pos, 0.5):
            # Try to fix by scaling
            scale = 1.0
            while not _min_dist_ok(init_pos * scale, 0.5) and scale < 3.0:
                scale += 0.1
            init_pos = init_pos * scale
        
        try:
            energy, positions = relax(init_pos.copy(), potential, rho)
            if energy < best_energy and _min_dist_ok(positions, 0.5):
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    # If no valid result, try a few more random starts
    if best_positions is None:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            init_pos = _random_compact(size, rng, min_dist=0.55)
            try:
                energy, positions = relax(init_pos.copy(), potential, rho)
                if _min_dist_ok(positions, 0.5):
                    best_energy = energy
                    best_positions = positions.copy()
            except:
                pass
    
    if best_positions is None:
        # Fallback: return FCC lattice points (should be valid)
        best_positions = _fcc_lattice_points(size, rng)
        # Ensure min distance
        while not _min_dist_ok(best_positions, 0.5):
            best_positions *= 1.05
    
    return best_positions
# EVOLVE-BLOCK-END
