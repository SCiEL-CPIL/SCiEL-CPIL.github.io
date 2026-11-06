"""Combined program: Vectorized lattice init + robust time management + hybrid strategies."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 295.0  # 5s safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = 0

def _generate_fcc_lattice(n, spacing=1.0):
    """
    Generate n points from an FCC lattice centered at origin using vectorization.
    FCC points in a cubic supercell of side L:
    Basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) * a
    We generate a grid of integer coordinates (i,j,k) in [-L, L]
    and apply the basis vectors.
    """
    # Estimate required grid size
    # Volume of sphere ~ (4/3) pi R^3. Number of points ~ n.
    # FCC density is ~ 0.74 * (1/spacing^3) per unit volume?
    # Simpler: Just generate a large enough cube and filter.
    r_max = spacing * (n ** (1.0 / 3.0) + 3.0)
    # In units of spacing/2, the range is approx 2 * r_max / spacing
    limit = int(2 * r_max / spacing) + 2
    
    # Create grid of integer indices
    # To save memory, we generate one basis at a time or use a compact loop
    # For n=80, limit is small. Vectorizing the full 3D grid is fine.
    
    # We want points (x, y, z) such that they belong to FCC.
    # FCC = { (i, j, k) * (spacing/2) | i,j,k integers, i+j+k even }
    
    i = np.arange(-limit, limit + 1)
    j = np.arange(-limit, limit + 1)
    k = np.arange(-limit, limit + 1)
    
    # Generate all combinations
    I, J, K = np.meshgrid(i, j, k, indexing='ij')
    
    # Filter: i + j + k must be even
    mask = (I + J + K) % 2 == 0
    
    # Extract valid points
    I_valid = I[mask]
    J_valid = J[mask]
    K_valid = K[mask]
    
    # Scale to real space
    scale = spacing / 2.0
    points = np.column_stack((I_valid * scale, J_valid * scale, K_valid * scale))
    
    # Filter by distance from origin
    dists = np.linalg.norm(points, axis=1)
    # Keep only points within a sphere of radius r_max
    sphere_mask = dists <= r_max
    
    points = points[sphere_mask]
    
    if len(points) < n:
        # Fallback if grid was too small (unlikely with +3 margin)
        # Increase limit and retry once
        limit += 5
        i = np.arange(-limit, limit + 1)
        j = np.arange(-limit, limit + 1)
        k = np.arange(-limit, limit + 1)
        I, J, K = np.meshgrid(i, j, k, indexing='ij')
        mask = (I + J + K) % 2 == 0
        I_valid = I[mask]
        J_valid = J[mask]
        K_valid = K[mask]
        scale = spacing / 2.0
        points = np.column_stack((I_valid * scale, J_valid * scale, K_valid * scale))
        dists = np.linalg.norm(points, axis=1)
        sphere_mask = dists <= r_max
        points = points[sphere_mask]
        
    if len(points) < n:
        # Absolute fallback: random
        rng = np.random.default_rng(42)
        points = rng.uniform(-r_max, r_max, (n, 3))
        return points

    # Sort by distance from origin to get the most compact cluster
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    selected = points[sorted_idx[:n]]
    
    # Center the cluster
    selected = selected - np.mean(selected, axis=0)
    
    return selected

def _generate_bcc_lattice(n, spacing=1.0):
    """
    Generate n points from a BCC lattice centered at origin using vectorization.
    BCC = { (i, j, k) * (spacing/2) | i,j,k integers, i,j,k have same parity }
    """
    r_max = spacing * (n ** (1.0 / 3.0) + 3.0)
    limit = int(2 * r_max / spacing) + 2
    
    i = np.arange(-limit, limit + 1)
    j = np.arange(-limit, limit + 1)
    k = np.arange(-limit, limit + 1)
    
    I, J, K = np.meshgrid(i, j, k, indexing='ij')
    
    # Parity check: (I % 2) == (J % 2) == (K % 2)
    # In numpy, negative mod can be tricky, so use abs or bitwise?
    # Python's % returns non-negative for positive divisor.
    # -1 % 2 = 1. So -1 is "odd" in this context.
    # 0 % 2 = 0.
    # So we can just compare I % 2, J % 2, K % 2.
    mask = (I % 2 == J % 2) & (J % 2 == K % 2)
    
    I_valid = I[mask]
    J_valid = J[mask]
    K_valid = K[mask]
    
    scale = spacing / 2.0
    points = np.column_stack((I_valid * scale, J_valid * scale, K_valid * scale))
    
    dists = np.linalg.norm(points, axis=1)
    sphere_mask = dists <= r_max
    points = points[sphere_mask]
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        points = rng.uniform(-r_max, r_max, (n, 3))
        return points

    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    selected = points[sorted_idx[:n]]
    selected = selected - np.mean(selected, axis=0)
    
    return selected

def _generate_random_close_packing(n, spacing=1.0, seed=None):
    """
    Generate n random points with a minimum distance constraint.
    This avoids the "high energy start" problem of pure random uniform.
    """
    if seed is None:
        seed = np.random.randint(10000)
    rng = np.random.default_rng(seed)
    
    # Estimate radius
    r_max = spacing * (n ** (1.0 / 3.0) * 1.2 + 2.0)
    
    points = []
    min_dist = 0.6 * spacing # Slightly less than 1.0 to allow relaxation to tighten
    
    # Try to place points one by one
    # For large N, this can be slow. Limit attempts.
    max_attempts = n * 50
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) > r_max:
            attempts += 1
            continue
            
        # Check distance to existing points
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
        
        points.append(p)
        
    if len(points) < n:
        # If we couldn't place enough, fill with random
        remaining = n - len(points)
        extra = rng.uniform(-r_max, r_max, (remaining, 3))
        points.extend([p for p in extra])
        
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def construct(instance):
    global _INSTANCE_INDEX
    _INSTANCE_INDEX += 1
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time Management
    # We have 6 instances. We don't know how much time we've used yet.
    # Assume equal distribution, but adjust if we are behind.
    elapsed = time.time() - _START_TIME
    remaining_total = _HARD_DEADLINE - time.time()
    
    # Heuristic time budget per instance
    # Larger clusters take longer to relax.
    # Base time + time per atom.
    base_time = 5.0
    per_atom_time = 0.5
    estimated_instance_time = base_time + per_atom_time * size
    
    # If we have more time than average, use it.
    # If we are behind, shrink.
    avg_remaining_per_instance = remaining_total / max(1, _INSTANCE_COUNT - _INSTANCE_INDEX + 1)
    
    # We want to finish this instance in a fraction of the remaining time
    # to be safe. Let's aim for 80% of the allocated slot.
    time_budget = min(estimated_instance_time, avg_remaining_per_instance * 0.8)
    # Ensure a minimum of 2 seconds to actually do something
    time_budget = max(2.0, time_budget)
    
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    # 1. FCC
    # 2. BCC
    # 3. Random Close Packing
    # 4. Random Close Packing (different seed)
    
    starts = []
    
    # FCC
    try:
        starts.append(_generate_fcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # BCC
    try:
        starts.append(_generate_bcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # Random Close Packing
    try:
        starts.append(_generate_random_close_packing(size, spacing=1.0, seed=size))
    except Exception:
        pass
        
    try:
        starts.append(_generate_random_close_packing(size, spacing=1.0, seed=size + 100))
    except Exception:
        pass
        
    # If we have time, add a pure random start (might find different basins)
    if time.time() - start_time < time_budget * 0.5:
        try:
            rng = np.random.default_rng(size + _INSTANCE_INDEX)
            radius = 1.0 * size ** (1.0 / 3.0) * 0.8
            points = rng.uniform(-radius, radius, (size, 3))
            starts.append(points)
        except Exception:
            pass
            
    # Relax each start
    for idx, points in enumerate(starts):
        if time.time() - start_time > time_budget:
            break
            
        # Add small jitter to break symmetry
        rng_jitter = np.random.default_rng(idx + size)
        points = points + rng_jitter.normal(0, 0.01, points.shape)
        
        try:
            energy, positions = relax(points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
    # If we still have time, do a few more random restarts
    # This is crucial for escaping local minima.
    while time.time() - start_time < time_budget * 0.9:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        radius = 1.0 * size ** (1.0 / 3.0) * 0.8
        points = rng.uniform(-radius, radius, (size, 3))
        
        try:
            energy, positions = relax(points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        # Fallback: return random points if everything failed
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (size, 3))
        
    return best_positions

# EVOLVE-BLOCK-END
