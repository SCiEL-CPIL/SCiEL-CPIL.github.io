"""Dzugutov cluster global-minima search using Mackay Icosahedral Seeds and Aggressive Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Safety margin below 300s

def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _icosahedron_vertices(scale=1.0):
    """Generate vertices of a regular icosahedron scaled to 'scale'."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _make_mackay_55(spacing=1.0):
    """
    Generate a 55-atom Mackay Icosahedron.
    Structure: 1 center + 12 first shell + 42 second shell.
    """
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius 'spacing'
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms
    # The second shell of a Mackay icosahedron consists of:
    # - 12 atoms at the vertices of a larger icosahedron
    # - 30 atoms at the centers of the faces of that larger icosahedron? 
    # Actually, the standard construction for the 42-atom shell in a Mackay icosahedron 
    # (specifically the one derived from the 147-atom structure) places atoms at:
    # 12 vertices of an icosahedron of radius R2 = spacing * phi (approx 1.618)
    # AND 30 atoms at the midpoints of the edges of that icosahedron? 
    # Let's check the geometry. 
    # In a Mackay icosahedron, the 42 atoms form the second shell.
    # A common construction:
    # R1 = 1.0
    # R2 = 1.618 (phi)
    # The 12 atoms are at the vertices of the icosahedron of radius R2.
    # The 30 atoms are at the centers of the 30 edges of that icosahedron? 
    # Let's verify the count: 12 vertices + 30 edge midpoints = 42.
    # This creates a structure with icosahedral symmetry.
    
    R2 = spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts.extend([p.copy() for p in shell2_verts])
    
    # Find edge midpoints of the icosahedron defined by shell2_verts
    # An icosahedron has 30 edges.
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
                
    if len(midpoints) == 30:
        pts.extend(midpoints)
    else:
        # Fallback: if geometry check fails, just add random points near surface
        # This is unlikely to happen with standard icosahedron vertices
        for _ in range(30):
            dir = np.random.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            p = dir * R2 * 0.9
            pts.append(p)
            
    pts = np.array(pts, dtype=float)
    return pts[:55]

def _make_seed(n, rng, base_spacing=1.0):
    """
    Generate a seed for n atoms based on the Mackay icosahedron.
    """
    # Start with the 55-atom Mackay icosahedron
    base = _make_mackay_55(base_spacing)
    
    if n == 55:
        return base.copy()
        
    if n < 55:
        # Remove atoms from the outer shell (indices 13-54)
        # Prefer removing from the outermost shell first to maintain core stability
        # Indices 0-12 are the core (1+12). Indices 13-54 are the outer shell (42).
        # We want to keep the core intact if possible.
        outer_indices = np.arange(13, 55)
        core_indices = np.arange(0, 13)
        
        num_remove = 55 - n
        
        if num_remove <= 42:
            to_remove = rng.choice(outer_indices, size=num_remove, replace=False)
        else:
            to_remove_outer = rng.choice(outer_indices, size=42, replace=False)
            remaining = num_remove - 42
            to_remove_core = rng.choice(core_indices, size=remaining, replace=False)
            to_remove = np.concatenate([to_remove_outer, to_remove_core])
            
        keep_mask = np.ones(55, dtype=bool)
        keep_mask[to_remove] = False
        return base[keep_mask]
        
    if n > 55:
        # Add atoms near the surface
        num_add = n - 55
        new_pts = []
        # Surface radius is approx R2 = spacing * 1.618
        R_surface = base_spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
        
        for _ in range(num_add):
            # Pick a random direction
            dir = rng.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            # Place slightly outside the current surface
            r = R_surface * (1.0 + rng.uniform(0.0, 0.2))
            p = dir * r
            new_pts.append(p)
            
        return np.vstack([base, np.array(new_pts)])
        
    # Fallback for other n (should not happen for this task)
    radius = base_spacing * n ** (1.0/3.0)
    pts = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    # The task contract says: instance = {'potential': 'dzugutov', 'n': atoms, 'A', 'B', 'a', 'b', 'c', 'd', 'm'}
    # relax(positions, 'dzugutov') is the signature.
    # The current code uses parameter = instance.get("rho", instance.get("kind")).
    # Let's stick to passing 'dzugutov' as the potential name to relax, as per the prompt: 
    # "chem_clusters.relax(positions, 'dzugutov')"
    # However, the seed program passes `parameter`. 
    # Looking at the prompt: "chem_clusters.relax(positions, 'dzugutov')"
    # It seems the second argument is the potential name.
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Time management
    # We have 6 instances. 300s total.
    # Let's assume we get called sequentially.
    # We should cap the time per instance to ensure we don't blow the budget.
    # If we are the first instance, we have ~290s.
    # If we are the last, we have ~0s.
    # Strategy: Use a fixed budget per instance, e.g., 45s, but check remaining time.
    # If remaining < 5s, return best found immediately.
    
    instance_budget = 45.0  # Seconds per instance
    if _remaining_time() < instance_budget + 10.0:
        # If we are running out of time for all instances, compress
        instance_budget = max(5.0, _remaining_time() - 2.0)

    rng = np.random.default_rng(42 + size)
    
    # Generate diverse seeds
    # 1. Standard Mackay
    # 2. Slightly compressed Mackay
    # 3. Slightly expanded Mackay
    # 4. Randomized Mackay (jittered)
    
    seeds = []
    spacings = [0.95, 1.0, 1.05, 1.1]
    for sp in spacings:
        seed = _make_seed(size, rng, base_spacing=sp)
        seeds.append(seed)
        
    # Add a few random seeds just in case
    for _ in range(2):
        radius = 1.0 * size ** (1.0/3.0)
        pts = rng.uniform(-radius, radius, (size, 3))
        seeds.append(pts)
        
    # Relax all seeds first
    relaxed_seeds = []
    for seed in seeds:
        if _remaining_time() < 5.0 or (time.time() - t0) > instance_budget - 5.0:
            break
        try:
            energy, positions = relax(seed, 'dzugutov')
            if np.all(np.isfinite(energy)) and np.all(np.isfinite(positions)):
                # Check minimum distance constraint
                min_dist = np.inf
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist >= 0.5:
                    relaxed_seeds.append((energy, positions))
        except Exception:
            continue
            
    if not relaxed_seeds:
        # Fallback: return a valid random structure if relaxation fails
        # This is a last resort
        pts = rng.uniform(-1, 1, (size, 3))
        # Simple local relaxation attempt
        try:
            _, pts = relax(pts, 'dzugutov')
        except:
            pass
        return pts

    # Sort by energy
    relaxed_seeds.sort(key=lambda x: x[0])
    best_energy, best_positions = relaxed_seeds[0]
    
    # Basin Hopping from the best seeds
    # We will run BH from the top 3 seeds
    top_seeds = relaxed_seeds[:min(3, len(relaxed_seeds))]
    
    bh_rng = np.random.default_rng(123 + size)
    
    for seed_energy, seed_pos in top_seeds:
        if _remaining_time() < 5.0 or (time.time() - t0) > instance_budget:
            break
            
        current_energy = seed_energy
        current_pos = seed_pos.copy()
        local_best_energy = seed_energy
        local_best_pos = seed_pos.copy()
        
        # BH parameters
        # Start with a small scale, increase if stuck
        scale = 0.1
        T = 0.05
        stuck_count = 0
        max_steps = 500 # Hard cap on steps per seed to avoid infinite loops if time check is slow
        
        while _remaining_time() > 5.0 and (time.time() - t0) < instance_budget and stuck_count < 100:
            # Perturb
            perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
            
            try:
                energy, positions = relax(perturbed, 'dzugutov')
            except Exception:
                continue
                
            if not np.all(np.isfinite(energy)):
                continue
                
            # Check min distance
            min_dist = np.inf
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, d)
            if min_dist < 0.5:
                # Invalid configuration, reject
                energy = np.inf
                
            # Metropolis acceptance
            accept = False
            if energy < current_energy:
                accept = True
            else:
                delta_e = energy - current_energy
                if delta_e < 10.0:
                    if bh_rng.random() < np.exp(-delta_e / T):
                        accept = True
                        
            if accept:
                current_energy = energy
                current_pos = positions.copy()
                stuck_count = 0
                
                if energy < local_best_energy:
                    local_best_energy = energy
                    local_best_pos = positions.copy()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
            else:
                stuck_count += 1
                # Adjust scale
                if stuck_count % 10 == 0:
                    scale *= 1.1 # Increase scale to escape
                    if scale > 0.5:
                        scale = 0.5
                    T *= 0.95
                    if T < 0.01:
                        T = 0.01

    return best_positions
# EVOLVE-BLOCK-END
