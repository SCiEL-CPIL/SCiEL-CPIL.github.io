"""Rewritten Dzugutov cluster global-minima search with Mackay Icosahedral Seeds and Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin below 300s (evaluator kills at 330s)

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
    # Normalize to unit sphere then scale
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _mackay_icosahedron_55(spacing=1.0):
    """
    Generate a 55-atom Mackay Icosahedron.
    Structure: 1 center + 12 first shell + 42 second shell.
    The first shell is at distance 'spacing'.
    The second shell is at distance 'spacing * phi' (approximately 1.618 * spacing).
    """
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius 'spacing'
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms
    # The 42 atoms consist of:
    # 12 atoms at the vertices of a larger icosahedron (radius R2 = spacing * phi)
    # 30 atoms at the midpoints of the edges of that larger icosahedron? 
    # Actually, the standard Mackay icosahedron construction for the second shell (42 atoms) 
    # places atoms at the vertices (12) and face centers (20) and edge midpoints (10)? No.
    # Let's use the known coordinates or a robust construction.
    # A robust way: The 55-atom icosahedron is a subset of the 147-atom one.
    # Alternatively, we can generate the 42 atoms as follows:
    # 12 atoms at the vertices of an icosahedron of radius R2.
    # 30 atoms at the centers of the faces of an icosahedron of radius R2? No, icosahedron has 20 faces.
    # The 42 atoms are: 12 vertices + 30 edge midpoints of an icosahedron with radius R2.
    # Let's verify the count: 12 + 30 = 42. Correct.
    
    R2 = spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts.extend([p.copy() for p in shell2_verts])
    
    # Find edge midpoints of the icosahedron defined by shell2_verts
    # The edge length of an icosahedron with circumradius R is R * sqrt(2 + 2/phi) ?
    # Let's just compute distances between vertices to find edges.
    # In a regular icosahedron, the nearest neighbors are the edges.
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
                
    # There should be 30 midpoints
    if len(midpoints) == 30:
        pts.extend(midpoints)
    else:
        # Fallback if construction fails (should not happen)
        # Add random points near the surface
        for _ in range(42 - len(pts)):
            dir = np.random.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            p = dir * R2
            pts.append(p)
        
    pts = np.array(pts, dtype=float)
    return pts

def _make_seed(n, rng):
    """
    Generate a seed for n atoms based on the 55-atom Mackay icosahedron.
    """
    # Use a standard spacing, maybe slightly varied
    spacing = 1.0
    
    if n == 55:
        return _mackay_icosahedron_55(spacing)
    
    base_pts = _mackay_icosahedron_55(spacing)
    
    if n < 55:
        # Remove atoms from the outer shell (indices 13 to 54)
        # We want to remove (55 - n) atoms
        num_remove = 55 - n
        # Prefer removing from the outer shell to maintain the core
        # Outer shell indices: 13-54 (42 atoms)
        # If we need to remove more than 42, remove from shell 1 (12 atoms) and center (1 atom)
        
        # Shuffle indices to randomize which atoms are removed
        indices = np.arange(55)
        rng.shuffle(indices)
        
        # Prioritize removing outer shell atoms? 
        # Actually, for global minima, different isomers are possible.
        # Random removal from the whole structure might be too destructive.
        # Let's try removing from the outer shell first.
        outer_indices = np.arange(13, 55)
        core_indices = np.arange(0, 13)
        
        to_remove = []
        if num_remove <= 42:
            to_remove = rng.choice(outer_indices, size=num_remove, replace=False)
        else:
            # Need to remove from core too
            to_remove_outer = rng.choice(outer_indices, size=42, replace=False)
            remaining_remove = num_remove - 42
            to_remove_core = rng.choice(core_indices, size=remaining_remove, replace=False)
            to_remove = np.concatenate([to_remove_outer, to_remove_core])
            
        keep_mask = np.ones(55, dtype=bool)
        keep_mask[to_remove] = False
        return base_pts[keep_mask]
        
    else: # n > 55
        # Add atoms to the surface
        num_add = n - 55
        new_pts = []
        R2 = spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
        
        for _ in range(num_add):
            # Pick a random direction
            dir = rng.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            # Pick a radius slightly larger than the shell
            # The surface is roughly at R2. New atoms should be close to existing ones.
            # Let's place them at a distance that forms a bond with some surface atom.
            # A simple heuristic: place at R2 * (1 + small_noise)
            r = R2 * (1.0 + rng.uniform(0.0, 0.2))
            p = dir * r
            new_pts.append(p)
            
        pts = np.vstack([base_pts, np.array(new_pts)])
        return pts

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    
    # 1. Generate Seeds
    # We want diverse seeds. Let's generate ~10-20 seeds.
    # Time budget for seeds: ~10% of total?
    # Relaxation is expensive. Let's limit the number of seeds to what we can relax in ~20s.
    # Assume relaxation takes ~0.5s per seed. 20s / 0.5s = 40 seeds.
    # But BH is also needed.
    
    # Let's allocate 30% of time to initial relaxation, 70% to BH.
    # Or just generate a fixed number of seeds, relax them, and use remaining time for BH.
    
    num_seeds = 15
    seeds = []
    rng = np.random.default_rng(42)
    
    for i in range(num_seeds):
        # Vary the seed generation slightly
        # For n=55, just use the perfect icosahedron for the first seed
        if i == 0 and size == 55:
            seed = _mackay_icosahedron_55(1.0)
        else:
            seed = _make_seed(size, rng)
        seeds.append(seed)
        
    # 2. Relax Seeds
    relaxed_seeds = []
    for seed in seeds:
        if _remaining_time() < 5.0:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            # Check validity
            if np.all(np.isfinite(energy)) and np.all(np.isfinite(positions)):
                relaxed_seeds.append((energy, positions))
        except Exception:
            continue
            
    if not relaxed_seeds:
        # Fallback: return a random relaxed seed
        seed = _make_seed(size, rng)
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = _make_seed(size, rng)
        return best_positions

    # Sort by energy
    relaxed_seeds.sort(key=lambda x: x[0])
    best_energy, best_positions = relaxed_seeds[0]
    
    # 3. Basin Hopping
    # Start from the best few relaxed seeds
    # Use the top 3 seeds as starting points for BH chains
    
    bh_rng = np.random.default_rng(123)
    
    # Determine BH budget
    bh_budget = _remaining_time() - 5.0  # Leave 5s for safety
    
    if bh_budget > 0:
        # We will run BH from the best seed, and potentially the 2nd and 3rd if time permits.
        # Let's split the budget among the top 3 seeds.
        top_seeds = relaxed_seeds[:3]
        
        for seed_energy, seed_pos in top_seeds:
            if _remaining_time() < 5.0:
                break
            
            current_energy = seed_energy
            current_pos = seed_pos
            local_best_energy = seed_energy
            local_best_pos = seed_pos
            
            # BH loop
            # Perturbation scale: start at 0.2, decrease slowly?
            # Or keep constant?
            scale = 0.2
            T = 0.1 # Metropolis temperature
            
            while _remaining_time() > 5.0:
                # Perturb
                perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
                
                try:
                    energy, positions = relax(perturbed, potential, parameter)
                except Exception:
                    continue
                    
                if not np.all(np.isfinite(energy)):
                    continue
                    
                # Metropolis acceptance
                if energy < current_energy or bh_rng.random() < np.exp(-(energy - current_energy) / T):
                    current_energy = energy
                    current_pos = positions
                    
                    if energy < local_best_energy:
                        local_best_energy = energy
                        local_best_pos = positions
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                            
                # Occasionally reduce scale to refine
                # scale *= 0.99
                # But this might get stuck. Let's keep scale constant or slightly increase if stuck.
                # For simplicity, keep constant.
                
    return best_positions
# EVOLVE-BLOCK-END
