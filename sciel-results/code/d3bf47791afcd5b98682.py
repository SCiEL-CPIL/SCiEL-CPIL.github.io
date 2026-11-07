"""
Strategy: Deterministic Geometric Assembly (FCC-Shell Packing) + Local Refinement.

This approach abandons stochastic search (random starts, basin hopping) in favor of
exploiting the known physical structure of the global minima for Sutton-Chen
transition metal clusters. The global minima for these systems are typically
FCC or HCP truncated structures.

1. **Geometry Construction**: We generate a candidate structure by greedily placing
   atoms in an FCC lattice shell by shell, starting from the center. This ensures
   the initial configuration is already in the correct "basin of attraction" for
   the global minimum, avoiding the vast empty regions of configuration space
   explored by random starts.
   
2. **Local Refinement**: We use the provided `relax` function (L-BFGS-B) to
   optimize the geometry. Since the start is close to the global minimum,
   L-BFGS-B converges quickly to the precise local minimum, which is likely
   the global minimum.

3. **Validation & Safety**: We strictly check for minimum distances > 0.35.
   If the geometric construction fails (e.g., due to tight packing constraints
   for specific N), we fall back to a slightly expanded FCC lattice or a
   randomized FCC perturbation, ensuring validity while maintaining structural
   bias.

This is a structurally different approach: it is deterministic in its
initialization (no random seed dependence for the main path) and relies on
physics-informed geometry rather than stochastic exploration.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0
_START_TIME = None
_LOCK = threading.Lock() if False else None  # Will be initialized in construct

def _init_global():
    global _START_TIME, _LOCK
    _START_TIME = time.time()
    import threading
    _LOCK = threading.Lock()

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _generate_fcc_truncated(n, scale=1.0):
    """
    Generate a truncated FCC cluster of n atoms.
    We generate points in an FCC lattice, sort them by distance from the origin,
    and take the first n.
    
    To handle the 'truncation' cleanly, we generate a large enough supercell
    of FCC points, center it, and pick the n closest to the origin.
    """
    # FCC lattice vectors
    # a = 1.0 (lattice constant in reduced units, but we might need to scale)
    # The equilibrium nearest neighbor distance in Sutton-Chen is ~2^(1/n_exp) * a?
    # Actually, the potential is defined with r in reduced units.
    # For SC potential, min is at r_eq where 12/r^13 - 6c/r^7 = 0 => r_eq^6 = 2/c? No.
    # dE/dr = -12 r^-13 - c * (-1/2) * 2 * 6 r^-7 = -12 r^-13 + 6c r^-7
    # 12/r^13 = 6c/r^7 => 12 = 6c r^6 => r^6 = 2/c.
    # For c=144.41, r^6 = 2/144.41 ~ 0.0138 => r ~ 0.42.
    # Wait, the standard SC potential is usually scaled.
    # Let's check the provided potential form:
    # E = sum [ 1/2 sum r^-n - c sqrt(sum r^-m) ]
    # This is the many-body term.
    # The nearest neighbor distance in the global minimum is typically around 2.0-2.2 in "lattice constant" units if the lattice constant is defined differently.
    # However, the problem states "reduced units (lattice constant 1)".
    # If lattice constant a=1, FCC nearest neighbor distance is a/sqrt(2) ~ 0.707.
    # Let's assume the optimal packing uses a=1.
    
    # Generate a grid of FCC points.
    # FCC points can be generated as (i, j, k) where i,j,k are integers
    # such that i+j+k is even, scaled by 1/sqrt(2)?
    # Standard FCC primitive vectors:
    # a1 = (0.5, 0.5, 0)
    # a2 = (0.5, 0, 0.5)
    # a3 = (0, 0.5, 0.5)
    # Lattice constant a=1.
    # Nearest neighbor distance = 0.5 * sqrt(1) = 0.5? No.
    # Distance between (0,0,0) and (0.5, 0.5, 0) is sqrt(0.25+0.25) = sqrt(0.5) ~ 0.707.
    
    # Let's generate points on the FCC lattice with a=1.
    # We need to generate enough points.
    # Max n is 80.
    # We generate points in a cube of size L x L x L in lattice indices.
    # Each index step corresponds to a vector.
    # Let's use the conventional cubic cell representation.
    # Points are (x, y, z) where x,y,z are integers, and (x+y+z) is even.
    # These are the points of the FCC lattice if we scale by 0.5?
    # No, the points (0,0,0), (1,0,1), (0,1,1), (1,1,0) form an FCC lattice with a=1?
    # Distance between (0,0,0) and (1,0,1) is sqrt(2).
    # So if we use integer coords with even sum, the nearest neighbor is sqrt(2).
    # If we want nearest neighbor ~ 0.707 (which is 1/sqrt(2)), we should scale by 1/sqrt(2).
    # Let's assume the "lattice constant 1" implies the cubic unit cell edge is 1.
    # Then FCC points are at (0,0,0), (0.5, 0.5, 0), etc.
    # Nearest neighbor distance is 0.5 * sqrt(2) = 0.707.
    
    # Let's generate points in the form:
    # x = (i + j + k) / 2 ? No.
    # Simplest: Generate all (i, j, k) in [-L, L]^3.
    # Keep if (i+j+k) % 2 == 0.
    # These points form a BCC lattice if scaled? No.
    # The set {(i,j,k) | i+j+k even} is an FCC lattice rotated?
    # Let's just use the primitive vectors method to be safe.
    
    L = 10  # Half-range in lattice indices
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # Primitive vectors for FCC with a=1:
                # v1 = (0.5, 0.5, 0)
                # v2 = (0.5, 0, 0.5)
                # v3 = (0, 0.5, 0.5)
                # Point = i*v1 + j*v2 + k*v3
                x = 0.5 * (i + j)
                y = 0.5 * (i + k)
                z = 0.5 * (j + k)
                points.append([x, y, z])
                
    points = np.array(points)
    
    # Center the cluster at the origin
    # The point (0,0,0) is in the list (i=0,j=0,k=0).
    # So it is already centered.
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first n points
    selected_indices = sorted_indices[:n]
    cluster = points[selected_indices]
    
    # Apply scale factor to adjust density
    cluster = cluster * scale
    
    return cluster

def _min_distance(positions):
    n = len(positions)
    if n < 2:
        return float('inf')
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _worker(thread_id, n, kind, potential, initial_positions, results_container):
    """
    Worker: Takes an initial config, relaxes it, and optionally does a few
    small perturbations if time allows, to ensure we are at the local min.
    """
    rng = np.random.default_rng(42 + thread_id)
    current_pos = initial_positions.copy()
    
    # First, try to relax the initial geometry
    try:
        energy, positions = relax(current_pos, potential, kind)
    except Exception:
        return

    if not np.isfinite(energy):
        return

    min_d = _min_distance(positions)
    if min_d < 0.35:
        # If relaxation collapsed atoms, reject this path
        return

    current_energy = energy
    current_pos = positions

    # Update global best
    with _LOCK:
        if energy < results_container['best_energy']:
            results_container['best_energy'] = energy
            results_container['best_positions'] = positions.copy()

    # If time permits, do a few "refinement" steps with very small noise
    # This helps if the L-BFGS-B converged to a saddle or if the initial
    # geometry was slightly off the exact minimum.
    # However, since we start from FCC, we are likely already at the min.
    # We will just do 2-3 small perturbations to be safe.
    
    for i in range(3):
        if _time_left() < 5:
            break
        
        noise = rng.normal(0, 0.05, (n, 3))
        perturbed = current_pos + noise
        
        try:
            e2, p2 = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(e2):
            continue
            
        md2 = _min_distance(p2)
        if md2 < 0.35:
            continue
            
        if e2 < current_energy:
            current_energy = e2
            current_pos = p2
            with _LOCK:
                if e2 < results_container['best_energy']:
                    results_container['best_energy'] = e2
                    results_container['best_positions'] = p2.copy()

def construct(instance):
    global _START_TIME, _LOCK
    _init_global()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    results_container = {
        'best_energy': float('inf'),
        'best_positions': None
    }
    
    # Generate diverse FCC-based initial configurations
    # 1. Standard FCC (scale=1.0)
    # 2. Slightly expanded FCC (scale=1.05) to avoid initial collisions
    # 3. Slightly compressed FCC (scale=0.95) to test robustness
    # 4. HCP approximation? FCC is usually better for these sizes.
    #    We'll stick to FCC variants to maintain the "FCC motif" strategy.
    
    initial_configs = []
    
    # Config 1: Ideal FCC
    try:
        cfg1 = _generate_fcc_truncated(n, scale=1.0)
        # Check min dist
        if _min_distance(cfg1) > 0.35:
            initial_configs.append(cfg1)
        else:
            # Expand if too tight
            cfg1_exp = cfg1 * 1.2
            if _min_distance(cfg1_exp) > 0.35:
                initial_configs.append(cfg1_exp)
    except Exception:
        pass

    # Config 2: Expanded FCC
    try:
        cfg2 = _generate_fcc_truncated(n, scale=1.1)
        if _min_distance(cfg2) > 0.35:
            initial_configs.append(cfg2)
    except Exception:
        pass
        
    # Config 3: Compressed FCC
    try:
        cfg3 = _generate_fcc_truncated(n, scale=0.9)
        if _min_distance(cfg3) > 0.35:
            initial_configs.append(cfg3)
    except Exception:
        pass
        
    # Config 4: Randomized FCC (add small noise to break symmetry)
    try:
        cfg4 = _generate_fcc_truncated(n, scale=1.0)
        rng = np.random.default_rng(123)
        cfg4 += rng.normal(0, 0.05, (n, 3))
        if _min_distance(cfg4) > 0.35:
            initial_configs.append(cfg4)
    except Exception:
        pass

    # Fallback: If no valid FCC configs were generated (unlikely), use random
    if not initial_configs:
        rng = np.random.default_rng(0)
        R = 1.5 * n ** (1/3)
        cfg_rand = rng.uniform(-R, R, (n, 3))
        # Ensure min dist
        while _min_distance(cfg_rand) < 0.35:
            cfg_rand = rng.uniform(-R, R, (n, 3))
        initial_configs.append(cfg_rand)
        
    # Run parallel workers
    # We have up to 4 threads. We have up to 4 configs.
    # If we have fewer configs, we can duplicate or just run fewer.
    # Let's run as many as we have configs, up to 4.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, cfg in enumerate(initial_configs[:4]):
            future = executor.submit(_worker, i, n, kind, potential, cfg, results_container)
            futures.append(future)
            
        try:
            for f in concurrent.futures.as_completed(futures, timeout=_time_left()):
                f.result()
        except concurrent.futures.TimeoutError:
            pass
        except Exception:
            pass

    best_positions = results_container['best_positions']
    best_energy = results_container['best_energy']
    
    # Fallback if parallel execution failed
    if best_positions is None or not np.isfinite(best_energy):
        try:
            best_energy, best_positions = relax(initial_configs[0], potential, kind)
        except Exception:
            best_positions = initial_configs[0]
            
    # Final safety check
    if best_positions is not None:
        md = _min_distance(best_positions)
        if md < 0.35:
            # Try to fix by scaling
            scale_factor = 0.36 / md
            best_positions = best_positions * scale_factor
            md = _min_distance(best_positions)
            if md < 0.35:
                # If still invalid, return a very spread out random config
                rng = np.random.default_rng(999)
                best_positions = rng.uniform(-5, 5, (n, 3))
                # Ensure this is valid
                while _min_distance(best_positions) < 0.35:
                    best_positions = rng.uniform(-5, 5, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
