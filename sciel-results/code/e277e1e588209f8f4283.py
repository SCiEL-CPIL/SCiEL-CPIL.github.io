"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    # Pairwise distances using broadcasting
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(centres)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _objective(centres_flat):
    """Objective function to minimize (negative radius)."""
    centres = centres_flat.reshape(-1, 3)
    # Check validity
    if not np.all(np.isfinite(centres)):
        return 1e10
    if np.any(np.abs(centres) > 0.5 + 1e-6):
        return 1e10
    r = _compute_radius(centres)
    return -r


def _local_optimize(centres, time_limit=10.0):
    """Run local optimization to improve the packing."""
    start_time = time.time()
    n = len(centres)
    centres = centres.copy()
    
    # Flatten centres for scipy
    x0 = centres.flatten()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple optimization methods with time budget
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() - start_time > time_limit:
            break
            
        try:
            # Use tighter tolerances for better convergence
            result = minimize(
                _objective, x0, method=method,
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            if result.fun < 0:  # Valid solution
                new_centres = result.x.reshape(-1, 3)
                new_radius = _compute_radius(new_centres)
                
                if new_radius > best_radius:
                    best_centres = new_centres
                    best_radius = new_radius
        except Exception:
            pass
    
    return best_centres


def _initial_grid():
    """Generate initial cubic grid 5x5x4 = 100."""
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N]


def _initial_fcc_like():
    """Create an FCC-inspired initial configuration."""
    # FCC structure has higher density than simple cubic
    # Use a 4x4x4 FCC-like arrangement with body-centred layers
    centres = []
    
    # Layer structure: alternate between face-centred and body-centred
    # For 100 points, use 4 layers of 25 each
    layer_size = 5
    num_layers = 4
    
    for layer in range(num_layers):
        z = -0.5 + 0.5 * (layer + 0.5) / num_layers
        for i in range(layer_size):
            for j in range(layer_size):
                x = -0.5 + 0.5 * (i + 0.5) / layer_size
                y = -0.5 + 0.5 * (j + 0.5) / layer_size
                # Offset every other layer for FCC-like structure
                if layer % 2 == 1:
                    x += 0.5 / layer_size
                    y += 0.5 / layer_size
                centres.append([x, y, z])
    
    return np.array(centres[:N])


def _random_initialization(seed=42):
    """Random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def construct():
    """Construct the optimal packing of 100 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with cubic grid
    centres = _initial_grid()
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple initializations and optimize each
    initializations = [
        ('grid', _initial_grid()),
        ('fcc', _initial_fcc_like()),
    ]
    
    # Add random initializations with different seeds
    for seed in [0, 1, 2, 3, 42, 123, 777, 2024]:
        if time.time() > deadline - 20:
            break
        initializations.append((f'random_{seed}', _random_initialization(seed=seed)))
    
    # Evaluate and improve each initialization
    for name, centres in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        r = _compute_radius(centres)
        if r > best_radius:
            best_centres = centres.copy()
            best_radius = r
        
        # Try to improve with local optimization
        remaining_time = deadline - elapsed - 5
        if remaining_time > 5:
            try:
                optimized = _local_optimize(centres, time_limit=remaining_time)
                opt_radius = _compute_radius(optimized)
                
                if opt_radius > best_radius:
                    best_centres = optimized
                    best_radius = opt_radius
            except Exception:
                pass
    
    # Final refinement on the best solution
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        try:
            final_time = deadline - elapsed - 1
            if final_time > 5:
                final_centres = _local_optimize(best_centres, time_limit=final_time)
                final_radius = _compute_radius(final_centres)
                
                if final_radius > best_radius:
                    best_centres = final_centres
                    best_radius = final_radius
        except Exception:
            pass
    
    # Ensure validity: clip to cube boundaries
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
# EVOLVE-BLOCK-END
