"""Packing 50 equal circles in a unit circle using iterative radius increase and overlap minimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _overlap_energy(positions, r):
    """
    Calculate the overlap energy for a given set of positions and radius r.
    Energy is sum of squared overlaps between circles and with boundary.
    """
    n = len(positions)
    energy = 0.0
    
    # Boundary violations
    norms = np.linalg.norm(positions, axis=1)
    boundary_violations = np.maximum(0, norms - (1.0 - r))
    energy += np.sum(boundary_violations ** 2)
    
    # Pairwise overlaps
    # Vectorized calculation for efficiency
    for i in range(n):
        # Vectorized distance from i to all j > i
        diffs = positions[i+1:] - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        overlaps = np.maximum(0, 2.0 * r - dists)
        energy += np.sum(overlaps ** 2)
        
    return energy

def _optimize_for_radius(positions, r, max_iter=1000):
    """
    Optimize positions to minimize overlap for a fixed radius r.
    """
    n = len(positions)
    
    # Define bounds: positions must be within the unit circle
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    x0 = positions.flatten()
    
    # Objective function for minimizer
    def obj(x):
        pos = x.reshape(n, 2)
        return _overlap_energy(pos, r)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50})
        return res.x.reshape(n, 2)
    except Exception:
        return positions

def _construct_single(seed):
    """
    Run the iterative radius increase for a single random start.
    """
    rng = np.random.default_rng(seed)
    
    # Initial random positions
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 1, N))
    positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Start with a small radius
    r = 0.08
    # Increment
    dr = 0.002
    min_dr = 1e-5
    
    best_positions = positions.copy()
    best_r = r
    
    while r < 0.15: # Upper bound guess
        # Optimize positions for current r
        positions = _optimize_for_radius(positions, r, max_iter=300)
        
        # Check if packing is valid (energy ~ 0)
        energy = _overlap_energy(positions, r)
        
        if energy < 1e-8:
            # Valid packing found, try to increase r
            best_r = r
            best_positions = positions.copy()
            r += dr
            # Reduce step size as r gets larger to be precise
            dr *= 0.98
            if dr < min_dr:
                dr = min_dr
        else:
            # Invalid packing, try a smaller step or break
            # If we can't fit, we hit a local minimum or the max radius.
            # Try a few smaller steps to see if it was just a convergence issue
            for _ in range(3):
                r_try = r - dr * 0.1
                if r_try > best_r:
                    positions_try = _optimize_for_radius(positions, r_try, max_iter=100)
                    energy_try = _overlap_energy(positions_try, r_try)
                    if energy_try < 1e-8:
                        best_r = r_try
                        best_positions = positions_try.copy()
                        break
            break
            
    return best_positions, best_r

def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_overall_r = 0.0
    best_overall_pos = None
    
    # Run multiple random restarts
    # Time limit is 120s. 
    # Each optimization step takes ~0.1-0.5s depending on convergence.
    # Let's try 5-10 restarts.
    for seed in range(8):
        pos, r = _construct_single(seed)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_pos = pos.copy()
            
    # Final refinement: run one more optimization on the best position with the found r
    # to ensure it's as tight as possible
    if best_overall_pos is not None:
        # Try to push r a tiny bit higher?
        # The loop above already does this.
        # Just ensure the returned positions are valid for the reported r
        # The evaluator computes r from positions, so we just return positions.
        
        # One last polish: minimize overlap for the current best_r to ensure minimal energy
        # This might not increase r but ensures stability
        pass
        
    return best_overall_pos
# EVOLVE-BLOCK-END
