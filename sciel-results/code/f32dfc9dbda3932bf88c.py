"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _pairwise_distances(c):
    """c: (N,2) array of centers. Returns (N,N) distance matrix."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _objective_and_constraints(x):
    """
    x: (3N,) array, first 2N are centers, last N are radii.
    Returns (value, feasibility_flag, penalty)
    We maximize sum of radii subject to:
      - circles inside [0,1]^2: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - disjoint interiors: r_i + r_j <= dist(i,j)
    We use a penalty-based approach: minimize negative sum of radii + large penalty for violations.
    """
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Clip to ensure basic feasibility for gradient calculation
    centers_clipped = np.clip(centers, 0.0, 1.0)
    radii_clipped = np.clip(radii, 1e-6, 0.5)
    
    # Compute distances
    dists = _pairwise_distances(centers_clipped)
    
    # Boundary violations: r_i - x_i, r_i - (1-x_i), etc.
    # We want r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # Violation = max(0, r_i - x_i) + max(0, r_i - (1-x_i)) + ...
    bnd_viol = (np.maximum(0, radii_clipped - centers_clipped[:, 0]) +
                np.maximum(0, radii_clipped - (1 - centers_clipped[:, 0])) +
                np.maximum(0, radii_clipped - centers_clipped[:, 1]) +
                np.maximum(0, radii_clipped - (1 - centers_clipped[:, 1])))
    
    # Pairwise violations: r_i + r_j - dist(i,j)
    # dists is symmetric, ignore diagonal
    r_sum = radii_clipped[:, np.newaxis] + radii_clipped[np.newaxis, :]
    pair_viol = np.maximum(0, r_sum - dists)
    # Set diagonal to 0
    np.fill_diagonal(pair_viol, 0)
    pair_viol_sum = np.sum(pair_viol)
    
    total_violation = np.sum(bnd_viol) + pair_viol_sum
    objective = -np.sum(radii_clipped) + 100.0 * total_violation
    
    return objective, total_violation

def _construct_from_x(x):
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    # Ensure strict interior
    radii = np.clip(radii, 1e-6, 0.5)
    centers = np.clip(centers, 0.0, 1.0)
    # Adjust radii to fit boundaries
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r * 0.999
    
    result = np.zeros((N, 3))
    result[:, :2] = centers
    result[:, 2] = radii
    return result

def _validate_and_fix(result):
    """Ensure all circles are strictly inside and disjoint."""
    for _ in range(10):
        valid = True
        for i in range(N):
            x, y, r = result[i]
            # Boundary check
            if x < r or x + r > 1 or y < r or y + r > 1:
                # Shrink radius
                max_r = min(x, 1 - x, y, 1 - y)
                if r > max_r:
                    result[i, 2] = max_r * 0.999
                    valid = False
        
        # Pairwise check
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = result[i]
                xj, yj, rj = result[j]
                d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if d < ri + rj:
                    # Shrink the larger radius or both
                    excess = ri + rj - d
                    if excess > 0:
                        # Reduce both radii slightly
                        result[i, 2] -= excess / 2
                        result[j, 2] -= excess / 2
                        valid = False
                        # Ensure non-negative
                        result[i, 2] = max(result[i, 2], 1e-6)
                        result[j, 2] = max(result[j, 2], 1e-6)
        
        if valid:
            break
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple random restarts
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random positions and small radii
        x0 = np.random.rand(3*N)
        x0[:2*N] = np.random.rand(N, 2).flatten()
        x0[2*N:] = np.random.rand(N) * 0.1 + 0.01
        
        # Use L-BFGS-B
        try:
            res = minimize(
                lambda x: _objective_and_constraints(x)[0],
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except:
            continue
        
        x_opt = res.x
        result = _construct_from_x(x_opt)
        result = _validate_and_fix(result)
        
        # Check validity
        valid = True
        for i in range(N):
            x, y, r = result[i]
            if x < r or x + r > 1 or y < r or y + r > 1 or r <= 0:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = result[i]
                    xj, yj, rj = result[j]
                    d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if d < ri + rj - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result.copy()
        
        # If we found a good solution, try to polish it
        if best_score > 1.0:
            # Try SLSQP for final polish
            try:
                res2 = minimize(
                    lambda x: _objective_and_constraints(x)[0],
                    x_opt,
                    method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                x_opt2 = res2.x
                result2 = _construct_from_x(x_opt2)
                result2 = _validate_and_fix(result2)
                
                valid2 = True
                for i in range(N):
                    x, y, r = result2[i]
                    if x < r or x + r > 1 or y < r or y + r > 1 or r <= 0:
                        valid2 = False
                        break
                
                if valid2:
                    for i in range(N):
                        for j in range(i+1, N):
                            xi, yi, ri = result2[i]
                            xj, yj, rj = result2[j]
                            d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                            if d < ri + rj - 1e-10:
                                valid2 = False
                                break
                        if not valid2:
                            break
                
                if valid2:
                    score2 = np.sum(result2[:, 2])
                    if score2 > best_score:
                        best_score = score2
                        best_result = result2.copy()
            except:
                pass
    
    # If no valid solution found, return a trivial one
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
