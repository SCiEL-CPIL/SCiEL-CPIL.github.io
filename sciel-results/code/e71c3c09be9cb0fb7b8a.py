"""Optimized 100-sphere packing in unit cube via simulated annealing + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5
DEADLINE = 140.0  # safety margin before 150s kill


def compute_radius(centers):
    """Compute the maximum non-overlapping radius for given centers."""
    # Ensure centers are within the cube
    centers = np.clip(centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Min distance between any pair
    min_pair_dist = np.min(dists)
    
    # Min distance to boundary
    min_bound_dist = np.min(CUBE_HALF - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_bound_dist)


def soft_min_objective(x, beta=10.0):
    """Soft-min surrogate for the packing objective (to be minimized, so we negate)."""
    centers = x.reshape(-1, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    bound_dists = CUBE_HALF - np.abs(centers)
    bound_min = np.min(bound_dists, axis=1)
    
    # We want to maximize min(min_pair_dist/2, min_bound_dist)
    # Equivalently minimize -min(...)
    # Use log-sum-exp for soft min
    
    # For pairwise: we want soft min of dists/2
    log_pair = np.log(np.sum(np.exp(-beta * dists / 2.0))) / (-beta)
    
    # For boundary: soft min of bound_dists
    log_bound = np.log(np.sum(np.exp(-beta * bound_dists))) / (-beta)
    
    # Combine: soft min of the two
    # Actually, we need min of (min_pair/2, min_bound)
    # Let's use a different approach: combine all constraints
    
    # Create a combined array of all "clearances"
    # Pairwise clearances: dists[i,j]/2 for i<j
    # Boundary clearances: min_k (CUBE_HALF - |c_ik|)
    
    # For efficiency, just compute the actual min and use it with a penalty
    min_pair = np.min(dists) / 2.0
    min_bound = np.min(bound_dists)
    
    r = min(min_pair, min_bound)
    
    # Negative for minimization
    return -r


def construct():
    start_time = time.time()
    
    # Initialize on a dense lattice
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy().astype(np.float64)
    
    best_radius = compute_radius(centers)
    best_centers = centers.copy()
    
    # Simulated annealing phase
    if time.time() - start_time < DEADLINE - 20:
        T = 0.01
        T_min = 1e-6
        alpha = 0.995
        n_iter = 50000
        
        for i in range(n_iter):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Random perturbation
            idx = np.random.randint(0, N)
            delta = np.random.normal(0, T * 0.5, size=3)
            new_centers = centers.copy()
            new_centers[idx] += delta
            
            # Keep within bounds
            new_centers[idx] = np.clip(new_centers[idx], -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            
            new_radius = compute_radius(new_centers)
            current_radius = compute_radius(centers)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_radius >= current_radius or np.random.rand() < np.exp((new_radius - current_radius) / T):
                centers = new_centers
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            
            T = max(T * alpha, T_min)
    
    # L-BFGS-B refinement phase
    if time.time() - start_time < DEADLINE - 5:
        # Use the best centers found so far
        x0 = best_centers.flatten()
        
        def objective(x):
            if time.time() - start_time > DEADLINE - 5:
                return 0.0  # Stop
            centers = x.reshape(-1, 3)
            r = compute_radius(centers)
            return -r
        
        def gradient(x):
            # Numerical gradient
            centers = x.reshape(-1, 3)
            r = compute_radius(centers)
            
            # Compute gradient via finite differences
            grad = np.zeros_like(x)
            eps = 1e-7
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                r_plus = compute_radius(x_plus.reshape(-1, 3))
                x_minus = x.copy()
                x_minus[i] -= eps
                r_minus = compute_radius(x_minus.reshape(-1, 3))
                grad[i] = (r_plus - r_minus) / (2 * eps)
            
            return -grad
        
        # Use L-BFGS-B
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(-CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)] * (3 * N),
                options={'maxiter': 1000, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result.success and -result.fun > best_radius:
                best_radius = -result.fun
                best_centers = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Final polish: a few more iterations with tighter tolerance
    if time.time() - start_time < DEADLINE - 2:
        try:
            x0 = best_centers.flatten()
            
            def objective2(x):
                if time.time() - start_time > DEADLINE - 2:
                    return 0.0
                centers = x.reshape(-1, 3)
                r = compute_radius(centers)
                return -r
            
            def gradient2(x):
                centers = x.reshape(-1, 3)
                r = compute_radius(centers)
                grad = np.zeros_like(x)
                eps = 1e-8
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    r_plus = compute_radius(x_plus.reshape(-1, 3))
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    r_minus = compute_radius(x_minus.reshape(-1, 3))
                    grad[i] = (r_plus - r_minus) / (2 * eps)
                return -grad
            
            result = minimize(
                objective2,
                x0,
                method='L-BFGS-B',
                jac=gradient2,
                bounds=[(-CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)] * (3 * N),
                options={'maxiter': 500, 'gtol': 1e-10, 'maxfun': 2000}
            )
            
            if result.success and -result.fun > best_radius:
                best_radius = -result.fun
                best_centers = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Ensure final result is valid
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers
