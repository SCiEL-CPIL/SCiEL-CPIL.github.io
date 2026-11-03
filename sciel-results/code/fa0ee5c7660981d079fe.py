"""Rewritten program for packing 50 circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 50

# EVOLVE-BLOCK-START
def _true_score(x_flat):
    """Compute the exact packing radius for a given configuration."""
    x = x_flat.reshape(N, 2)
    # Check if any point is outside the unit circle
    # The radius available for each circle is 1 - |x_i|
    radii_boundary = 1.0 - np.linalg.norm(x, axis=1)
    
    # If any circle is outside, the score is 0 (or negative, but 0 is safer for invalid)
    if np.any(radii_boundary < 0):
        return 0.0
        
    # Compute pairwise distances
    # dist[i, j] = ||x_i - x_j||
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # We only care about i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # The radius constrained by pairs is min(pair_dists) / 2
    r_pair = np.min(pair_dists) / 2.0
    
    # The radius constrained by boundary is min(radii_boundary)
    r_bound = np.min(radii_boundary)
    
    return min(r_pair, r_bound)

def _smooth_objective(x_flat):
    """
    Smooth surrogate for the negative of the packing radius.
    Minimizing this function maximizes the packing radius.
    Uses log-sum-exp for soft-min.
    """
    x = x_flat.reshape(N, 2)
    
    # Boundary terms: v_i = 1 - |x_i|
    radii_boundary = 1.0 - np.linalg.norm(x, axis=1)
    
    # Pairwise terms: v_ij = ||x_i - x_j|| / 2
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Concatenate all constraints
    # We want to maximize min(all_constraints)
    # This is equivalent to minimizing -softmin(all_constraints)
    # softmin(v) = -1/beta * log(sum(exp(-beta * v)))
    
    # Combine arrays
    all_vals = np.concatenate([radii_boundary, pair_dists])
    
    # To avoid overflow in exp, we can shift by the min value, 
    # but for softmin, we just need the log-sum-exp of -beta * v
    # log(sum(exp(-beta * v))) = -beta * max(v) + log(sum(exp(-beta * (v - max(v)))))
    # Actually, simpler:
    beta = 50.0
    scaled = -beta * all_vals
    
    # Subtract max for numerical stability in sum-exp
    max_scaled = np.max(scaled)
    # If max_scaled is very large positive, exp will overflow.
    # Since we are doing exp(-beta * v), and v are distances (positive), 
    # -beta * v are negative.
    # So max_scaled will be the least negative (closest to 0).
    # exp(max_scaled) is <= 1.
    # So no overflow risk in the numerator.
    
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    soft_min = -log_sum_exp / beta
    
    # We want to maximize soft_min, so we minimize -soft_min
    return -soft_min

def _generate_hexagonal_init():
    """Generate a hexagonal lattice configuration as a good initial guess."""
    # Create a hexagonal grid
    # We need enough points to pick N=50 from.
    # Approximate density: Area ~ pi. N=50.
    # Side length of hexagon ~ sqrt(pi/50) * 2/sqrt(3) ?
    # Let's just generate a grid and filter.
    
    # Try a few spacings
    best_points = None
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 1.0:
                    rows.append([x, y])
        points = np.array(rows)
        if len(points) >= N:
            # Sort by distance to origin to keep central points
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:N]]
            if best_points is None or _true_score(points.flatten()) > _true_score(best_points.flatten()):
                best_points = points
    if best_points is not None:
        return best_points.flatten()
    
    # Fallback: Random
    return np.random.uniform(-0.8, 0.8, (N, 2)).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initial guess from Hexagonal Lattice
    x0 = _generate_hexagonal_init()
    
    # 2. Local refinement with L-BFGS-B
    # We minimize the smooth objective
    result = minimize(
        _smooth_objective, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    x_best = result.x
    score_best = _true_score(x_best)
    
    # 3. Basin Hopping to escape local minima
    # We use a step generator that makes random jumps
    def step(x):
        # Random displacement
        # Scale displacement with current 'radius' or just fixed?
        # Fixed is safer.
        displacement = np.random.normal(0, 0.1, N * 2)
        return x + displacement
    
    # Basin Hopping parameters
    # We want to run as many steps as possible before deadline
    # Each step involves a local minimization (L-BFGS-B)
    
    # We will manually manage the loop to respect the deadline
    # basinhopping in scipy has a maxfun parameter, but it doesn't check wall-clock time.
    # So we implement a simple loop.
    
    current_x = x_best.copy()
    current_score = score_best
    
    # Number of steps to try
    # L-BFGS-B takes ~5-10ms per call for N=50.
    # 100s / 0.01s = 10,000 steps possible.
    # But basin hopping also has overhead.
    # Let's aim for ~500-1000 basin hopping steps.
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Generate a perturbed point
        x_perturbed = step(current_x)
        
        # Refine the perturbed point
        res = minimize(
            _smooth_objective, 
            x_perturbed, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        x_candidate = res.x
        score_candidate = _true_score(x_candidate)
        
        # Accept if better (or with some probability for annealing, but greedy is fine for packing usually)
        # For packing, accepting worse solutions can help escape local minima, 
        # but L-BFGS-B from a random jump often lands in a different basin.
        # Simple greedy: if better, update.
        if score_candidate > current_score:
            current_score = score_candidate
            current_x = x_candidate
            
        # Optional: If we stuck, try a bigger jump?
        # The step function uses fixed sigma.
        
    # 4. Final Polish
    # Run L-BFGS-B again on the best found solution with tighter tolerances
    if time.time() < deadline - 5.0:
        final_res = minimize(
            _smooth_objective, 
            current_x, 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        x_final = final_res.x
        score_final = _true_score(x_final)
        if score_final > current_score:
            current_x = x_final
            current_score = score_final

    return current_x.reshape(N, 2)
# EVOLVE-BLOCK-END
