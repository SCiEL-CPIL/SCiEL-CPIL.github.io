"""
Physics-based circle packing using smooth energy minimization with L-BFGS-B.
Strategy:
1. Initialize with hexagonal packing.
2. Use a smooth surrogate (log-sum-exp) for the minimum distance objective.
3. Maximize the radius via binary search on r, with L-BFGS-B optimizing the smooth energy.
4. Final high-precision polish pass with tighter tolerances.
"""
import time
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

N = 200
SEED = 42
TIME_LIMIT = 115.0  # Safety margin before 150s kill
RNG = np.random.RandomState(SEED)


def _compute_radius(centers):
    """Compute the packing radius: min of half pairwise distances and distance to walls."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # Pairwise distances
    d = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0

    # Distance to walls (square is [-0.5, 0.5]^2)
    dist_to_walls = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_walls)

    return min(min_pair, min_wall)


def _hexagonal_init():
    """Create a hexagonal packing initialization for N points."""
    best_centers = None
    best_radius = 0.0

    # Try various hexagonal configurations
    for n_rows in range(12, 25):
        for n_cols in range(12, 25):
            # Hexagonal grid: rows alternate between n_cols and n_cols-1 points
            # Vertical spacing: s * sqrt(3)/2
            # Horizontal spacing: s
            # Total points: roughly n_rows * n_cols
            
            # Calculate spacing
            s_horiz = 1.0 / n_cols
            if n_rows > 1:
                s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
            else:
                s_vert = 1.0
            
            s = min(s_horiz, s_vert)
            
            # Create the grid
            centers = []
            y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
            y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
            
            for row in range(n_rows):
                y = y_start + row * s * np.sqrt(3) / 2
                if row % 2 == 0:
                    x_extent = (n_cols - 1) * s
                    x_start = -0.5 + (1.0 - x_extent) / 2
                    for col in range(n_cols):
                        x = x_start + col * s
                        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                            centers.append((x, y))
                else:
                    x_extent = (n_cols - 2) * s
                    x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                    for col in range(n_cols - 1):
                        x = x_start + col * s
                        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                            centers.append((x, y))
            
            centers = np.array(centers)
            if len(centers) >= N:
                centers = centers[:N]
                r = _compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()

    if best_centers is not None:
        return best_centers
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _smooth_min_energy(centers, r_target, beta=50.0):
    """
    Compute smooth energy that penalizes violations of radius r_target.
    Uses log-sum-exp for smooth approximation of the minimum.
    Energy = -log-sum-exp of (distances - 2*r_target) for pairs,
             plus wall violations.
    We want to MINIMIZE this energy (negative of log-sum-exp of violations).
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Smooth approximation of min(pair_dists - 2*r_target)
    # We want to penalize when pair_dists < 2*r_target
    # violation = 2*r_target - pair_dists (positive when violated)
    violations = 2.0 * r_target - pair_dists
    
    # Log-sum-exp for smooth max of violations
    # If all violations are negative (no violations), energy should be small
    v_max = np.max(violations)
    if v_max < 0:
        # No violations, return small positive value
        energy = -beta * v_max  # This is negative, but we want positive energy
        # Actually, let's use: energy = log(sum(exp(beta * violations)))
        # When violations are all negative, this is small
        exp_v = np.exp(beta * violations)
        energy = np.log(np.sum(exp_v)) / beta
    else:
        exp_v = np.exp(beta * (violations - v_max))
        energy = (v_max + np.log(np.sum(exp_v))) / beta
    
    # Wall violations
    wall_violations = []
    for axis in range(2):
        # Left wall
        wall_violations.append(r_target - (centers[:, axis] + 0.5))
        # Right wall
        wall_violations.append(r_target - (0.5 - centers[:, axis]))
    
    wall_violations = np.array(wall_violations)  # shape (4, N)
    wall_violations = wall_violations.flatten()
    
    w_max = np.max(wall_violations)
    if w_max < 0:
        exp_w = np.exp(beta * wall_violations)
        wall_energy = np.log(np.sum(exp_w)) / beta
    else:
        exp_w = np.exp(beta * (wall_violations - w_max))
        wall_energy = (w_max + np.log(np.sum(exp_w))) / beta
    
    total_energy = energy + wall_energy
    return total_energy


def _smooth_min_energy_grad(centers, r_target, beta=50.0):
    """
    Compute gradient of smooth energy with respect to centers.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    violations = 2.0 * r_target - pair_dists
    
    # Softmax weights for log-sum-exp
    v_max = np.max(violations)
    exp_v = np.exp(beta * (violations - v_max))
    sum_exp = np.sum(exp_v)
    weights = exp_v / sum_exp  # softmax weights
    
    # Gradient of log-sum-exp wrt pair_dists
    # d/dpair_dists [log(sum(exp(beta*(2*r - pair_dists))))]
    # = sum(exp(beta*(2*r - pair_dists)) * (-beta)) / sum(exp(...))
    # = -beta * weights
    # But we also have the chain rule: d pair_dists / d centers
    
    grad = np.zeros_like(centers)
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        if dists[i, j] < 1e-12:
            continue
        w = weights[idx] * (-beta)  # d energy / d pair_dists[i,j]
        
        # d pair_dists[i,j] / d centers[i] = (centers[i] - centers[j]) / dists[i,j]
        d_ij = diff[i, j] / dists[i, j]
        grad[i] += w * d_ij
        grad[j] -= w * d_ij
    
    # Wall gradients
    wall_grad = np.zeros_like(centers)
    wall_violations = []
    wall_indices = []  # (axis, point_index, sign)
    
    for axis in range(2):
        # Left wall: violation = r_target - (centers[:, axis] + 0.5)
        # d violation / d centers[:, axis] = -1
        viol_left = r_target - (centers[:, axis] + 0.5)
        wall_violations.append(viol_left)
        for k in range(N):
            wall_indices.append((axis, k, -1))
        
        # Right wall: violation = r_target - (0.5 - centers[:, axis])
        # d violation / d centers[:, axis] = +1
        viol_right = r_target - (0.5 - centers[:, axis])
        wall_violations.append(viol_right)
        for k in range(N):
            wall_indices.append((axis, k, 1))
    
    wall_violations = np.array(wall_violations)
    w_max = np.max(wall_violations)
    exp_w = np.exp(beta * (wall_violations - w_max))
    sum_exp_w = np.sum(exp_w)
    w_weights = exp_w / sum_exp_w
    
    for idx, (axis, k, sign) in enumerate(wall_indices):
        w = w_weights[idx] * (-beta) * sign
        wall_grad[k, axis] += w
    
    return grad + wall_grad


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Initialize
    centers = _hexagonal_init()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    print(f"Initial radius: {best_radius:.12f}")
    
    # Binary search on radius
    # Lower bound: current best
    # Upper bound: theoretical max (for 200 circles in unit square, roughly 0.037)
    r_low = best_radius
    r_high = 0.0375  # Slightly above best known
    
    beta = 50.0  # Temperature for smooth approximation
    
    # Try different beta values
    betas_to_try = [50.0, 100.0, 200.0]
    
    for beta in betas_to_try:
        if time.time() > deadline - 10.0:
            break
        
        # Binary search on r
        for iteration in range(15):
            if time.time() > deadline - 10.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            print(f"  Trying r = {r_mid:.12f} (beta={beta})")
            
            # Try to find a configuration with radius >= r_mid
            # Use L-BFGS-B to minimize the smooth energy
            success = False
            for attempt in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                # Perturb current best if not first attempt
                if attempt == 0:
                    x0 = best_centers.copy()
                else:
                    x0 = best_centers + RNG.randn(N, 2) * 0.002
                    x0 = np.clip(x0, -0.5, 0.5)
                
                # Define objective and gradient
                def obj(x):
                    return _smooth_min_energy(x, r_mid, beta)
                
                def grad(x):
                    return _smooth_min_energy_grad(x, r_mid, beta)
                
                # Bounds
                bounds = [(-0.5, 0.5)] * (2 * N)
                
                try:
                    result = minimize(
                        obj, x0, jac=grad, method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    new_centers = result.x
                    new_radius = _compute_radius(new_centers)
                    
                    if new_radius >= r_mid - 1e-9:
                        # Success! We can achieve this radius
                        r_low = r_mid
                        if new_radius > best_radius:
                            best_radius = new_radius
                            best_centers = new_centers.copy()
                        success = True
                        break
                    else:
                        # Failed to achieve r_mid
                        # Update best if we improved
                        if new_radius > best_radius:
                            best_radius = new_radius
                            best_centers = new_centers.copy()
                        r_high = r_mid
                        break
                        
                except Exception as e:
                    print(f"    L-BFGS-B failed: {e}")
                    continue
            
            if not success:
                # Can't achieve r_mid, narrow upper bound
                # Already done above in the failure case
                pass
            
            # Check if converged
            if r_high - r_low < 1e-10:
                break
    
    # Final polish: use the best radius found and do a final optimization
    if time.time() < deadline - 5.0:
        r_target = best_radius * 1.0001
        beta_final = 200.0
        
        def obj_final(x):
            return _smooth_min_energy(x, r_target, beta_final)
        
        def grad_final(x):
            return _smooth_min_energy_grad(x, r_target, beta_final)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                obj_final, best_centers, jac=grad_final, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            new_centers = result.x
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception as e:
            print(f"Final polish failed: {e}")
    
    print(f"Final radius: {best_radius:.12f}")
    
    return best_centers


if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Packing radius: {r:.12f}")
    print(f"Best known:     0.036612798904")
    print(f"Ratio:          {r / 0.036612798904:.6f}")
