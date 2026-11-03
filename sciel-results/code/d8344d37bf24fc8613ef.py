"""Pack 50 equal circles in a unit square using Simulated Annealing + High-Beta L-BFGS-B Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n):
    """Create a hexagonal grid of n points centered in the square."""
    # For N=50, a 7x8 grid with hexagonal offset is a good start.
    # dx = 1/7, dy = sqrt(3)/14
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter to square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Sort by distance from center, keep closest N to form a compact blob?
        # Or just take first N?
        # Taking first N from a centered grid is usually fine.
        points = points[:N]
    elif len(points) < N:
        # Should not happen for 8x7=56
        pass
        
    return points

def _make_square_grid(n):
    """Create a square grid of n points."""
    # 7x8 grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    sq_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    return sq_centers

def _soft_min_radius_sq(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp on SQUARED distances.
    We minimize -r.
    r = min( wall_constraints, pair_dists/2 )
    
    To avoid sqrt, we can work with r^2.
    Let t = r^2.
    Constraints:
    wall: (0.5 - |x|)^2 >= t
    pair: ||c_i - c_j||^2 >= 4t
    
    We want to maximize t.
    Objective: -t.
    Surrogate for min(t_constraints):
    min_k a_k  ~  -1/beta * log(sum exp(-beta * a_k))
    
    Here a_k are the squared distances.
    For pairs, a_k = dist_sq. We want dist_sq >= 4t.
    So constraint value is dist_sq - 4t? No, we are fixing centers and estimating r.
    
    Actually, the standard soft-min for radius r is:
    r_soft = -1/beta * log( sum_{walls} exp(-beta * w_i) + sum_{pairs} exp(-beta * d_ij/2) )
    
    If we use squared distances, we define:
    t_soft = -1/beta * log( sum_{walls} exp(-beta * w_i^2) + sum_{pairs} exp(-beta * (d_ij^2/4)) )
    
    Then r_soft = sqrt(t_soft).
    Minimizing -r_soft is equivalent to minimizing -sqrt(t_soft).
    Gradient of -sqrt(t) is -1/(2*sqrt(t)) * grad(t).
    
    Let's just minimize -t_soft directly?
    Maximizing t (radius squared) is equivalent to maximizing r.
    So we can minimize -t_soft.
    This is smoother and avoids the sqrt in the objective value calculation (though sqrt is needed for the final radius check).
    
    Wait, the gradient of the log-sum-exp is well-defined.
    Let's define the function to minimize: f(x) = -t_soft(x)
    where t_soft(x) = -1/beta * log( sum exp(-beta * A_k(x)) )
    A_k are the squared "slack" variables.
    
    Wall slacks:
    A_w1 = (0.5 - x_i)^2
    A_w2 = (0.5 + x_i)^2
    A_w3 = (0.5 - y_i)^2
    A_w4 = (0.5 + y_i)^2
    
    Pair slacks:
    A_p = (dist_ij^2) / 4.0
    
    We sum exp(-beta * A).
    
    Note: If A is large, exp(-beta * A) is 0.
    If A is small (close to 0, i.e., tight constraint), exp(-beta * A) is large.
    The log-sum-exp will be dominated by the smallest A (tightest constraint).
    t_soft approximates min(A_k).
    Since r = sqrt(min(A_k)), minimizing -t_soft maximizes min(A_k), which maximizes r.
    """
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Squared wall distances
    wall_sq = np.concatenate([c1**2, c2**2, c3**2, c4**2])
    
    # Squared pair distances / 4
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, 1)
    pair_sq = dist_sq[iu] / 4.0
    
    all_sq = np.concatenate([wall_sq, pair_sq])
    
    # Clip to avoid underflow/overflow issues, though exp(-beta * positive) is safe.
    # If all_sq is very large, exp(-beta * all_sq) -> 0.
    # If all_sq is negative (shouldn't happen for walls if inside, but for pairs dist_sq is always >= 0).
    # Actually, if a point is outside, wall_sq is still positive (squared).
    # But we want to penalize being outside.
    # If x > 0.5, c1 = 0.5 - x < 0. c1^2 > 0.
    # The constraint is x <= 0.5. The "slack" is 0.5 - x.
    # If we square it, the sign is lost.
    # If x = 0.6, slack = -0.1, sq = 0.01.
    # If x = 0.4, slack = 0.1, sq = 0.01.
    # This is symmetric. It doesn't distinguish inside vs outside.
    # However, L-BFGS-B will push it towards the center if the objective decreases.
    # But for SA, we need a penalty for going outside.
    
    # Let's stick to linear slacks for walls to preserve direction, but squared for pairs?
    # No, mixing units in log-sum-exp is bad.
    
    # Let's use linear slacks for walls, and linear slacks for pairs (dist/2).
    # The sqrt in dist is the bottleneck.
    # Can we approximate dist/2 with a smooth function of dist_sq?
    # f(u) = sqrt(u)/2.
    # We can use the soft-min on the linear constraints.
    # The cost of sqrt on 1225 pairs is actually not that high in NumPy (vectorized).
    # The bottleneck in L-BFGS-B is the number of iterations, not the single evaluation cost, 
    # UNLESS the gradient is noisy or the function is expensive.
    # 
    # Let's optimize the evaluation cost anyway.
    # 
    # Alternative: Use the squared formulation but with a penalty for going outside.
    # Penalty: max(0, |x| - 0.5 + r)^2.
    # 
    # Let's try a different approach for the objective:
    # Minimize: -r + lambda * (violation^2)
    # 
    # Actually, let's just use the linear soft-min but optimize the NumPy operations.
    # The previous code calculated dist = sqrt(sum(diff^2)).
    # This is fine.
    
    # Let's go back to linear soft-min but ensure it's as fast as possible.
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    # Clip constraints to a minimum value to avoid exp overflow if beta is high and constraints are negative?
    # If constraint is negative (overlap), -beta * negative = positive. exp(positive) is large.
    # This is correct: large penalty.
    
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    centers = x.reshape(N, 2)
    r = _soft_min_radius_sq(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    x0 = centers.ravel()
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _simulated_annealing(centers, time_limit=5.0):
    """
    Perform Simulated Annealing to escape local minima.
    Objective: Maximize radius.
    We use a penalty function that penalizes overlaps and wall violations.
    Penalty = (sum of overlap depths^2) + (sum of wall violations^2)
    We want to minimize Penalty, but also maximize the minimum radius.
    
    A common approach:
    Score = min_radius - lambda * total_violation
    Or:
    Score = - (sum of (max(0, 2r - d_ij))^2 + sum of (max(0, r - wall_dist))^2)
    This finds a valid configuration.
    
    To maximize radius, we can slowly increase r and anneal.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Start with a small radius and increase it
    r_current = _radius(centers)
    if r_current <= 0:
        r_current = 0.01
    
    # Target radius is ~0.071
    r_target = 0.0715
    
    # Annealing parameters
    T = 0.01  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    
    # We will iterate, increasing r slightly each step if we have been stable
    # and annealing to find a new configuration for that r.
    
    # Actually, a simpler SA:
    # Objective: f(x) = -min_radius(x) + penalty(x)
    # penalty(x) = sum (overlap_i)^2 + sum (wall_viol_i)^2
    
    def penalty_and_radius(c):
        # Walls
        wall_dists = np.min([0.5 - c[:, 0], 0.5 + c[:, 0], 0.5 - c[:, 1], 0.5 + c[:, 1]], axis=0)
        wall_viol = np.maximum(0, r_current - wall_dists)
        
        # Pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        # For penalty, we sum all overlaps
        # Overlap for pair (i,j) is max(0, 2*r_current - d_ij)
        # This is expensive to compute for all pairs in a loop, but vectorized it's fine.
        pair_viol_matrix = np.maximum(0, 2 * r_current - dist)
        np.fill_diagonal(pair_viol_matrix, 0)
        pair_viol_sum = np.sum(pair_viol_matrix ** 2)
        
        wall_viol_sum = np.sum(wall_viol ** 2)
        
        total_penalty = pair_viol_sum + wall_viol_sum
        min_r = min(np.min(wall_dists), min_pair / 2.0)
        
        return total_penalty, min_r

    # Initial state
    current_penalty, current_r = penalty_and_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    best_penalty = current_penalty
    
    step_size = 0.05
    num_steps = 0
    
    while time.time() - start_time < time_limit:
        num_steps += 1
        
        # Generate a move: pick a random point and move it
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Keep within bounds? Or allow and penalize?
        # Allow and penalize is better for SA.
        
        new_penalty, new_r = penalty_and_radius(new_centers)
        
        # Energy difference
        # We want to minimize penalty, but maximize r.
        # Let's define Energy = penalty - alpha * r
        # If we increase r, energy increases? No.
        # We want to find a configuration with LOW penalty and HIGH r.
        # Let's just minimize Penalty. If Penalty is 0, we check R.
        # If we use Penalty only, we might get stuck in a valid config with small R.
        # So we must increase R over time.
        
        # Strategy: 
        # 1. If current config is valid (penalty == 0), try to increase r_target slightly.
        # 2. If invalid, try to reduce penalty.
        
        # Let's use a combined score: Score = -penalty + beta_anneal * r
        # If we accept moves that increase Score, we balance validity and radius.
        
        alpha = 10.0 # Weight for radius in acceptance
        
        current_score = -current_penalty + alpha * current_r
        new_score = -new_penalty + alpha * new_r
        
        delta_E = new_score - current_score
        
        if delta_E > 0 or np.random.rand() < np.exp(-delta_E / T):
            centers = new_centers
            current_penalty = new_penalty
            current_r = new_r
            
            if current_r > best_r and current_penalty < 1e-6:
                best_r = current_r
                best_centers = centers.copy()
                # If we found a better valid config, increase r_target slightly
                # This forces the search to look for larger radii
                r_current = min(r_current + 0.0005, r_target)
                T = max(T * 0.99, min_T) # Cool down slightly when improving
            else:
                T = max(T * 0.999, min_T)
        else:
            T = max(T * 0.999, min_T)
            
        # Reduce step size over time
        step_size *= 0.999
        
        # Check for timeout
        if time.time() - start_time > time_limit:
            break
            
    return best_centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal grid
    hex_centers = _make_hexagonal_grid(N)
    if hex_centers is not None and len(hex_centers) == N:
        candidates.append(hex_centers)
        
    # 2. Square grid
    sq_centers = _make_square_grid(N)
    candidates.append(sq_centers)
    
    # 3. Random starts
    for i in range(5):
        c_rand = np.random.uniform(-0.4, 0.4, size=(N, 2))
        candidates.append(c_rand)
        
    # Phase 1: Simulated Annealing on all candidates
    # Allocate time based on number of candidates
    # Let's give 10s to each candidate for SA, max 5 candidates -> 50s
    # Then use remaining time for L-BFGS-B polish on the best one.
    
    sa_time_per_candidate = 10.0
    
    for idx, centers in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 20: # Leave 20s for polish
            break
            
        # Run SA
        sa_centers = _simulated_annealing(centers, time_limit=sa_time_per_candidate)
        r_sa = _radius(sa_centers)
        
        if r_sa > best_r:
            best_r = r_sa
            best_centers = sa_centers.copy()
            
    # Phase 2: L-BFGS-B Polish on the best center found
    if best_centers is not None:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 1.0:
            # Use a moderate beta first
            polished = _local_refine_smooth(best_centers, time_limit=min(10.0, time_left * 0.5), beta=200.0)
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
                
            # Final high-precision polish
            time_left_final = DEADLINE - 2 - (time.time() - start_time)
            if time_left_final > 1.0:
                final_polished = _local_refine_smooth(best_centers, time_limit=time_left_final, beta=500.0)
                r_final = _radius(final_polished)
                
                if r_final > best_r:
                    best_r = r_final
                    best_centers = final_polished.copy()
                    
    if best_centers is None:
        # Fallback
        best_centers = _make_hexagonal_grid(N)
        
    return best_centers

# Ensure the function is accessible
if __name__ == "__main__":
    # Quick test
    centers = construct()
    print(f"Radius: {_radius(centers)}")
