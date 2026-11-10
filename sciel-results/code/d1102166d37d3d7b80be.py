# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy: Constrained Nonlinear Programming (NLP)
    1. Fix the scale by normalizing points such that the maximum pairwise distance is 1.
       This eliminates scale invariance and turns the ratio maximization into maximizing d_min.
    2. Use a smooth surrogate for the non-differentiable d_min: softmin(dists, beta).
       As beta increases, softmin approaches the true minimum.
    3. Use L-BFGS-B optimizer to maximize this smooth objective.
    4. Use multiple random restarts and a schedule of increasing beta (continuation method)
       to navigate the energy landscape and converge to the global optimum.
    
    Returns
        points: np.ndarray of shape (16, 2)
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (600s limit, keep 10s buffer)

    def time_exceeded():
        return time.time() > deadline

    def normalize_points(points):
        """Normalize points so that the max pairwise distance is 1."""
        if len(points) < 2:
            return points
        dists = pdist(points, 'euclidean')
        d_max = np.max(dists)
        if d_max < 1e-12:
            return points
        # Center the points to avoid drift, then scale
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        return centered / d_max

    def softmin(dists, beta):
        """
        Compute the soft minimum of a set of distances.
        softmin(x) = -1/beta * log(sum(exp(-beta * x)))
        """
        # Numerical stability: subtract min before exp
        min_val = np.min(dists)
        # If beta is large, exp(-beta * (dists - min_val)) can underflow for large dists, 
        # but we only care about the relative values.
        # Actually, standard stable softmin:
        # -1/beta * log(sum(exp(-beta * (dists - min_val)))) - min_val
        # Wait, if we shift by min_val, the term for min_val is exp(0)=1.
        # Others are exp(-beta * (d_i - d_min)) which is <= 1.
        # So sum is >= 1.
        # Result: -1/beta * log(sum) - min_val.
        # This approaches min_val as beta -> infinity.
        
        # For very large beta, we can just return min_val to avoid numerical issues
        if beta > 1000:
            return min_val
        
        # Compute -beta * (dists - min_val)
        # To avoid underflow in exp for very large beta * (dists - min_val),
        # we can clip the exponent.
        exponents = -beta * (dists - min_val)
        # Clip to avoid underflow (exp(-700) is ~0)
        exponents = np.clip(exponents, -700, 0)
        
        sum_exp = np.sum(np.exp(exponents))
        return -1.0 / beta * np.log(sum_exp) - min_val

    def objective_and_gradient(x_flat, beta):
        """
        Objective: -softmin(dists) (since we want to maximize softmin, we minimize its negative).
        Gradient: d/dx (-softmin)
        
        x_flat: shape (32,)
        """
        points = x_flat.reshape(n, 2)
        
        # Compute all pairwise distances and their gradients
        # dists[i, j] = ||p_i - p_j||
        # grad dists[i, j] wrt p_i = (p_i - p_j) / d_ij
        # grad dists[i, j] wrt p_j = (p_j - p_i) / d_ij = -(p_i - p_j) / d_ij
        
        # We need the gradient of softmin wrt points.
        # softmin = -1/beta * log(sum(exp(-beta * d_k))) - d_min
        # Let S = sum(exp(-beta * d_k))
        # d(softmin)/d(d_k) = -1/beta * (1/S) * exp(-beta * d_k) * (-beta) = exp(-beta * d_k) / S
        # Note: The -d_min term derivative is 0 wrt d_k (treating d_min as constant for softmin definition? 
        # No, the standard softmin derivative is just the softmax weights.
        # Let's verify:
        # f(x) = -1/beta log(sum(exp(-beta x_i)))
        # df/dx_k = -1/beta * 1/sum(...) * exp(-beta x_k) * (-beta) = exp(-beta x_k) / sum(exp(-beta x_j))
        # This is exactly the softmax probability p_k.
        # So grad softmin wrt d_k is p_k.
        
        # However, we used the stable version:
        # softmin = -1/beta log(sum(exp(-beta (d_k - d_min)))) - d_min
        # Let y_k = d_k - d_min.
        # grad softmin wrt d_k = exp(-beta y_k) / sum(exp(-beta y_j))
        # This is the same probability distribution.
        
        # So, grad softmin wrt points = sum_k p_k * (grad d_k wrt points)
        
        # Efficient calculation:
        # 1. Compute pairwise vectors and distances
        # 2. Compute weights p_k
        # 3. Accumulate gradient
        
        # Vectorized pairwise distance computation
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (distance 0)
        mask = ~np.eye(n, dtype=bool)
        
        # Get unique pairs (i < j) to avoid double counting in sum, 
        # but for gradient accumulation, we can use the full matrix and divide by 2?
        # No, it's easier to just iterate or use the full matrix.
        # Let's use the full matrix for simplicity, as n=16 is small.
        
        # d_k corresponds to pair (i, j).
        # grad d_k wrt p_i = diff[i, j] / d_k
        # grad d_k wrt p_j = -diff[i, j] / d_k
        
        # We need weights w_k = exp(-beta * (d_k - d_min)) / sum(...)
        d_min = np.min(dists[mask])
        
        # Calculate exponents for all pairs (i, j)
        # We only care about i != j
        exponents = np.zeros((n, n))
        exponents[mask] = -beta * (dists[mask] - d_min)
        exponents = np.clip(exponents, -700, 0)
        
        exp_weights = np.exp(exponents)
        # Zero out diagonal
        np.fill_diagonal(exp_weights, 0)
        
        sum_exp = np.sum(exp_weights)
        
        if sum_exp < 1e-15:
            # Fallback to hard min gradient (undefined, so use 0 or random)
            # This shouldn't happen with beta < 1000
            return 0.0, np.zeros_like(x_flat)
        
        weights = exp_weights / sum_exp
        
        # Gradient of softmin wrt d_k is weights[i, j]
        # Gradient wrt points:
        # grad_points[i] = sum_j weights[i, j] * (diff[i, j] / dists[i, j])
        # Note: dists[i, j] is 0 on diagonal, but weights are 0 there.
        # To avoid division by zero, we can set inv_dists to 0 where dists is 0.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dists = np.where(dists > 1e-12, 1.0 / dists, 0.0)
        
        # grad_dists[i, j] = diff[i, j] * inv_dists[i, j]
        grad_dists = diff * inv_dists[:, :, np.newaxis]
        
        # grad_points[i] = sum_j weights[i, j] * grad_dists[i, j]
        # grad_points = np.einsum('ij,ijk->ik', weights, grad_dists)
        grad_points = np.tensordot(weights, grad_dists, axes=([1], [1]))
        
        # Objective is -softmin, so gradient is -grad_points
        grad_x_flat = -grad_points.flatten()
        
        # Calculate softmin value
        # Reuse sum_exp from above
        softmin_val = -1.0 / beta * np.log(sum_exp) - d_min
        
        return -softmin_val, grad_x_flat

    def run_optimization(init_points, betas):
        """
        Run L-BFGS-B optimization with a schedule of increasing beta.
        """
        current_points = normalize_points(init_points)
        
        # Flatten for optimizer
        x_flat = current_points.flatten()
        
        for beta in betas:
            if time_exceeded():
                break
            
            # Optimize
            try:
                res = minimize(
                    objective_and_gradient,
                    x_flat,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.success or res.status in [0, 1, 2, 3, 4, 5, 6, 7]:
                    x_flat = res.x
                    current_points = x_flat.reshape(n, 2)
                    current_points = normalize_points(current_points)
                    x_flat = current_points.flatten()
            except Exception:
                pass
        
        return current_points

    def get_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial configurations
    inits = []
    
    # 1. Grid-like init
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Circle init
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)
    
    # 3. Random inits
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.uniform(-1, 1, (n, 2)))
    
    best_points = None
    best_score = 0.0
    
    # Beta schedule: Start low (smooth), increase to high (sharp)
    # Log-spaced betas
    betas = np.logspace(0, 3, 10) # 1 to 1000
    
    # Time management
    # We have ~590 seconds.
    # Let's allocate time per init.
    time_per_init = 580.0 / len(inits)
    
    for i, init in enumerate(inits):
        if time_exceeded():
            break
        
        init_start = time.time()
        
        # Run optimization
        p = run_optimization(init, betas)
        
        # Calculate score
        s = get_score(p)
        
        if s > best_score:
            best_score = s
            best_points = p
            
        # Check time
        if time.time() - init_start > time_per_init:
            # If we are running long, we might want to reduce betas or skip some inits
            # But L-BFGS-B is usually fast for n=16.
            pass

    # Final Polish: Take the best found and run a very high beta optimization
    if best_points is not None and not time_exceeded():
        # Use very high betas to sharpen the minimum
        high_betas = np.logspace(3, 4, 10) # 1000 to 10000
        p = run_optimization(best_points, high_betas)
        s = get_score(p)
        if s > best_score:
            best_score = s
            best_points = p

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        best_points = normalize_points(best_points)

    return best_points

# EVOLVE-BLOCK-END
