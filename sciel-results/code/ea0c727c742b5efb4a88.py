# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline

    def get_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_max(params, temp):
        """
        Smooth objective function.
        We minimize -log(sum(exp(d_i^2 / T)))
        This approximates -d_min^2 / T for large T (explore)
        and -d_min^2 for small T (exploit).
        We also divide by d_max^2 to normalize the ratio.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e10 # Penalize zero diameter
        # We want to maximize d_min / d_max.
        # Equivalent to minimizing -d_min / d_max.
        # Let's use the soft-min on distances.
        # To make it dimensionless and scale-invariant, we can normalize points or use the ratio directly.
        # Using log-sum-exp on d^2 is stable.
        d2 = dists ** 2
        # Soft min of d^2
        # min(d^2) approx -T * log(sum(exp(-d^2/T)))
        # We want to maximize min(d^2) / max(d^2).
        # Let's minimize - (soft_min(d^2) / max(d^2)).
        
        # Numerical stability for log-sum-exp
        # We are minimizing -soft_min.
        # soft_min(d^2) = -T * log( sum( exp(-d^2/T) ) )
        # Objective = - ( -T * log( sum( exp(-d^2/T) ) ) ) / d_max^2
        # Objective = T * log( sum( exp(-d^2/T) ) ) / d_max^2
        
        # To avoid overflow in exp(-d^2/T), we can subtract the min of -d^2/T (which is -max(d^2)/T)
        # Or just rely on T being large enough initially.
        
        scaled = -d2 / temp
        # Subtract max for stability
        max_scaled = np.max(scaled)
        sum_exp = np.sum(np.exp(scaled - max_scaled))
        soft_min_val = -temp * (np.log(sum_exp) + max_scaled)
        
        ratio = soft_min_val / (d_max ** 2)
        return -ratio

    def gradient(params, temp):
        """
        Numerical gradient for L-BFGS-B.
        """
        # Central difference
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            f_plus = soft_min_max(p_plus, temp)
            f_minus = soft_min_max(p_minus, temp)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    best_points = None
    best_score = 0.0

    # Initial configurations
    inits = []
    
    # 1. Grid-like init (often a good start for packing)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Circle init
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)

    # 3. Hexagonal-like init
    # Create a hex grid
    hex_points = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Pick 16 from the center out
    centers = np.argpartition(np.sum(hex_points**2, axis=1), 16)[:16]
    inits.append(hex_points[centers])
    
    # 4. Random inits
    for seed in range(20):
        np.random.seed(seed)
        inits.append(np.random.uniform(-1, 1, (n, 2)))

    # Optimization Loop
    # We will run a few rounds with decreasing temperature (annealing)
    # T starts high to smooth the landscape, ends low to refine.
    
    temperatures = [10.0, 2.0, 0.5, 0.1, 0.05]
    
    current_best_init = inits[0]
    
    for temp in temperatures:
        if time_exceeded():
            break
            
        # Try to improve the current best, and also try other inits if time permits
        # Strategy: Refine current best first. If time remains, try other inits with this temp.
        
        # 1. Refine current best
        try:
            result = minimize(
                soft_min_max, 
                current_best_init.ravel(), 
                method='L-BFGS-B', 
                jac=gradient, 
                args=(temp,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = result.x.reshape((n, 2))
            score = get_score(cand)
            if score > best_score:
                best_score = score
                best_points = cand
                current_best_init = cand
        except Exception:
            pass

        # 2. Try other inits if we have time
        if not time_exceeded():
            # Only try a few other inits per temperature step to save time
            for init in inits:
                if time_exceeded():
                    break
                if np.array_equal(init, current_best_init):
                    continue
                try:
                    result = minimize(
                        soft_min_max, 
                        init.ravel(), 
                        method='L-BFGS-B', 
                        jac=gradient, 
                        args=(temp,),
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    cand = result.x.reshape((n, 2))
                    score = get_score(cand)
                    if score > best_score:
                        best_score = score
                        best_points = cand
                        current_best_init = cand
                except Exception:
                    pass

    # Final polish with very low temperature on the best found
    if not time_exceeded() and best_points is not None:
        try:
            result = minimize(
                soft_min_max, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                jac=gradient, 
                args=(0.01,),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            cand = result.x.reshape((n, 2))
            score = get_score(cand)
            if score > best_score:
                best_score = score
                best_points = cand
        except Exception:
            pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        
    return best_points

# EVOLVE-BLOCK-END
