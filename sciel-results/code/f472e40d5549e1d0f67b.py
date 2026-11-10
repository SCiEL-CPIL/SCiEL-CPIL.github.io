import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    time_limit = 590.0  # Safety margin before 630s kill

    def exact_score(points):
        """Calculate the exact score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, beta=50.0):
        """
        Smoothed objective to minimize.
        Minimizes 2 * (log(d_max_approx) - log(d_min_approx))
        which is equivalent to maximizing (d_min_approx / d_max_approx)^2
        """
        points = params.reshape(n, 2)
        d = pdist(points)
        
        # Penalty for points going too far out (scale invariance allows any scale, 
        # but we need a bound for numerical stability. 
        # Actually, the ratio is scale invariant. 
        # If we don't bound, the optimizer might drift. 
        # Let's bound to radius 10.
        r = np.linalg.norm(points, axis=1)
        if np.any(r > 10):
            return 1e10
        
        # Log-sum-exp for min and max
        # d_min approx = (1/beta) * log(sum(exp(-beta * d)))
        # d_max approx = (1/beta) * log(sum(exp(beta * d)))
        
        # To avoid overflow/underflow, use log-sum-exp trick
        # log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
        
        a_min = -beta * d
        max_a_min = np.max(a_min)
        log_sum_exp_min = max_a_min + np.log(np.sum(np.exp(a_min - max_a_min)))
        log_d_min_approx = log_sum_exp_min / beta
        
        a_max = beta * d
        max_a_max = np.max(a_max)
        log_sum_exp_max = max_a_max + np.log(np.sum(np.exp(a_max - max_a_max)))
        log_d_max_approx = log_sum_exp_max / beta
        
        if log_d_min_approx <= 0 or log_d_max_approx <= 0:
            return 1e10
            
        # Minimize log(d_max) - log(d_min)
        return 2.0 * (log_d_max_approx - log_d_min_approx)

    def step(x, T):
        """
        Perturbation step for basin hopping.
        Add Gaussian noise.
        """
        # T is the temperature, which controls the step size
        # We scale the step size by T
        return x + T * np.random.normal(0, 1, x.shape)

    def accept(x_new, f_new, x_old, f_old, T):
        """
        Acceptance criterion for basin hopping.
        """
        # Standard Metropolis
        if f_new < f_old:
            return True
        return np.random.rand() < np.exp(-(f_new - f_old) / T)

    best_points = None
    best_score = -1.0

    # Initial configurations
    inits = []
    
    # 1. Hexagonal-like cluster
    s3 = np.sqrt(3)
    p0 = [(0,0)]
    p1 = [(-0.5, s3/2), (0.5, s3/2), (1.5, s3/2)]
    p2 = [(-1, s3), (0, s3), (1, s3), (2, s3), (3, s3)]
    p3 = [(-1.5, 3*s3/2), (-0.5, 3*s3/2), (0.5, 3*s3/2), (1.5, 3*s3/2), (2.5, 3*s3/2), (3.5, 3*s3/2), (4.5, 3*s3/2)]
    all_pts = p0 + p1 + p2 + p3
    arr = np.array(all_pts)
    center = arr.mean(axis=0)
    arr -= center
    inits.append(arr)

    # 2. Square grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    inits.append(np.column_stack([xx.ravel(), yy.ravel()]))

    # 3. Rings
    def generate_rings(n_inner, n_outer):
        p_center = np.array([0.0, 0.0])
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        angles2 += np.pi / n_outer
        p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([p_center, p_inner, p_outer])
        
    inits.append(generate_rings(6, 9))
    inits.append(generate_rings(5, 10))
    inits.append(generate_rings(7, 8))

    # 4. Random inits
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)

    # Run Basin Hopping from each initial point
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        x0 = init_pts.flatten()
        
        # We use a fixed beta for the smooth objective
        # We can adapt beta, but fixed is simpler. 
        # Higher beta = sharper min, but harder to optimize.
        # Let's try beta=50.
        
        try:
            # Basin hopping parameters
            # T0: initial temperature
            # stepsize: step size multiplier
            # maxiter: max iterations
            # niter: iterations per step? No, niter is for the local minimizer?
            # In scipy basinhopping:
            #   x0: initial
            #   objective: function
            #   step: function
            #   accept: function
            #   T: initial temp
            #   stepsize: step size
            #   maxfun: max function evals
            #   niter: max iterations (outer loop)
            
            res = basinhopping(
                lambda x: smooth_objective(x, beta=50.0),
                x0,
                step=step,
                accept=accept,
                T=1.0,
                stepsize=0.5,
                maxfun=1000,
                niter=50,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}}
            )
            
            new_points = res.x.reshape(n, 2)
            new_score = exact_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                
        except Exception as e:
            # Fallback to Powell if basin hopping fails
            try:
                res_powell = minimize(
                    lambda x: smooth_objective(x, beta=50.0), 
                    x0, 
                    method='Powell', 
                    options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                new_points = res_powell.x.reshape(n, 2)
                new_score = exact_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
            except Exception:
                continue

    # If no valid points were found, return a default
    if best_points is None:
        best_points = inits[0]

    return best_points
