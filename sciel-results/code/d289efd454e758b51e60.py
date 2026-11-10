import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    def get_distances(points):
        return scipy.spatial.distance.pdist(points)

    def compute_score(points):
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_neg(points_flat):
        points = points_flat.reshape(n, d)
        return -compute_score(points)

    def soft_min_max_objective(points_flat, tau=0.05):
        """
        Smooth surrogate: minimize -(log(d_min) - log(d_max))
        using soft-min and soft-max for differentiability.
        """
        points = points_flat.reshape(n, d)
        dists = get_distances(points)
        if np.any(dists <= 1e-12):
            return 1e6
        
        # Soft min and max
        # log(d_min) approx -tau * log(sum(exp(-dists/tau)))
        # log(d_max) approx  tau * log(sum(exp(dists/tau)))
        
        # Numerical stability
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # For soft min:
        # exp(-dists/tau) can overflow if dists is large negative? No, dists are positive.
        # If dists is large, exp(-large) is 0.
        # If dists is small, exp(-small) is 1.
        # To be safe, subtract max of -dists/tau (which is -min_d/tau)
        # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
        
        x_min = -dists / tau
        m_min = np.max(x_min)
        log_sum_exp_min = m_min + np.log(np.sum(np.exp(x_min - m_min)))
        log_dmin = -tau * log_sum_exp_min
        
        x_max = dists / tau
        m_max = np.max(x_max)
        log_sum_exp_max = m_max + np.log(np.sum(np.exp(x_max - m_max)))
        log_dmax = tau * log_sum_exp_max
        
        if log_dmax == 0:
            return 1e6
            
        return -(log_dmin - log_dmax)

    # Initializations
    def hexagonal_initial(n_points):
        # Generate a hexagonal cluster
        # 19 points is a full hexagon of radius 2.
        # We need 16.
        # Let's generate 19 and pick the 16 with smallest norm? Or just use a 4x4 grid.
        # Actually, a 4x4 grid is a good start.
        points = np.zeros((n_points, 2))
        spacing = 1.0
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * np.sqrt(3) / 2
                points[idx] = [x, y]
        points -= np.mean(points, axis=0)
        return points

    def random_initial(n_points, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time() * 1000) % 2**31)
        points = np.random.randn(n_points, d)
        points -= np.mean(points, axis=0)
        return points

    best_points = None
    best_score = -1.0

    # Candidate initializations
    inits = [
        hexagonal_initial(n),
        random_initial(n, seed=42),
        random_initial(n, seed=123),
        random_initial(n, seed=456),
    ]
    
    # Add a circular initial
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    for init in inits:
        if time.time() > deadline - 60:
            break
            
        current_points = init.copy()
        
        # 1. Differential Evolution on Smooth Surrogate
        # Bounds: [-10, 10] is safe, but we can normalize after.
        # Let's use bounds based on the initial scale.
        # For simplicity, fixed bounds [-5, 5].
        bounds = [(-5, 5)] * (n * d)
        
        try:
            de_result = scipy.optimize.differential_evolution(
                soft_min_max_objective,
                bounds=bounds,
                x0=current_points.flatten(),
                maxiter=100,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=False, # We will polish manually
                init='sobol'
            )
            de_points = de_result.x.reshape(n, d)
            
            # 2. Nelder-Mead on Exact Objective
            # This refines the solution to the true metric
            nm_result = scipy.optimize.minimize(
                objective_neg,
                de_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_points = nm_result.x.reshape(n, d)
            
            score = compute_score(nm_points)
            if score > best_score:
                best_score = score
                best_points = nm_points.copy()
                
        except Exception as e:
            print(f"Error in optimization loop: {e}")
            pass

    # 3. Basin Hopping on the best result found so far
    if best_points is not None and time.time() < deadline - 30:
        try:
            bh_result = scipy.optimize.basinhopping(
                objective_neg,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}},
                niter=200,
                stepsize=0.5,
                seed=42,
                disp=False
            )
            bh_points = bh_result.x.reshape(n, d)
            score = compute_score(bh_points)
            if score > best_score:
                best_score = score
                best_points = bh_points.copy()
        except Exception as e:
            print(f"Error in basin hopping: {e}")
            pass

    # 4. Final Polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 5:
        try:
            nm_result = scipy.optimize.minimize(
                objective_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_points = nm_result.x.reshape(n, d)
            score = compute_score(final_points)
            if score > best_score:
                best_score = score
                best_points = final_points.copy()
        except Exception as e:
            print(f"Error in final polish: {e}")
            pass

    if best_points is None:
        # Fallback
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points
