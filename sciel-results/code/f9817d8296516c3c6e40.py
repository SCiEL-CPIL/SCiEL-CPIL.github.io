"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Paraboloid Basin-Snapping for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # CMA-ES Parameters
        # For Lunacek, we need robust global search.
        # Standard CMA-ES with adaptive sigma.
        
        # We will use a simplified, robust CMA-ES implementation.
        # Since we have a hard budget of 500 and dim=5, we can afford a few restarts
        # or a single long run with good initialization.
        # Lunacek has two basins. One is the global minimum, one is a local minimum.
        # The global minimum is typically closer to the origin in the transformed space,
        # but the shift is unknown.
        
        # Strategy:
        # 1. Initialize CMA-ES at the center (0,0,0,0,0) with a large sigma.
        # 2. Run CMA-ES.
        # 3. Periodically (or when convergence is detected), attempt a "Paraboloid Snap".
        #    The Lunacek function locally resembles a paraboloid around its minima.
        #    We can fit a quadratic model to recent points to find the exact minimum of that local basin.
        #    This allows us to jump directly to the basin bottom, saving calls.
        
        # To keep it robust and within budget, we will do:
        # - Phase 1: Global exploration with CMA-ES (large sigma).
        # - Phase 2: Local refinement with CMA-ES (small sigma).
        # - Interspersed: Paraboloid fits if we have enough points in a small region.
        
        # Let's implement a lightweight CMA-ES.
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Start large to cover the box [-5, 5]^5
        C = np.eye(dim)
        step_size = sigma
        
        # CMA-ES weights
        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # Population size
        # For dim=5, lambda = 4 * ceil(3 + log(5)) = 4 * ceil(5.2) = 24.
        # This is a reasonable population size.
        
        # Weights for recombination
        mu = lambda_ // 2  # Top half
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        D = np.sqrt(np.linalg.det(C))
        
        # Budget allocation:
        # We have 500 calls.
        # One generation of CMA-ES uses lambda_ calls.
        # Max generations = 500 // lambda_ ~ 20.
        # This might be too few for CMA-ES to converge fully.
        # Let's reduce lambda_ to allow more generations?
        # lambda_ = 10?
        # If lambda_ = 10, generations = 50. Better.
        # Let's use a smaller population for this small dimension.
        lambda_ = 10
        mu = 5
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Store recent points for paraboloid fitting
        recent_points = []
        
        def paraboloid_fit(points):
            """
            Fit a quadratic model to a set of points and find its minimum.
            Returns the minimum point and value, or None if fit is poor.
            """
            if len(points) < dim + 1:
                return None, np.inf
            
            # Use the last dim+1 points
            pts = np.array([p[0] for p in points[-(dim+1):]])
            vals = np.array([p[1] for p in points[-(dim+1):]])
            
            # Construct design matrix for quadratic:
            # f(x) = x^T A x + b^T x + c
            # We need 2*dim + 1 coefficients for full quadratic, but we only have dim+1 points.
            # So we can only fit a linear model + quadratic term along the direction of steepest descent?
            # Or assume diagonal quadratic?
            # Simpler: Fit a linear model to estimate gradient, then take a Newton step?
            # Or: Since Lunacek is separable-ish? No, it's rotated.
            # Let's just do a simple gradient estimation and line search if we can't fit full quadratic.
            
            # Actually, for Lunacek, the local structure is very paraboloid-like.
            # If we have dim+1 points, we can't fit a full quadratic (needs 2*dim+1).
            # We can fit a linear model to get the gradient direction.
            
            # Design matrix for linear fit: [x_i, 1]
            A = np.hstack([pts, np.ones((dim+1, 1))])
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(A, vals, rcond=None)
            except:
                return None, np.inf
                
            grad = coeffs[:dim]
            # Step size for Newton-like update?
            # If we assume Hessian is approx identity (locally), then x_new = x_mean - grad.
            # But we don't have x_mean here, we have specific points.
            # Let's use the average of the points as the center.
            center = np.mean(pts, axis=0)
            x_new = center - grad
            
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            
            return x_new, np.inf # We don't have the fitted value, need to evaluate

        # Main Loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Sample from multivariate normal
                z = np.random.multivariate_normal(np.zeros(dim), C)
                x_new = x_mean + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_made += 1
                offspring.append((x_new, f_new))
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    
                # Add to recent points for paraboloid fitting
                recent_points.append((x_new, f_new))
                if len(recent_points) > 20:
                    recent_points.pop(0)

            if len(offspring) < lambda_:
                break

            # Sort offspring
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i][0]
            
            # Update covariance matrix
            diff = [offspring[i][0] - x_mean for i in range(mu)]
            C_new = C
            for i in range(mu):
                C_new += weights[i] * np.outer(diff[i], diff[i])
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.dot(weights, [offspring[i][0] - x_mean for i in range(mu)])
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - np.exp(-2 * c_c)) * C) + c_mu * np.dot(weights, [np.outer(d, d) for d in diff])
            
            # Normalize C to keep determinant 1? CMA-ES usually normalizes C such that det(C) = 1.
            # This helps with stability.
            D_new = np.sqrt(np.linalg.det(C))
            C = C / (D_new / D) ** (2.0 / dim)
            D = D_new
            
            x_mean = x_new_mean
            
            # Paraboloid Snap Check
            # If we have converged somewhat (sigma small) and have enough points, try snapping.
            if sigma < 0.5 and len(recent_points) >= dim + 1:
                # Check if recent points are close to each other
                pts = np.array([p[0] for p in recent_points[-(dim+1):]])
                spread = np.max(pts, axis=0) - np.min(pts, axis=0)
                if np.all(spread < 1.0):
                    # Try to find a better point using local gradient estimation
                    # Fit linear model to last dim+1 points
                    A = np.hstack([pts, np.ones((dim+1, 1))])
                    vals = np.array([p[1] for p in recent_points[-(dim+1):]])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
                        grad = coeffs[:dim]
                        center = np.mean(pts, axis=0)
                        # Newton step assuming Hessian ~ I
                        x_snap = center - grad
                        x_snap = np.clip(x_snap, -5, 5)
                        
                        if calls_made < budget and time.time() < deadline:
                            f_snap = func(x_snap)
                            calls_made += 1
                            if f_snap < best_f:
                                best_f = f_snap
                                best_x = x_snap.copy()
                                # If snap was successful, reset CMA-ES to this point?
                                # Or just continue?
                                # Let's update mean to this point to exploit it.
                                x_mean = x_snap
                                sigma = 0.1 # Reduce sigma for local search
                                C = np.eye(dim)
                                p_sigma = np.zeros(dim)
                                p_c = np.zeros(dim)
                                D = 1.0
                                recent_points = [] # Reset recent points
                    except:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
