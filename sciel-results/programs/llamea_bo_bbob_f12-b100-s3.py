"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Nelder-Mead Simplex Method with adaptive restarts.
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned and has a narrow valley.
    Nelder-Mead is a derivative-free local optimizer that works well on smooth functions.
    While it can struggle with global exploration, the hidden shift/rotation means the
    minimum is still a single point in a smooth landscape. 
    
    Key structural difference from CMA-ES:
    - Uses a simplex (triangle/tetrahedron) of points rather than a Gaussian distribution.
    - Performs local geometric operations (reflection, expansion, contraction) rather than
      statistical updates to mean and covariance.
    - Can be more aggressive in exploiting local structure once close to the minimum.
    
    To handle the global aspect (unknown shift), we use a multi-start strategy:
    1. Start with a few random initial points.
    2. Run Nelder-Mead from the best one.
    3. If it converges (small simplex size), restart from a new random point or the best found.
    
    This approach leverages the fact that the bent cigar, while ill-conditioned, is still
    a single-basin function. A good local optimizer can converge very quickly once in the
    basin, potentially outperforming CMA-ES in the final convergence phase if the initial
    point is not too far off.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Nelder-Mead Parameters
        # Initial simplex size
        initial_step = 1.0
        # Reflection, expansion, contraction, shrink coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5
        # Tolerance for convergence
        tol = 1e-10
        
        # Multi-start strategy
        # We will try several random starting points.
        # If one converges quickly, we might restart or just continue.
        # Given the budget is 500, and NM uses dim+1 = 6 points per iteration,
        # we can afford roughly 500/6 ~ 83 iterations per start.
        # Let's do up to 5 restarts if the first ones don't converge well.
        
        max_restarts = 5
        budget_per_restart = budget // max_restarts
        
        for restart in range(max_restarts):
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            
            # Initialize simplex
            # Start from a random point in [-3, 3] to stay within bounds but explore
            x0 = np.random.uniform(-3, 3, dim)
            
            # Build simplex
            simplex = []
            f_values = []
            
            # First vertex
            simplex.append(x0.copy())
            f_values.append(wrapped_func(x0))
            
            # Other vertices: x0 + step * e_i
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += initial_step
                simplex.append(x_i)
                f_values.append(wrapped_func(x_i))
            
            # Sort simplex by fitness
            simplex = np.array(simplex)
            f_values = np.array(f_values)
            sort_idx = np.argsort(f_values)
            simplex = simplex[sort_idx]
            f_values = f_values[sort_idx]
            
            # Nelder-Mead Loop
            converged = False
            while self.calls < budget and time.time() < self.deadline:
                # Check convergence: if the spread of the simplex is small
                # and the function value variation is small
                if np.max(f_values) - np.min(f_values) < tol * (1 + np.abs(np.min(f_values))):
                    converged = True
                    break
                
                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                f_reflect = wrapped_func(x_reflect)
                
                if f_reflect < f_values[0]:
                    # If reflection is better than best, try expansion
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    f_expand = wrapped_func(x_expand)
                    
                    if f_expand < f_reflect:
                        # Accept expansion
                        simplex[-1] = x_expand
                        f_values[-1] = f_expand
                    else:
                        # Accept reflection
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                elif f_reflect < f_values[-2]:
                    # If reflection is better than second worst, accept it
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                else:
                    # Otherwise, try contraction
                    if f_reflect < f_values[-1]:
                        # Outside contraction
                        x_contract = centroid + 0.5 * (x_reflect - centroid)
                        f_contract = wrapped_func(x_contract)
                        
                        if f_contract <= f_reflect:
                            simplex[-1] = x_contract
                            f_values[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                f_values[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        x_contract = centroid - 0.5 * (simplex[-1] - centroid)
                        f_contract = wrapped_func(x_contract)
                        
                        if f_contract < f_values[-1]:
                            simplex[-1] = x_contract
                            f_values[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                f_values[i] = wrapped_func(simplex[i])
                
                # Re-sort simplex
                sort_idx = np.argsort(f_values)
                simplex = simplex[sort_idx]
                f_values = f_values[sort_idx]
                
                # Check for stall (simplex size)
                simplex_size = np.linalg.norm(simplex[0] - simplex[-1])
                if simplex_size < 1e-8:
                    converged = True
                    break
            
            if not converged:
                # If we ran out of budget or time, break
                if self.calls >= budget or time.time() > self.deadline:
                    break
                # If converged or stalled, restart from a new random point
                # But we need to be careful not to waste too much budget
                # If we've used more than 80% of budget, stop restarting
                if self.calls > 0.8 * budget:
                    break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
