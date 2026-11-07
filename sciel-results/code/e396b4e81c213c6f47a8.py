import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Generate weight vectors
            # For 3 objectives, we can use a predefined set of 10-15 vectors
            # Or generate a grid.
            # Let's use 10 vectors to keep budget per vector at 20.
            weights = np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.5, 0.3, 0.2],
                [0.5, 0.2, 0.3],
                [0.3, 0.5, 0.2],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.2, 0.3, 0.5],
                [1/3, 1/3, 1/3]
            ])
            
            n_weights = len(weights)
            budget_per_weight = self.budget // n_weights
            
            all_evals = [] # Not needed, problem() records them? 
            # "Every returned objective vector is recorded"
            # The problem function returns the vector. The evaluator records it.
            # So we just need to call problem().
            
            for w in weights:
                remaining = budget_per_weight
                if remaining < self.dim + 1:
                    # Not enough for a simplex, just random search
                    for _ in range(remaining):
                        x = lo + (hi - lo) * np.random.random(self.dim)
                        problem(x)
                    continue
                
                # Initialize Nelder-Mead Simplex
                # 1 point at center, others perturbed
                center = lo + (hi - lo) * 0.5
                simplex = np.empty((self.dim + 1, self.dim))
                simplex[0] = center
                
                # Perturbations
                step_size = 0.25 * (hi - lo)
                for i in range(1, self.dim + 1):
                    simplex[i] = center.copy()
                    simplex[i][i-1] += step_size[i-1]
                    
                # Ensure within bounds
                simplex = np.clip(simplex, lo, hi)
                
                # Evaluate simplex
                F_simplex = np.array([problem(x) for x in simplex])
                remaining -= (self.dim + 1)
                
                # Scalarize
                scalar_F = F_simplex @ w
                
                # Nelder-Mead Loop
                # Standard parameters
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                # Indices sorted by scalar_F
                idx = np.argsort(scalar_F)
                simplex = simplex[idx]
                scalar_F = scalar_F[idx]
                
                while remaining > 0:
                    # Current best and worst
                    # Find worst index (last in sorted array)
                    worst_idx = self.dim
                    best_idx = 0
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[worst_idx])
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = problem(x_ref) @ w
                    remaining -= 1
                    
                    if f_ref < scalar_F[0]:
                        # Expansion
                        if remaining == 0:
                            break
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = problem(x_exp) @ w
                        remaining -= 1
                        
                        if f_exp < f_ref:
                            new_x = x_exp
                            new_f = f_exp
                        else:
                            new_x = x_ref
                            new_f = f_ref
                    elif f_ref < scalar_F[-1]: # scalar_F[-1] is worst
                        new_x = x_ref
                        new_f = f_ref
                    else:
                        # Contraction
                        if remaining == 0:
                            break
                        # Inside contraction
                        x_con = centroid + rho * (simplex[worst_idx] - centroid)
                        x_con = np.clip(x_con, lo, hi)
                        f_con = problem(x_con) @ w
                        remaining -= 1
                        
                        if f_con < scalar_F[-1]:
                            new_x = x_con
                            new_f = f_con
                        else:
                            # Shrink
                            if remaining == 0:
                                break
                            # Shrink towards best
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                # Evaluate shrunk points? 
                                # Standard Nelder-Mead evaluates them.
                                # This uses dim evaluations.
                                # If remaining < dim, we can't do full shrink.
                                # Let's just break if we can't afford shrink.
                                pass 
                            # Simplified: if we can't shrink, just break or accept worst
                            # Let's just break to save complexity and avoid budget overrun
                            break
                            
                    # Update simplex
                    simplex[worst_idx] = new_x
                    scalar_F[worst_idx] = new_f
                    
                    # Re-sort
                    idx = np.argsort(scalar_F)
                    simplex = simplex[idx]
                    scalar_F = scalar_F[idx]
                    
                # End of weight loop
            return None
