import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = time.time() + 285
        
        def eval_func(x):
            if self.calls >= budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # 1. Initial Sampling
        # Use a mix of random and LHS-like points
        # For dim=5, 10 initial points is reasonable.
        n_init = 10
        initial_points = []
        initial_vals = []
        
        # Fixed seed for reproducibility in the run, but random for different instances?
        # The task says "hidden seeded shift". We should use random points.
        # We don't have a seed provided to us, so we use np.random default.
        
        for _ in range(n_init):
            if self.calls >= budget or time.time() > self.deadline:
                break
            x = np.random.uniform(-5, 5, size=dim)
            f = eval_func(x)
            if np.isfinite(f):
                initial_points.append(x)
                initial_vals.append(f)
        
        if not initial_points:
            return self.best_x, self.best_f
            
        # Sort by fitness
        sorted_indices = np.argsort(initial_vals)
        initial_points = [initial_points[i] for i in sorted_indices]
        
        # 2. Multi-Start Nelder-Mead
        # We will run NM on the best 3 starts.
        # Budget allocation:
        # Remaining budget after init: budget - self.calls
        rem_budget = budget - self.calls
        
        # Allocation: 50%, 30%, 20%
        b1 = int(rem_budget * 0.5)
        b2 = int(rem_budget * 0.3)
        b3 = rem_budget - b1 - b2
        
        def nelder_mead(x0, max_calls):
            # Check if we have enough budget for simplex init (dim+1 points)
            # We already have x0 evaluated? 
            # In the previous code, x0 was one of the initial points, so it was already evaluated.
            # But to be safe and simple, we will assume x0 is NOT in the simplex yet, 
            # OR we can reuse the value. 
            # Reusing is better.
            # However, the simplex needs dim+1 points.
            # We have x0. We need dim more points.
            # Cost: dim evaluations.
            
            if self.calls + dim > budget or time.time() > self.deadline:
                return x0, self.best_f
                
            simplex = [x0.copy()]
            # We need the value of x0. We don't store it in a local variable here easily 
            # without passing it. 
            # Let's just re-evaluate x0 to keep the code simple and safe? 
            # No, that wastes 1 call. 
            # Better: Pass f0 as well.
            # But the signature is nelder_mead(x0, max_calls).
            # Let's modify to accept f0.
            
            # Actually, let's just evaluate x0 again. It's 1 call out of 500. 
            # Or, we can just start the simplex with x0 and generate dim new points.
            # We need f0.
            # If we don't have f0, we must evaluate it.
            # Since x0 comes from initial_points, we know its value from initial_vals.
            # But we didn't pass it.
            
            # Let's change the call to pass f0.
            
            pass # Placeholder

        # Let's rewrite nelder_mead properly inside the call
        def nm_run(x0, f0, max_calls):
            # max_calls is the budget for the *loop*, not including init?
            # Let's define max_calls as the total budget available for this restart, 
            # including the simplex construction if we haven't done it yet.
            # Here, x0 is already evaluated (f0 is known).
            # We need to construct the simplex: x0 + dim perturbations.
            
            if self.calls + dim > budget or time.time() > self.deadline:
                return x0, self.best_f

            simplex = [x0.copy()]
            f_values = [f0]
            
            # Construct simplex
            for i in range(dim):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                xi = x0.copy()
                # Step size
                step = 0.5 # 10% of range [-5,5] is 1.0. Let's use 0.5 for finer start.
                if abs(x0[i]) < 1e-8:
                    xi[i] = step
                else:
                    # Scale step based on magnitude? 
                    # Standard NM uses 1.0 if 0, else 1.1*x.
                    # Let's use 0.1 absolute step for better precision in 5D box.
                    # Actually, 1.0 is better for global structure.
                    xi[i] = x0[i] + step if x0[i] >= 0 else x0[i] - step
                xi = np.clip(xi, -5, 5)
                fi = eval_func(xi)
                if np.isinf(fi):
                    fi = self.best_f + 1000 # Penalize
                simplex.append(xi)
                f_values.append(fi)
                
            if len(simplex) < dim + 1:
                return x0, self.best_f

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Budget for the main loop
            # We spent 'dim' calls for simplex construction.
            # 'max_calls' was allocated for the whole restart.
            # Let's say max_calls includes the construction?
            # In the allocation above, b1, b2, b3 are total budgets for the restarts.
            # So, remaining_loop_budget = max_calls - dim (if we consider f0 free)
            # But f0 was already evaluated in the init phase.
            # So cost so far for this restart is 'dim' (for the new points).
            # Loop budget = max_calls - dim.
            
            loop_budget = max_calls - dim
            if loop_budget <= 0:
                return x0, self.best_f

            for _ in range(loop_budget):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                
                # Sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = eval_func(x_reflect)
                
                if np.isinf(f_reflect):
                    # If we hit budget, break
                    break

                if f_reflect < f_values[0]:
                    # Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    # Check budget before expand
                    if self.calls < budget and time.time() < self.deadline:
                        f_expand = eval_func(x_expand)
                        if np.isinf(f_expand):
                            break
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            f_values[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            f_values[-1] = f_reflect
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                        break
                elif f_reflect < f_values[-2]:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = x_centroid + rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    if self.calls < budget and time.time() < self.deadline:
                        f_contract = eval_func(x_contract)
                        if np.isinf(f_contract):
                            break
                        if f_contract < f_worst:
                            simplex[-1] = x_contract
                            f_values[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if self.calls < budget and time.time() < self.deadline:
                                    f_values[i] = eval_func(simplex[i])
                                    if np.isinf(f_values[i]):
                                        break
                                else:
                                    break
                            if self.calls >= budget or time.time() >= self.deadline:
                                break
                    else:
                        break

                # Convergence check
                if max(f_values) - min(f_values) < 1e-12:
                    break

            return x0, self.best_f

        # Run NM on top starts
        # We need to pass f0.
        # initial_points and initial_vals are sorted.
        
        if len(initial_points) > 0:
            nm_run(initial_points[0], initial_vals[0], b1)
        if len(initial_points) > 1 and self.calls < budget:
            nm_run(initial_points[1], initial_vals[1], b2)
        if len(initial_points) > 2 and self.calls < budget:
            nm_run(initial_points[2], initial_vals[2], b3)
            
        return self.best_x, self.best_f
