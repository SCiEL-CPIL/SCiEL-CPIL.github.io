"""Hybrid SPSA + Nelder-Mead Optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        # Helper to check time
        def time_ok():
            return time.time() < deadline

        # Helper to evaluate with budget tracking
        calls_used = [0]
        def eval_func(x):
            if time_ok() and calls_used[0] < budget:
                calls_used[0] += 1
                try:
                    return float(func(x))
                except:
                    return float('inf')
            return float('inf')

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Initial Exploration: Quick random sampling to find a good starting point
        # This helps with AOCC early progress by ensuring we don't start from a terrible point
        n_init = 5
        rng = np.random.default_rng(42)
        
        # Sample center and a few random points
        candidates = [np.zeros(dim)]
        for _ in range(n_init - 1):
            candidates.append(rng.uniform(-5, 5, dim))
            
        for x in candidates:
            fx = eval_func(x)
            update_best(x, fx)
            
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        # 2. Phase 1: SPSA (Simultaneous Perturbation Stochastic Approximation)
        # SPSA estimates gradient with 2 evals per step.
        # It is very efficient for smooth functions and makes fast initial progress.
        # We run it for about 50% of the remaining budget.
        
        x_spsa = best_x.copy()
        f_spsa = best_f
        
        # SPSA parameters
        a = 0.5 * (dim + 1) ** 0.6  # Step size coefficient
        c = 0.1 * (dim + 1) ** 0.2  # Perturbation coefficient
        gamma = 0.6
        alpha = 0.6
        
        n_spsa_steps = int(0.5 * (budget - calls_used[0]))
        
        for k in range(n_spsa_steps):
            if not time_ok() or calls_used[0] >= budget:
                break
            
            # Decay parameters
            a_k = a / (k + 1) ** alpha
            c_k = c / (k + 1) ** gamma
            
            # Generate random perturbation direction (±1)
            delta = rng.choice([-1, 1], size=dim)
            
            # Evaluate at x + c_k * delta
            x_plus = np.clip(x_spsa + c_k * delta, -5, 5)
            f_plus = eval_func(x_plus)
            
            # Evaluate at x - c_k * delta
            x_minus = np.clip(x_spsa - c_k * delta, -5, 5)
            f_minus = eval_func(x_minus)
            
            # Estimate gradient: (f_plus - f_minus) / (2 * c_k * delta)
            # Update step: -a_k * grad_est
            # g_est = (f_plus - f_minus) / (2 * c_k) * (1/delta)
            # Since delta is ±1, 1/delta = delta
            # step = -a_k * (f_plus - f_minus) / (2 * c_k) * delta
            
            if f_plus != float('inf') and f_minus != float('inf'):
                grad_est = (f_plus - f_minus) / (2 * c_k) * delta
                step = -a_k * grad_est
                x_new = np.clip(x_spsa + step, -5, 5)
                f_new = eval_func(x_new)
                
                if f_new < f_spsa:
                    x_spsa = x_new
                    f_spsa = f_new
                    update_best(x_spsa, f_spsa)
                # If not improved, we still update x_spsa in standard SPSA, 
                # but for robustness in black-box optimization, we can be conservative.
                # However, standard SPSA updates regardless to explore. 
                # Let's stick to standard SPSA update but keep track of best.
                # Actually, if we don't update x_spsa when it's worse, it becomes a hill climber.
                # SPSA is stochastic, so it should explore. 
                # But given the strict budget and need for progress, let's only accept improvements 
                # or accept with a small probability to escape local minima? 
                # No, SPSA's stochasticity is in the gradient estimate. 
                # Let's use the standard update rule: x_{k+1} = x_k - a_k * g_k
                # But if it goes to a bad place, we lose the best. 
                # Strategy: Keep x_spsa as the "current" point for SPSA, but always track global best.
                # If SPSA gets stuck, we restart from global best.
                
                # Let's refine: Use a "restart" mechanism if SPSA stagnates
                # For now, just do the standard update.
                x_spsa = x_new
            else:
                # If one side is inf, just try the other or break
                break

        # 3. Phase 2: Nelder-Mead Simplex Method
        # Nelder-Mead is excellent for local polishing and finding the precise minimum.
        # It uses a simplex of dim+1 points.
        # We use the remaining budget for this.
        
        x_nm = best_x.copy()
        f_nm = best_f
        
        # Initialize Simplex
        # Start with best_x and perturb each coordinate slightly
        simplex = [x_nm.copy()]
        simplex_f = [f_nm]
        
        # Small perturbation for initial simplex vertices
        pert = 0.1
        for i in range(dim):
            x_i = x_nm.copy()
            x_i[i] += pert
            x_i = np.clip(x_i, -5, 5)
            f_i = eval_func(x_i)
            simplex.append(x_i)
            simplex_f.append(f_i)
            update_best(x_i, f_i)
            
        # Sort simplex by function value
        def sort_simplex(simplex, simplex_f):
            indices = np.argsort(simplex_f)
            return [simplex[i] for i in indices], [simplex_f[i] for i in indices]
        
        simplex, simplex_f = sort_simplex(simplex, simplex_f)
        
        # Nelder-Mead Parameters
        alpha_r = 1.0   # Reflection
        gamma_e = 2.0   # Expansion
        rho_c = 0.5     # Contraction
        sigma_s = 0.5   # Shrinkage
        
        n_nm_steps = budget - calls_used[0]
        
        for _ in range(n_nm_steps):
            if not time_ok() or calls_used[0] >= budget:
                break
            
            # 1. Reflection
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_x = simplex[-1]
            worst_f = simplex_f[-1]
            
            x_reflect = centroid + alpha_r * (centroid - worst_x)
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = eval_func(x_reflect)
            
            if f_reflect < simplex_f[0]: # Better than best
                # 2. Expansion
                x_expand = centroid + gamma_e * (centroid - worst_x)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = eval_func(x_expand)
                
                if f_expand < f_reflect:
                    x_new = x_expand
                    f_new = f_expand
                else:
                    x_new = x_reflect
                    f_new = f_reflect
            elif f_reflect < worst_f: # Better than worst, but not best
                x_new = x_reflect
                f_new = f_reflect
            else: # Worse than worst
                # 3. Contraction
                # Outside contraction
                x_contract_out = centroid + 0.5 * (worst_x - centroid)
                x_contract_out = np.clip(x_contract_out, -5, 5)
                f_contract_out = eval_func(x_contract_out)
                
                if f_contract_out < worst_f:
                    x_new = x_contract_out
                    f_new = f_contract_out
                else:
                    # Inside contraction
                    x_contract_in = centroid - 0.5 * (centroid - worst_x)
                    x_contract_in = np.clip(x_contract_in, -5, 5)
                    f_contract_in = eval_func(x_contract_in)
                    
                    if f_contract_in < worst_f:
                        x_new = x_contract_in
                        f_new = f_contract_in
                    else:
                        # 4. Shrinkage
                        # Shrink all points except the best towards the best
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = eval_func(simplex[i])
                        
                        simplex, simplex_f = sort_simplex(simplex, simplex_f)
                        continue # Skip the rest of the loop for this iteration
            
            # Replace the worst point with the new point
            simplex[-1] = x_new
            simplex_f[-1] = f_new
            update_best(x_new, f_new)
            
            # Re-sort
            simplex, simplex_f = sort_simplex(simplex, simplex_f)
            
            # Check for convergence (simple check)
            if len(simplex_f) > 1 and abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                break

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
