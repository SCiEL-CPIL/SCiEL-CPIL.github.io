import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Parallel Tempering with CMA-ES instances
        n_replicas = 8
        calls_per_replica = budget // n_replicas
        
        # Temperatures: lower temperature = more exploitation, higher = more exploration
        # Use a geometric sequence of temperatures
        temps = np.geomspace(0.1, 5.0, n_replicas)

        # Initialize CMA-ES instances
        # Each replica has its own state
        # We use a simplified CMA-ES implementation for speed and reliability
        
        def cma_es_step(func, x, sigma, B, D, pc, ps, f_best, best_x, budget_left, dim, lambda_):
            """
            Perform one step of CMA-ES. Returns updated state and number of calls used.
            """
            calls_used = 0
            if budget_left <= 0:
                return x, sigma, B, D, pc, ps, f_best, best_x, calls_used

            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))
            n_c = 1.0 / (c_c * (dim + 2.0))

            # Sample lambda_ points
            z = np.random.randn(lambda_, dim)
            y = z @ (B * D)  # Transform
            x_new = np.array([x + sigma * y[i] for i in range(lambda_)])

            # Evaluate
            f_new = np.empty(lambda_)
            for i in range(lambda_):
                if calls_used >= budget_left or not time_ok():
                    break
                f_new[i] = func(x_new[i])
                calls_used += 1
                if f_new[i] < f_best:
                    f_best = f_new[i]
                    best_x = x_new[i].copy()

            if calls_used < lambda_:
                # Pad with inf for un-evaluated points
                f_new[calls_used:] = np.inf

            # Sort
            order = np.argsort(f_new)
            y_sorted = y[order]
            f_sorted = f_new[order]
            x_sorted = x_new[order]

            # Update mean
            x_new_mean = x + sigma * (weights[:mu] @ y_sorted[:mu])

            # Update covariance
            yw = y_sorted[:mu] / sigma
            # Rank-one update
            pc = (1 - c_sigma) * pc + c_sigma * (x_new_mean - x) / sigma
            # Rank-mu update
            C = None  # We don't store C explicitly for speed, we use B*D
            # For simplicity and speed, we skip full covariance matrix update
            # and just update the step size and rotation via B, D approximation
            # This is a simplified version. A full CMA-ES would update C.
            # However, for this task, even a simplified version with rotation
            # via B and D might be sufficient. Let's implement a proper update.
            
            # Actually, implementing full CMA-ES covariance update is complex.
            # Let's use a simpler approach: Nelder-Mead with multiple starts
            # but with a smarter restart strategy and maybe some randomization.
            
            # Given the complexity of CMA-ES and the risk of bugs, let's revert to
            # a robust multi-start Nelder-Mead but with better initial points
            # and a dynamic step size. The previous score was 0.8412.
            # The inspiration was 0.9517 using CMA-ES.
            # Let's try to implement a correct, fast CMA-ES.

            # To save time and ensure correctness, I will implement a standard CMA-ES
            # but keep it simple.

            # For now, let's just return the current state if we didn't do a full update.
            # This function is a placeholder for the full CMA-ES logic.
            # I will restructure the main loop to use a simpler, robust method.
            
            return x, sigma, B, D, pc, ps, f_best, best_x, calls_used

        # Since implementing full CMA-ES is risky in this constrained environment,
        # I will use a Multi-Start Nelder-Mead with a more sophisticated
        # restart strategy based on the best found so far and random perturbations.
        # I will also use a larger number of starts with smaller budgets,
        # and then a final polishing phase.

        best_x = np.zeros(dim)
        best_f = func(best_x)
        total_calls = 1

        # Phase 1: Global exploration with many short Nelder-Mead runs
        n_global = 20
        calls_per_global = 20  # 20 * 20 = 400 calls
        remaining = budget - total_calls

        if remaining > 0:
            for i in range(n_global):
                if total_calls >= budget or not time_ok():
                    break
                
                # Generate a random start point in [-5, 5]^dim
                # Use a quasi-random sequence for better coverage
                # Simple LCG for reproducibility
                seed = 42 + i
                rng = np.random.RandomState(seed)
                start_point = rng.uniform(-5, 5, dim)
                
                # Run a short Nelder-Mead
                initial_step = 1.0
                simplex = [start_point.copy()]
                for d in range(dim):
                    point = start_point.copy()
                    point[d] += initial_step
                    simplex.append(point)
                
                f_simplex = []
                for point in simplex:
                    if total_calls >= budget or not time_ok():
                        break
                    f_val = func(point)
                    total_calls += 1
                    f_simplex.append(f_val)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = point.copy()
                
                if len(f_simplex) < dim + 1:
                    continue

                # Nelder-Mead iterations
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                max_iters = calls_per_global - (dim + 1)
                
                for _ in range(max_iters):
                    if total_calls >= budget or not time_ok():
                        break
                    
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst_point = simplex[-1]
                    reflect_point = centroid + alpha * (centroid - worst_point)
                    
                    if total_calls >= budget or not time_ok():
                        break
                    f_reflect = func(reflect_point)
                    total_calls += 1
                    if f_reflect < best_f:
                        best_f = f_reflect
                        best_x = reflect_point.copy()
                    
                    if f_simplex[0] <= f_reflect < f_simplex[-2]:
                        simplex[-1] = reflect_point
                        f_simplex[-1] = f_reflect
                    elif f_reflect < f_simplex[0]:
                        expand_point = centroid + gamma * (reflect_point - centroid)
                        if total_calls >= budget or not time_ok():
                            break
                        f_expand = func(expand_point)
                        total_calls += 1
                        if f_expand < best_f:
                            best_f = f_expand
                            best_x = expand_point.copy()
                        
                        if f_expand < f_reflect:
                            simplex[-1] = expand_point
                            f_simplex[-1] = f_expand
                        else:
                            simplex[-1] = reflect_point
                            f_simplex[-1] = f_reflect
                    else:
                        if f_reflect < f_simplex[-1]:
                            contract_point = centroid + rho * (reflect_point - centroid)
                        else:
                            contract_point = centroid - rho * (worst_point - centroid)
                        
                        if total_calls >= budget or not time_ok():
                            break
                        f_contract = func(contract_point)
                        total_calls += 1
                        if f_contract < best_f:
                            best_f = f_contract
                            best_x = contract_point.copy()
                        
                        if f_contract < min(f_reflect, f_simplex[-1]):
                            simplex[-1] = contract_point
                            f_simplex[-1] = f_contract
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if total_calls >= budget or not time_ok():
                                    break
                                f_shrink = func(simplex[i])
                                total_calls += 1
                                f_simplex[i] = f_shrink
                                if f_shrink < best_f:
                                    best_f = f_shrink
                                    best_x = simplex[i].copy()
                            break

        # Phase 2: Local polishing around the best point found
        if total_calls < budget and time_ok():
            polish_calls = budget - total_calls
            # Start Nelder-Mead from the best point with a small step
            start_point = best_x.copy()
            initial_step = 0.1
            simplex = [start_point.copy()]
            for d in range(dim):
                point = start_point.copy()
                point[d] += initial_step
                simplex.append(point)
            
            f_simplex = []
            for point in simplex:
                if total_calls >= budget or not time_ok():
                    break
                f_val = func(point)
                total_calls += 1
                f_simplex.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = point.copy()
            
            if len(f_simplex) < dim + 1:
                return best_x, best_f

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iters = polish_calls - (dim + 1)
            
            for _ in range(max_iters):
                if total_calls >= budget or not time_ok():
                    break
                
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                reflect_point = centroid + alpha * (centroid - worst_point)
                
                if total_calls >= budget or not time_ok():
                    break
                f_reflect = func(reflect_point)
                total_calls += 1
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = reflect_point.copy()
                
                if f_simplex[0] <= f_reflect < f_simplex[-2]:
                    simplex[-1] = reflect_point
                    f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[0]:
                    expand_point = centroid + gamma * (reflect_point - centroid)
                    if total_calls >= budget or not time_ok():
                        break
                    f_expand = func(expand_point)
                    total_calls += 1
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = expand_point.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = expand_point
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = reflect_point
                        f_simplex[-1] = f_reflect
                else:
                    if f_reflect < f_simplex[-1]:
                        contract_point = centroid + rho * (reflect_point - centroid)
                    else:
                        contract_point = centroid - rho * (worst_point - centroid)
                    
                    if total_calls >= budget or not time_ok():
                        break
                    f_contract = func(contract_point)
                    total_calls += 1
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = contract_point.copy()
                    
                    if f_contract < min(f_reflect, f_simplex[-1]):
                        simplex[-1] = contract_point
                        f_simplex[-1] = f_contract
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if total_calls >= budget or not time_ok():
                                break
                            f_shrink = func(simplex[i])
                            total_calls += 1
                            f_simplex[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                        break

        return best_x, best_f
