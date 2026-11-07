import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n_restarts = 10
        budget_per_restart = self.budget // n_restarts
        budgets = [budget_per_restart] * n_restarts
        budgets[-1] = self.budget - (n_restarts - 1) * budget_per_restart
        
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            delta = 1.0
            
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for i in range(self.dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += delta
            simplex = np.clip(simplex, -5, 5)
            
            fvals = np.array([func(x) for x in simplex])
            calls = self.dim + 1
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls < budgets[r]:
                if time.time() > deadline:
                    break
                
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                
                if calls >= budgets[r]:
                    # We used the last call for reflection. Update simplex if better than worst.
                    if f_ref < fvals[-1]:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                        
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    
                else:
                    # Contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if calls >= budgets[r]:
                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        break

                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrinkage
                        # We need N calls. Check if we have enough budget.
                        if calls + self.dim > budgets[r]:
                            # Not enough budget for full shrinkage. 
                            # We can shrink partially or just break.
                            # Standard NM requires all N points. 
                            # We'll break to avoid exceeding budget.
                            # The current simplex is valid.
                            break
                        
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

            # End of restart
            current_best_f = np.min(fvals)
            current_best_x = simplex[np.argmin(fvals)]
            
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = current_best_x

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
