import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Differential Evolution
        # Use 80% of budget for DE
        de_budget = int(budget * 0.8)
        pop_size = 10
        if de_budget < pop_size:
            pop_size = de_budget
            
        # Initialize population
        # Mix of random and center-biased to help early convergence
        x0 = rng.uniform(-5, 5, (pop_size, dim))
        # Force one individual to be near center
        x0[0] = rng.uniform(-1, 1, dim)
        
        f_x0 = np.array([func(x) for x in x0])
        best_idx = np.argmin(f_x0)
        best_x = x0[best_idx].copy()
        best_f = f_x0[best_idx]
        calls_made = pop_size
        
        F = 0.8
        CR = 0.5
        
        # DE Loop
        while calls_made < de_budget:
            # If we don't have enough budget for a full generation, break
            if calls_made + pop_size > de_budget:
                break
                
            new_x0 = x0.copy()
            new_f_x0 = f_x0.copy()
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                a, b, c = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation
                v = x0[a] + F * (x0[b] - x0[c])
                
                # Crossover
                u = x0[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                        
                # Clip to box
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= f_x0[i]:
                    new_x0[i] = u
                    new_f_x0[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
            x0 = new_x0
            f_x0 = new_f_x0

        # Phase 2: Nelder-Mead Polish
        # Use remaining budget
        remaining = budget - calls_made
        
        if remaining > 0:
            # Implement a simple Nelder-Mead
            # Initialize simplex around best_x
            simplex = np.zeros((dim + 1, dim))
            f_simplex = np.zeros(dim + 1)
            
            simplex[0] = best_x
            f_simplex[0] = best_f
            
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                # Perturb one dimension
                step = 0.5 if best_x[i-1] > 0 else -0.5
                simplex[i][i-1] += step
                simplex[i] = np.clip(simplex[i], -5, 5)
                f_simplex[i] = func(simplex[i])
                calls_made += 1
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()
                
                if calls_made >= budget:
                    break

            # NM Iterations
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while calls_made < budget:
                # Sort simplex
                sort_idx = np.argsort(f_simplex)
                simplex = simplex[sort_idx]
                f_simplex = f_simplex[sort_idx]
                
                # Check convergence (optional, but we are budget-constrained)
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Replace worst
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made < budget:
                                f_simplex[i] = func(simplex[i])
                                calls_made += 1
                                if f_simplex[i] < best_f:
                                    best_f = f_simplex[i]
                                    best_x = simplex[i].copy()
                            else:
                                break
                        if calls_made >= budget:
                            break

        return best_x, best_f
