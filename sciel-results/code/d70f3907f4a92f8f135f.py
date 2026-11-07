"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization: Space-filling points
        n_starts = 10
        rng = np.random.default_rng(42)
        
        simplexes = []
        calls = 0
        global_best_x = np.zeros(dim) # Initialize to avoid None issues
        global_best_f = np.inf
        
        # Initialize simplexes
        for i in range(n_starts):
            # Center of simplex
            if i == 0:
                center = np.zeros(dim)
            else:
                center = rng.uniform(-5, 5, dim)
            
            # Initialize simplex vertices
            # Vertex 0 is center
            # Other vertices are center + delta * e_i
            delta = 0.5 # Initial step size
            simplex_points = [center.copy()]
            for j in range(dim):
                v = center.copy()
                v[j] += delta
                simplex_points.append(v)
            
            # Evaluate all vertices
            simplex_fvals = []
            for p in simplex_points:
                if calls >= budget:
                    return global_best_x, global_best_f
                p_clipped = np.clip(p, -5, 5)
                fval = func(p_clipped)
                calls += 1
                simplex_fvals.append(fval)
                
                if fval < global_best_f:
                    global_best_f = fval
                    global_best_x = p_clipped.copy()
            
            # Store simplex state
            # Order points by fval
            points_fvals = list(zip(simplex_points, simplex_fvals))
            points_fvals.sort(key=lambda x: x[1])
            
            simplexes.append({
                'points': [p for p, f in points_fvals],
                'fvals': [f for p, f in points_fvals],
                'active': True
            })
        
        if calls >= budget:
            return global_best_x, global_best_f

        # 2. Main Loop: Nelder-Mead steps
        while calls < budget and time.time() < deadline:
            # Select simplex with lowest best f
            active_indices = [i for i, s in enumerate(simplexes) if s['active']]
            if not active_indices:
                break
            
            # Find index of active simplex with min fvals[0]
            best_idx = min(active_indices, key=lambda i: simplexes[i]['fvals'][0])
            simplex = simplexes[best_idx]
            
            # Perform one Nelder-Mead iteration
            # Points are sorted: P[0] is best, P[-1] is worst
            P = simplex['points']
            F = simplex['fvals']
            
            # Centroid of all points except the worst
            centroid = np.sum(P[:-1], axis=0) / dim
            
            # 1. Reflection
            alpha = 1.0
            x_r = centroid + alpha * (centroid - P[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if calls >= budget: break
            f_r = func(x_r)
            calls += 1
            
            if f_r < global_best_f:
                global_best_f = f_r
                global_best_x = x_r.copy()
            
            if f_r < F[0]:
                # 2. Expansion
                gamma = 2.0
                x_e = centroid + gamma * (centroid - P[-1])
                x_e = np.clip(x_e, -5, 5)
                
                if calls >= budget: break
                f_e = func(x_e)
                calls += 1
                
                if f_e < global_best_f:
                    global_best_f = f_e
                    global_best_x = x_e.copy()
                
                if f_e < f_r:
                    P[-1] = x_e
                    F[-1] = f_e
                else:
                    P[-1] = x_r
                    F[-1] = f_r
            elif f_r < F[-2]:
                # Accept reflection
                P[-1] = x_r
                F[-1] = f_r
            else:
                # 3. Contraction
                rho = 0.5
                # Outer contraction
                x_c = centroid + rho * (P[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                
                if calls >= budget: break
                f_c = func(x_c)
                calls += 1
                
                if f_c < global_best_f:
                    global_best_f = f_c
                    global_best_x = x_c.copy()
                
                if f_c < f_r:
                    P[-1] = x_c
                    F[-1] = f_c
                else:
                    # 4. Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, dim + 1):
                        if calls >= budget: break
                        P[i] = P[0] + 0.5 * (P[i] - P[0])
                        P[i] = np.clip(P[i], -5, 5)
                        f_shrink = func(P[i])
                        calls += 1
                        if f_shrink < global_best_f:
                            global_best_f = f_shrink
                            global_best_x = P[i].copy()
                        F[i] = f_shrink
            
            # Re-sort the simplex
            points_fvals = list(zip(P, F))
            points_fvals.sort(key=lambda x: x[1])
            P = [p for p, f in points_fvals]
            F = [f for p, f in points_fvals]
            
            simplex['points'] = P
            simplex['fvals'] = F
            
            # Check for convergence of this simplex
            if max(F) - min(F) < 1e-10:
                simplex['active'] = False

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
