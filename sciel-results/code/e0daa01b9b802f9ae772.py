import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Deterministic multi-scale approach
        # Start with large steps and geometrically decay
        # Use coordinate descent-like deterministic moves
        
        # Phase 1: Coarse global search with deterministic steps
        # Try several starting points and do local refinement
        
        # Geometric sequence of step sizes from 4.0 to 1e-4
        n_scales = 20
        step_sizes = np.geomspace(4.0, 1e-4, n_scales)
        
        # For each scale, do a few iterations of deterministic local search
        # Use a deterministic set of directions (coordinate axes + diagonals)
        
        # Initialize at center
        x = np.zeros(dim)
        f = evaluate(x)
        
        # Deterministic directions: coordinate axes (both signs) and some diagonals
        directions = []
        for d in range(dim):
            e = np.zeros(dim)
            e[d] = 1.0
            directions.append(e)
            directions.append(-e)
        
        # Add some diagonal directions
        for d1 in range(dim):
            for d2 in range(d1+1, dim):
                e = np.zeros(dim)
                e[d1] = 1.0
                e[d2] = 1.0
                directions.append(e)
                e = np.zeros(dim)
                e[d1] = 1.0
                e[d2] = -1.0
                directions.append(e)
        
        # Normalize directions
        directions = [d / np.linalg.norm(d) for d in directions]
        
        # Multi-scale deterministic search
        for scale_idx, step in enumerate(step_sizes):
            if calls >= budget or time.time() > deadline:
                break
            
            # Do multiple passes at this scale
            max_passes = max(1, (budget - calls) // (len(directions) * 3))
            max_passes = min(max_passes, 10)
            
            for _pass in range(max_passes):
                if calls >= budget or time.time() > deadline:
                    break
                
                improved = False
                for d in directions:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try moving in this direction
                    x_trial = x + step * d
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                    else:
                        # Try opposite direction
                        x_trial2 = x - step * d
                        f_trial2 = evaluate(x_trial2)
                        if f_trial2 is None:
                            break
                        if f_trial2 < f:
                            x = x_trial2
                            f = f_trial2
                            improved = True
                
                # If no improvement in this pass, we can stop early for this scale
                if not improved:
                    break
        
        # Phase 2: Fine-tuning with smaller steps if budget remains
        if calls < budget and time.time() < deadline:
            # Do a final Nelder-Mead-like refinement
            simplex = [x.copy()]
            for d in range(dim):
                v = x.copy()
                v[d] += 0.01
                simplex.append(v)
            
            simplex_f = [evaluate(v) if v is not x else f for v in simplex]
            # Handle the first point which we already evaluated
            simplex_f[0] = f
            
            # Simplex update
            for _ in range(min(50, (budget - calls) // (dim + 1))):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_ref = centroid + 1.0 * (centroid - worst)
                f_ref = evaluate(x_ref)
                if f_ref is None:
                    break
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + 2.0 * (centroid - worst)
                    f_exp = evaluate(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-1]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (worst - centroid)
                    f_con = evaluate(x_con)
                    if f_con is None:
                        break
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
                            if simplex_f[i] is None:
                                break
                        if simplex_f[-1] is None:
                            break
        
        return best_x, best_f
