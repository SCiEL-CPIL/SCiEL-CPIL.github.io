import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: (1+1)-Evolution Strategy for rapid initial convergence
        # Fast early progress is critical for AOCC. The 1.5/0.87 adaptation
        # rates provide good balance between exploration and exploitation.
        x = np.zeros(dim)
        fx = evaluate(x)
        if fx is None:
            return best_x, best_f
            
        sigma = 1.0
        es_budget = min(120, budget - 1)
        
        for _ in range(es_budget):
            if calls >= budget or time.time() > deadline:
                break
            
            x_trial = x + sigma * np.random.randn(dim)
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = evaluate(x_trial)
            if f_trial is None:
                break
            
            if f_trial < fx:
                x = x_trial
                fx = f_trial
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            if time.time() > deadline:
                break

        # Phase 2: Nelder-Mead Simplex Method for refinement
        # NM's multi-point simplex better tracks the sharp ridge geometry
        # than a single-point ES, enabling finer convergence on the ridge.
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        x_start = x.copy()
        
        # Initial spread: use the current sigma (clamped) to match the ES scale
        # but ensure it's not too small or too large
        initial_spread = max(0.1, min(1.0, sigma))
        
        simplex = []
        simplex.append(x_start.copy())
        for i in range(dim):
            point = x_start.copy()
            point[i] += initial_spread
            point = np.clip(point, -5, 5)
            simplex.append(point)
        
        # Evaluate all simplex vertices
        simplex_f = []
        for point in simplex:
            fx_i = evaluate(point)
            if fx_i is None:
                return best_x, best_f
            simplex_f.append(fx_i)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Run Nelder-Mead until budget or deadline
        while calls < budget and time.time() < deadline:
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = evaluate(x_reflect)
            if f_reflect is None:
                break
            
            if f_reflect < simplex_f[0]:
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = evaluate(x_expand)
                if f_expand is None:
                    break
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            elif f_reflect < simplex_f[-2]:
                simplex[-1] = x_reflect
                simplex_f[-1] = f_reflect
            else:
                if f_reflect < simplex_f[-1]:
                    x_contract = centroid + rho * (x_reflect - centroid)
                else:
                    x_contract = centroid - rho * (x_reflect - centroid)
                
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = evaluate(x_contract)
                if f_contract is None:
                    break
                
                if f_contract < min(f_reflect, simplex_f[-1]):
                    simplex[-1] = x_contract
                    simplex_f[-1] = f_contract
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_new = evaluate(simplex[i])
                        if f_new is None:
                            return best_x, best_f
                        simplex_f[i] = f_new
            
            if time.time() > deadline:
                break
        
        return best_x, best_f
