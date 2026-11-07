import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors on Simplex - structured grid
        # For 3 objectives, use H=5 -> 21 points
        H = 5
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors - 6 nearest for better exploration
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 7) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using LHS
        X = np.zeros((N, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        budget_for_main = budget - 30  # Reserve 30 for polish
        
        i_counter = 0
        gen = 0
        
        while used < budget_for_main:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            gen += 1
            
            # Adaptive mutation factor: start higher, decrease over time
            progress = used / max(budget_for_main, 1)
            F_mut = 0.9 - 0.3 * progress  # 0.9 -> 0.6
            
            # Select parents from neighbors
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                dists_to_i = np.array([np.linalg.norm(weights[c] - weights[i]) for c in candidate_indices])
                probs = 1.0 / (dists_to_i + 1e-6)
                probs = probs / probs.sum()
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False, p=probs)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            CR = 0.5
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = np.array(problem(trial))
            used += 1
            
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population (MOEA/D with Tchebycheff)
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial
        
        # 5. Local polish phase - weighted sum local search with hill climbing
        if time.time() < deadline - 10:
            # Find non-dominated points
            non_dom_idx = []
            for i in range(N):
                dominated = False
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)
            
            # Sort by hypervolume contribution (approximate: distance to ideal)
            if non_dom_idx:
                # Use simple local search: perturb and keep if better in weighted sum
                evals_per_point = max(1, (budget - used) // max(len(non_dom_idx), 1))
                evals_per_point = min(evals_per_point, 8)  # Cap at 8 per point
                
                for idx in non_dom_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    x0 = X[idx].copy()
                    best_x = x0.copy()
                    best_f = F[idx].copy()
                    w = weights[idx]
                    best_val = np.dot(w, best_f)
                    
                    # Adaptive step size local search
                    step = 0.1
                    for _ in range(evals_per_point):
                        if used >= budget:
                            break
                        if time.time() > deadline - 3:
                            break
                        
                        # Try random perturbation
                        perturbation = np.random.uniform(-step, step, dim)
                        new_x = np.clip(best_x + perturbation, lo, hi)
                        
                        new_f = np.array(problem(new_x))
                        used += 1
                        new_val = np.dot(w, new_f)
                        
                        if new_val < best_val:
                            best_val = new_val
                            best_x = new_x.copy()
                            best_f = new_f.copy()
                            step *= 0.7  # Shrink step on success
                        else:
                            step *= 0.8  # Shrink step on failure too, slowly
                    
                    # Update population
                    if best_val < np.dot(w, F[idx]):
                        X[idx] = best_x
                        F[idx] = best_f
                        z_star = np.minimum(z_star, best_f)
        
        # 6. Final refinement: try to improve each point with tiny perturbations
        if time.time() < deadline - 2:
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                
                w = weights[i]
                best_val = np.dot(w, F[i])
                best_x = X[i].copy()
                best_f = F[i].copy()
                
                # Tiny step local search
                for _ in range(2):
                    if used >= budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    
                    perturbation = np.random.uniform(-0.02, 0.02, dim)
                    new_x = np.clip(best_x + perturbation, lo, hi)
                    new_f = np.array(problem(new_x))
                    used += 1
                    new_val = np.dot(w, new_f)
                    
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        best_f = new_f.copy()
                
                if best_val < np.dot(w, F[i]):
                    X[i] = best_x
                    F[i] = best_f
                    z_star = np.minimum(z_star, best_f)
        
        return X
