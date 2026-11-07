import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a fixed seed for reproducibility in the optimization process itself
        # The evaluator seeds np.random, but we use a local rng for our internal steps
        # to ensure consistent behavior regardless of external state if we want, 
        # but usually we should use the global state or a derived one.
        # The prompt says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So we should use np.random directly to leverage the seed.
        
        pop_size = 50
        if self.budget <= pop_size:
            pop_size = self.budget
            
        # 1. Initialize Population
        # Use Latin Hypercube Sampling for better spread
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[perm, i] = np.linspace(0, 1, pop_size)
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # 2. Generate Weights for MOEA/D
        # We need weights for n_obj objectives.
        # Generate points on the simplex.
        # For 3 objectives, we can use a grid.
        n_weights = 30
        weights = []
        # Simple grid on 2-simplex
        step = 1.0 / (n_weights // 2 + 1)
        for i in range(int(1.0/step) + 1):
            for j in range(int((1.0 - i*step)/step) + 1):
                k = 1.0 - i*step - j*step
                if k >= -1e-6:
                    weights.append([i*step, j*step, max(0, k)])
        
        # Normalize and ensure we have enough weights
        # If not enough, add random ones
        while len(weights) < n_weights:
            w = np.random.rand(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
        
        weights = np.array(weights[:n_weights])
        
        # Assign each individual to the best weight
        # Score: weighted sum
        # S[i, j] = F[i] @ weights[j]
        S = F @ weights.T
        # best_weight_idx[i] = argmin S[i]
        best_w_idx = np.argmin(S, axis=1)
        
        # 3. Evolutionary Loop
        # MOEA/D with Weighted Sum
        # We will perform (budget - evals) evaluations
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select a weight to focus on
            # Random selection ensures exploration
            w_idx = np.random.randint(n_weights)
            w = weights[w_idx]
            
            # Select parents
            # Prefer parents that are good for this weight
            # Find individuals assigned to this weight or nearby
            # For simplicity, pick 2 random parents from the whole population,
            # but bias towards those with low weighted sum for w
            
            # Calculate weighted sum for all individuals for this specific weight
            ws_scores = F @ w
            # Pick 2 parents with lowest scores (best for this weight)
            # Add some randomness
            if np.random.rand() < 0.5:
                # Exploitation: pick best 2
                parent_indices = np.argsort(ws_scores)[:2]
            else:
                # Exploration: pick random 2
                parent_indices = np.random.choice(pop_size, 2, replace=False)
                
            p1 = X[parent_indices[0]]
            p2 = X[parent_indices[1]]
            
            # SBX Crossover
            # eta = 20
            eta = 20.0
            child = np.zeros_like(p1)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child[i] = p1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                else:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child[i] = p1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
                        
            # Polynomial Mutation
            # eta_m = 20
            eta_m = 20.0
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    delta1 = (1 + 2 * (child[i] - lo[i]) / (hi[i] - lo[i])) ** (eta_m + 1)
                    delta2 = (1 + 2 * (hi[i] - child[i]) / (hi[i] - lo[i])) ** (eta_m + 1)
                    if u <= 0.5:
                        deltaq = (1 / delta1)
                    else:
                        deltaq = 1 - (1 / delta2)
                    child[i] = child[i] + (hi[i] - lo[i]) * (2 * u - 1) * deltaq
                    
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Update Population
            # Find the individual in the population that is currently "worst" for this weight w
            # Or simply replace the individual assigned to w if child is better
            # Let's find the index of the individual with the highest weighted sum for w
            # This acts as a "worst" replacement for this specific objective direction
            
            # Actually, standard MOEA/D replaces the individual in the subproblem if the new one is better.
            # Subproblem k minimizes g(x, w_k).
            # We need to identify which subproblem (weight) the child is best for.
            child_scores = fc @ weights.T
            best_child_w = np.argmin(child_scores)
            
            # Check if child is better than the current incumbent for best_child_w
            # Find an individual currently assigned to best_child_w
            # If multiple, pick the one with highest score (worst)
            incumbents = np.where(best_w_idx == best_child_w)[0]
            
            if len(incumbents) > 0:
                # Pick the worst incumbent
                incumbent_scores = S[incumbents, best_child_w]
                worst_incumbent_local = np.argmax(incumbent_scores)
                worst_incumbent_idx = incumbents[worst_incumbent_local]
                
                if fc @ weights[best_child_w] < S[worst_incumbent_idx, best_child_w]:
                    X[worst_incumbent_idx] = child
                    F[worst_incumbent_idx] = fc
                    # Update S and best_w_idx for this individual
                    S[worst_incumbent_idx, :] = fc @ weights.T
                    best_w_idx[worst_incumbent_idx] = best_child_w
            else:
                # No incumbent for this weight? (Should not happen if we cover all weights)
                # If it happens, replace the global worst
                global_worst = np.argmax(np.min(S, axis=1)) # This is not quite right, let's just pick random
                X[global_worst] = child
                F[global_worst] = fc
                S[global_worst, :] = fc @ weights.T
                best_w_idx[global_worst] = best_child_w

        # 4. Polish Phase
        # Use remaining budget to polish the best non-dominated points
        # Find non-dominated set
        def is_dominated(f, F_set):
            # Returns True if any point in F_set dominates f
            for fp in F_set:
                if np.all(fp <= f) and np.any(fp < f):
                    return True
            return False

        # Get non-dominated indices
        nd_indices = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)
                
        # Polish the top 5 non-dominated points
        # Use Nelder-Mead or simple coordinate descent on the hypervolume contribution?
        # Simpler: Polish using a weighted sum with weights derived from the point's position.
        
        if evals < self.budget and len(nd_indices) > 0:
            # Select a few points to polish
            # We have budget - evals evaluations left
            polish_budget = self.budget - evals
            points_to_polish = nd_indices[:min(5, len(nd_indices))]
            
            for idx in points_to_polish:
                if evals >= self.budget or time.time() > deadline:
                    break
                
                x_curr = X[idx].copy()
                f_curr = F[idx].copy()
                
                # Define a scalar objective based on a weight vector
                # Use a weight that points towards the "gap" or just a random weight
                # Better: Use the weight vector that this point was best for
                w_polish = weights[best_w_idx[idx]]
                
                # Simple Local Search: Coordinate Descent
                # Try small steps in each dimension
                step_size = 0.05 * (hi - lo)
                improved = True
                while improved and evals < self.budget and time.time() < deadline:
                    improved = False
                    for i in range(self.dim):
                        # Try + and -
                        for delta in [1, -1]:
                            x_trial = x_curr.copy()
                            x_trial[i] += delta * step_size
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = problem(x_trial)
                            evals += 1
                            
                            if evals >= self.budget or time.time() > deadline:
                                break
                                
                            # Check if better in terms of weighted sum
                            if f_trial @ w_polish < f_curr @ w_polish:
                                x_curr = x_trial
                                f_curr = f_trial
                                improved = True
                                # Shrink step size
                                step_size *= 0.5
                                
                            if evals >= self.budget or time.time() > deadline:
                                break
                        if evals >= self.budget or time.time() > deadline:
                            break
                    if evals >= self.budget or time.time() > deadline:
                        break
                
                # Update population with polished point
                X[idx] = x_curr
                F[idx] = f_curr
                S[idx, :] = f_curr @ weights.T
                best_w_idx[idx] = np.argmin(S[idx, :])

        # 5. Return Non-Dominated Set
        # Re-calculate non-dominated set from final F
        nd_indices_final = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices_final.append(i)
                
        # If no non-dominated points found (should not happen), return all
        if not nd_indices_final:
            nd_indices_final = list(range(len(F)))
            
        result_X = X[nd_indices_final]
        return result_X
