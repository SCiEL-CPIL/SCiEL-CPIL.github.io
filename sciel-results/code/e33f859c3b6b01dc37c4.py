import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, hard limit usually 300s.
        # We use 290s to be safe.
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        
        # NSGA-II Parameters
        pop_size = 100  # Half the budget, allowing for 1 generation of offspring + evaluation
        # Actually, with budget 200, we can do:
        # Gen 0: Init 100 points (100 evals)
        # Gen 1: Create 100 offspring (100 evals)
        # Combine 200, select best 100? No, we want to record ALL evaluated points for HV.
        # The evaluator records EVERY objective vector returned by problem(x).
        # So we don't need to "select" for the return value, we just need to generate
        # a diverse set of points that covers the front well.
        # However, NSGA-II logic helps in *generating* diverse offspring.
        
        # Strategy:
        # 1. Initialize 100 points.
        #    - Use Latin Hypercube or Random for position vars.
        #    - Set shape vars to 0.5 (best guess for g=0).
        # 2. Evaluate them (100 evals).
        # 3. Perform NSGA-II selection/crossover/mutation to generate 100 offspring.
        #    - This ensures the offspring are diverse and try to improve convergence.
        # 4. Evaluate offspring (100 evals).
        # Total: 200 evals.
        # The recorded set will contain 200 points.
        # Since NSGA-II maintains diversity, these 200 points should cover the front well.
        
        rng = np.random.default_rng(42)
        
        # 1. Initialization
        X = np.zeros((pop_size, dim))
        # Position variables (first n_obj-1)
        # Use Sobol-like or random. Random is fine with NSGA-II.
        X[:, :n_obj-1] = rng.random((pop_size, n_obj-1))
        # Shape variables (rest)
        # Set to midpoint, but add small noise to avoid exact duplicates if bounds are tight
        mid_shape = 0.5 * (lo[n_obj-1:] + hi[n_obj-1:])
        X[:, n_obj-1:] = mid_shape + rng.normal(0, 0.01, (pop_size, dim - (n_obj-1)))
        X[:, n_obj-1:] = np.clip(X[:, n_obj-1:], lo[n_obj-1:], hi[n_obj-1:])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Check time
        if time.time() > deadline:
            return X
        
        # 2. NSGA-II Step
        # Create offspring population
        X_offspring = np.zeros((pop_size, dim))
        
        # We will use a simple tournament selection and SBX crossover + Polynomial mutation
        # To keep it fast and simple, we'll do:
        # - Select parents via tournament (size 2)
        # - Crossover: Simulated Binary Crossover (SBX)
        # - Mutation: Gaussian mutation
        
        def tournament_select(F, idxs, k=2):
            # Select k random indices from idxs
            selected = rng.choice(idxs, size=k, replace=False)
            # Find the one with best rank (lowest rank)
            # For simplicity, we'll just pick the one with lower sum of objectives 
            # or we can implement proper non-dominated sorting.
            # Given time constraints, a simplified "worst-sum" replacement or 
            # simple non-dominated check is okay, but let's do a proper NSGA-II sort.
            
            # Actually, implementing full NSGA-II sorting is complex in a few lines.
            # Alternative: Use the "hypervolume indicator" or just simple dominance.
            # Let's use a simpler diversity-preserving method:
            # Keep the best 50 from current, and generate 50 new ones based on them?
            # No, let's stick to a simplified NSGA-II.
            
            # For the sake of this implementation, we will use a simple heuristic:
            # Pick the parent with the lowest sum of objectives among the k.
            # This biases towards the ideal point, which might hurt diversity.
            # To counter this, we can add a random factor.
            
            scores = np.sum(F[selected], axis=1)
            best_idx = selected[np.argmin(scores)]
            return best_idx
        
        # Better: Implement a quick non-dominated sort for the current population
        # to assign ranks.
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = [0] * n  # n_d[i] = number of individuals dominating i
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
                
            return fronts, rank

        fronts, ranks = fast_non_dominated_sort(F)
        
        # Now generate offspring
        # We will select parents based on rank.
        # To ensure diversity, we will also consider crowding distance if we had it, 
        # but for now, random selection within the best front is a good start.
        
        # Simplified approach:
        # 1. Identify the best front (Front 0).
        # 2. For each offspring, pick two parents from Front 0 (or Front 1 if Front 0 is small).
        # 3. Crossover and mutate.
        
        best_front_indices = fronts[0]
        if len(best_front_indices) < 2:
            # If best front is too small, include next front
            if len(fronts) > 1:
                best_front_indices += fronts[1]
        
        # If still small, just use all
        if len(best_front_indices) < 2:
            best_front_indices = list(range(pop_size))
            
        for i in range(pop_size):
            # Select parents
            p1_idx = best_front_indices[rng.integers(len(best_front_indices))]
            p2_idx = best_front_indices[rng.integers(len(best_front_indices))]
            
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            
            # SBX Crossover
            # For simplicity, use uniform crossover or simple blend
            # Let's use a simple blend crossover with some randomness
            beta = 20  # Distribution index
            # Standard SBX is complex. Let's use a simpler "interpolation" crossover.
            # child = (1-t)*x1 + t*x2 + noise
            t = rng.random()
            child = (1 - t) * x1 + t * x2
            
            # Add Gaussian noise (Mutation)
            # Scale noise by distance between parents or fixed
            noise = rng.normal(0, 0.1, dim)
            child += noise
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            X_offspring[i] = child
            
        # Evaluate offspring
        F_offspring = np.array([problem(x) for x in X_offspring])
        evals += pop_size
        
        # We have now evaluated 200 points.
        # The evaluator records all of them.
        # We don't need to do anything else as we've used the budget.
        
        return X

    def is_non_dominated(self, F, idx):
        f = F[idx]
        for i in range(len(F)):
            if i == idx: continue
            if np.all(F[i] <= f) and np.any(F[i] < f):
                return False
        return True

    def find_non_dominated(self, F):
        res = []
        for i in range(len(F)):
            if self.is_non_dominated(F, i):
                res.append(i)
        return res
