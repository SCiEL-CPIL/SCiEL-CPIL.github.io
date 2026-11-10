import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f1, f2):
        return np.all(f2 <= f1) and np.any(f2 < f1)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominating = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if self._is_dominated(F[p], F[q]):
                    S[p].append(q)
                elif self._is_dominated(F[q], F[p]):
                    n_dominating[p] += 1
            if n_dominating[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dominating[q] -= 1
                    if n_dominating[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break
        return fronts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Initial LHS
        N_init = 50
        X = np.zeros((N_init, dim))
        for d in range(dim):
            perm = np.random.permutation(N_init)
            X[:, d] = (perm + np.random.rand(N_init)) / N_init
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        evals = N_init
        
        # Archive of non-dominated points
        # We will maintain a list of non-dominated points
        # and use local search to improve them.
        
        # To keep it simple and fast, we maintain the population and filter.
        # But the structural change is: Local Search on the Front.
        
        # Let's use a "Pareto Local Search" strategy.
        # 1. Get current non-dominated front.
        # 2. For each point in the front, try a local move.
        # 3. Accept if it improves (non-dominated).
        
        # To avoid getting stuck, we also keep a small random population for diversity.
        # But with 200 evals, we can't keep a large population.
        # Let's stick to refining the best points.
        
        # Strategy:
        # While evals < budget:
        #   Identify non-dominated points in current set X, F.
        #   Pick one (randomly or round-robin).
        #   Generate neighbor via Gaussian perturbation.
        #   Evaluate.
        #   If neighbor is better (non-dominated), replace or add.
        
        # To ensure diversity, we might need to keep some points that are dominated but diverse?
        # No, for hypervolume, non-dominated is best.
        
        # Let's try a simpler structural change: **Memetic NSGA-II**.
        # After each generation, apply a local search to the top few non-dominated individuals.
        
        # However, the previous attempt was NSGA-II + DE.
        # Let's try **MOEA/D** style with local search.
        
        # Actually, let's just switch to a **Local Search** based on the initial LHS points.
        # This is a very different family: Direct Local Search vs Population-Based Evolution.
        
        # 1. Initialize LHS points.
        # 2. Evaluate.
        # 3. Sort by hypervolume contribution or just pick non-dominated.
        # 4. Perform Nelder-Mead or Simplex on the best points.
        
        # Given the low budget, Nelder-Mead is expensive.
        # Let's use **Coordinate Descent** or **Random Perturbation** local search.
        
        # We will maintain a set of "active" points (initially LHS).
        # We will iteratively pick a point, perturb it, and keep the best.
        
        active_X = X.copy()
        active_F = F.copy()
        
        # We will use the remaining budget to perform local searches.
        # Each local search step takes 1 evaluation.
        # We can do (budget - evals) steps.
        
        # To prevent converging to a single point, we will keep the best non-dominated set.
        # If a new point is dominated, we discard it.
        # If it is non-dominated, we add it.
        
        # This is essentially **Pareto Local Search**.
        
        # To improve diversity, we can periodically re-initialize some points?
        # No, let's stick to the structural change: Local Search.
        
        # We need to be careful with the time limit.
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points in active_X
            fronts = self._non_dominated_sort(active_F)
            if not fronts or not fronts[0]:
                break
                
            # Pick a random point from the first front
            idx = np.random.choice(fronts[0])
            x_current = active_X[idx]
            
            # Generate a neighbor
            # Use a decaying step size?
            # Or fixed small step?
            # Let's use a step size relative to the bound range.
            scale = 0.1 # 10% of range
            step = np.random.normal(0, scale, dim)
            x_new = x_current + step
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            evals += 1
            
            # Check if x_new is non-dominated w.r.t active_X
            # If it dominates some points, remove them.
            # If it is dominated by any active point, discard.
            
            is_dominated = False
            for i in range(len(active_X)):
                if self._is_dominated(f_new, active_F[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove points dominated by x_new
                new_X = []
                new_F = []
                for i in range(len(active_X)):
                    if not self._is_dominated(active_F[i], f_new):
                        new_X.append(active_X[i])
                        new_F.append(active_F[i])
                
                new_X.append(x_new)
                new_F.append(f_new)
                
                active_X = np.array(new_X)
                active_F = np.array(new_F)
            
            # Optional: If the population grows too large, prune?
            # With 200 evals, it won't grow too large.
            
        return active_X
