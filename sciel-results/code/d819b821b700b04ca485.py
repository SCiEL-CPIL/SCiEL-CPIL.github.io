"""Multi-objective optimizer for RE37 using GP-surrogate MBEA with NSGA-II fallback."""
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                if np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-15:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initialization
        # Use Halton sequence for better coverage than random
        n_init = min(20, n_budget)
        X_init = np.zeros((n_init, dim))
        for i in range(1, n_init + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                X_init[i - 1, d] = result
        
        X_init = lo + X_init * span
        
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: GP Surrogate Model-Based Optimization (MBEA)
        # We implement a simple GP using RBF kernel.
        
        def matern_kernel(X, Y, length_scale=0.5, variance=1.0, nu=2.5):
            # X: (n, d), Y: (m, d)
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            else:
                return variance * np.exp(-0.5 * dist**2 / length_scale**2)

        class SimpleGP:
            def __init__(self, length_scale=0.5, variance=1.0, nu=2.5):
                self.length_scale = length_scale
                self.variance = variance
                self.nu = nu
                self.X = None
                self.y = None
                self.K = None
                self.alpha = None

            def fit(self, X, y):
                self.X = X
                self.y = y
                self.K = matern_kernel(X, X, self.length_scale, self.variance, self.nu)
                self.K += 1e-6 * np.eye(len(X))
                try:
                    self.alpha = np.linalg.solve(self.K, y)
                except:
                    self.alpha = np.linalg.lstsq(self.K, y, rcond=None)[0]

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance, self.nu)
                mean = K_s @ self.alpha
                # Approximate std for exploration
                # var = k(x,x) - K_s @ K^{-1} @ K_s.T
                # To save time, we use a heuristic or simple approximation if n is large
                # For n < 100, we can compute it.
                try:
                    K_inv = np.linalg.inv(self.K)
                    K_s_K_inv = K_s @ K_inv
                    var = self.variance - np.sum(K_s * K_s_K_inv, axis=1)
                    var = np.maximum(var, 0.0)
                    std = np.sqrt(var)
                except:
                    std = np.zeros(len(X_new))
                return mean, std

        # Initialize GP models
        X_elite = np.array(X_all)
        F_elite = np.array(F_all)
        
        gps = []
        for m in range(n_obj):
            gp = SimpleGP(length_scale=0.5, variance=1.0, nu=2.5)
            gp.fit(X_elite, F_elite[:, m])
            gps.append(gp)

        # Main Loop
        # We will use a loop that proposes points based on the GP model.
        # Strategy: 
        # 1. Generate a set of candidate points (random + local perturbations).
        # 2. Evaluate GP mean and std for each candidate.
        # 3. Use an acquisition function (e.g., Expected Hypervolume Improvement approximation or simple EI) to select the best point.
        # 4. Evaluate the selected point.
        # 5. Update GP.
        
        # Since EHVI is expensive, we use a simpler acquisition: 
        # Minimize the weighted sum of predicted means, with a small exploration term.
        # Or, we can use a "greedy" approach: pick the point that minimizes the predicted 
        # objective values, considering uncertainty.
        
        # Let's try a simpler approach: 
        # At each step, generate a batch of candidates, score them using GP predictions, 
        # and pick the best one.
        
        n_candidates = 100
        
        while eval_count < n_budget and time.time() < deadline:
            # Generate candidates
            # Mix of random and local perturbations around current Pareto front
            n_rand = n_candidates // 2
            n_local = n_candidates - n_rand
            
            candidates = []
            
            # Random
            candidates.append(lo + span * np.random.random((n_rand, dim)))
            
            # Local perturbations of current Pareto front
            fronts = self._fast_non_dominated_sort(F_elite)
            if fronts:
                front0 = fronts[0]
                if len(front0) > 0:
                    # Pick a few from front 0
                    sample_size = min(len(front0), 5)
                    idxs = np.random.choice(len(front0), size=sample_size, replace=False)
                    for idx in idxs:
                        x_base = X_elite[front0[idx]]
                        # Generate several perturbations
                        for _ in range(n_local // max(1, sample_size)):
                            pert = 0.1 * span * np.random.randn(dim)
                            candidates.append(x_base + pert)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lo, hi)
            
            # Evaluate acquisition function
            # Predict means and stds
            means = np.zeros((len(candidates), n_obj))
            stds = np.zeros((len(candidates), n_obj))
            for m in range(n_obj):
                means[:, m], stds[:, m] = gps[m].predict(candidates)
            
            # Score: We want to minimize objectives.
            # Use a weighted sum with uncertainty exploration.
            # weights = 1.0 for all objectives (or normalize)
            # score = -sum(means) + beta * sum(stds)
            beta = 0.5
            scores = -np.sum(means, axis=1) + beta * np.sum(stds, axis=1)
            
            # Select the best candidate
            best_idx = np.argmax(scores)
            x_next = candidates[best_idx]
            
            f_next = evaluate(x_next)
            if f_next is None:
                break
            
            # Update elites and GP
            X_elite = np.vstack([X_elite, x_next])
            F_elite = np.vstack([F_elite, f_next])
            
            # Refit GPs
            for m in range(n_obj):
                gps[m].fit(X_elite, F_elite[:, m])
        
        # Phase 3: Final Polish (optional, if time remains)
        # We can do a few more random evaluations to ensure we haven't missed anything
        while eval_count < n_budget and time.time() < deadline:
            x = lo + span * np.random.random(dim)
            evaluate(x)
            
        return np.array(X_all)
