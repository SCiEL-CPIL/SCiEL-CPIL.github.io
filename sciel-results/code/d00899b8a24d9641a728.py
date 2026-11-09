"""Multi-objective optimizer on WFG7 using Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


class Optimizer:
    """Bayesian Optimization for multi-objective WFG7 with GP surrogate and EHVI-like acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3.0, 5.0, 7.0])
        
        # Normalize to [0,1] for GP
        def norm(x):
            return (x - lo) / (hi - lo + 1e-12)
        
        def denorm(xn):
            return lo + xn * (hi - lo)
        
        X_raw = []
        F_raw = []
        X_norm = []
        F_norm = []
        
        used = 0
        rng = np.random.default_rng(0)
        
        def evaluate(x_raw):
            nonlocal used
            x_clipped = np.clip(x_raw, lo, hi)
            f = np.array(problem(x_clipped), dtype=np.float64)
            X_raw.append(x_clipped)
            F_raw.append(f)
            X_norm.append(norm(x_clipped))
            F_norm.append(f)
            used += 1
            return f
        
        # Phase 1: Initial random sampling
        n_init = 30
        for i in range(n_init):
            if used >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        # Build initial GP-like surrogate using simple kernel regression
        # We'll use a weighted average approach as a lightweight surrogate
        
        def fit_surrogate(Xn, Fn):
            """Fit a simple kernel regression surrogate."""
            Xn = np.array(Xn)
            Fn = np.array(Fn)
            return Xn, Fn
        
        def predict(Xn, X_train, F_train, length_scale=0.1, noise=0.01):
            """GP-like prediction using RBF kernel. Returns mean and std."""
            X_train = np.array(X_train)
            F_train = np.array(F_train)
            Xn = np.array(Xn)
            
            # RBF kernel
            d = cdist(Xn, X_train, 'sqeuclidean')
            K = np.exp(-d / (2 * length_scale ** 2))
            
            # Simple kernel regression (no full GP for speed)
            weights = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            mean = weights @ F_train
            # Estimate uncertainty
            var = np.exp(-d.min(axis=1) / (2 * length_scale ** 2))
            std = np.sqrt(1.0 - var) * np.std(F_train, axis=0) + noise
            return mean, std
        
        def hypervolume_contribution(f, F_pareto, ref):
            """Compute hypervolume contribution of point f given existing Pareto front."""
            if np.any(f >= ref):
                return 0.0
            # Simple approximation: box volume that f dominates not already covered
            # Use a rough estimate based on distance to nearest Pareto point
            if len(F_pareto) == 0:
                return np.prod(ref - f)
            # Find if f is dominated
            dominated = np.any(np.all(F_pareto <= f, axis=1) & np.any(F_pareto < f, axis=1))
            if dominated:
                return 0.0
            # Estimate contribution as volume of box from f to ref minus overlap
            vol = np.prod(ref - f)
            # Subtract overlap with existing Pareto points (rough)
            for fp in F_pareto:
                if np.all(fp < ref) and np.all(f < ref):
                    overlap = np.prod(np.minimum(ref, np.maximum(f, fp)) - np.minimum(f, fp))
                    if overlap > 0:
                        vol -= overlap
            return max(0.0, vol)
        
        def get_pareto_front(F):
            """Get non-dominated points."""
            F = np.array(F)
            n = len(F)
            if n == 0:
                return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return [F[i] for i in range(n) if not dominated[i]]
        
        # Phase 2: BO loop
        n_bo_samples = 20  # samples for acquisition function
        n_bo_candidates = 50  # candidate points to evaluate acquisition
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            Xn = np.array(X_norm)
            Fn = np.array(F_norm)
            
            # Generate candidate points
            candidates = []
            # Random candidates
            for _ in range(n_bo_candidates // 2):
                candidates.append(rng.random(self.dim))
            # Perturb existing points
            for _ in range(n_bo_candidates // 2):
                idx = rng.integers(len(Xn))
                c = Xn[idx] + 0.1 * rng.standard_normal(self.dim)
                c = np.clip(c, 0, 1)
                candidates.append(c)
            
            candidates = np.array(candidates)
            
            # Predict for candidates
            mean, std = predict(candidates, Xn, Fn)
            
            # Compute acquisition: simplified EHVI via sampling
            F_pareto = get_pareto_front(Fn)
            best_acq = -np.inf
            best_cand = candidates[0]
            
            for i in range(len(candidates)):
                # Sample from posterior
                f_sample = mean[i] + std[i] * rng.standard_normal(self.n_obj)
                # Clip to reasonable range
                f_sample = np.clip(f_sample, 0, ref + 1)
                # Estimate hypervolume contribution
                contrib = hypervolume_contribution(f_sample, F_pareto, ref)
                # Add exploration bonus based on uncertainty
                expl_bonus = np.mean(std[i]) * 0.1
                acq = contrib + expl_bonus
                if acq > best_acq:
                    best_acq = acq
                    best_cand = candidates[i]
            
            # Evaluate best candidate
            x_raw = denorm(best_cand)
            evaluate(x_raw)
            
            # Periodically refit with updated data
            # (surrogate is refit implicitly in each iteration)
        
        return X_raw
# EVOLVE-BLOCK-END
