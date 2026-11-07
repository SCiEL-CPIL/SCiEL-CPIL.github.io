"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Weighted Sum + L-BFGS-B hybrid with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Store all evaluated points and their objective vectors
        X = []  # list of x vectors
        F = []  # list of f vectors
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        # Phase 1: Latin Hypercube Sampling for diverse initialization (30 points)
        n_lhs = 30
        if n_lhs <= self.budget:
            for i in range(n_lhs):
                if time.time() > deadline:
                    break
                # LHS: divide each dimension into n_lhs equal intervals, pick one per interval
                x = np.zeros(self.dim)
                for d in range(self.dim):
                    interval = i % n_lhs
                    x[d] = lo[d] + (interval + rng.random()) * (hi[d] - lo[d]) / n_lhs
                eval_point(x)

        # Phase 2: Weighted Sum + L-BFGS-B scalar optimizations
        # Generate diverse weight vectors to cover the objective space
        n_scalar = 10
        if F and time.time() < deadline and used < self.budget:
            # Create weight vectors: simplex-lattice style for 3 objectives
            weight_sets = []
            # Include single-objective weights
            for i in range(self.n_obj):
                w = np.zeros(self.n_obj)
                w[i] = 1.0
                weight_sets.append(w)
            
            # Include some mixed weights
            mixed_weights = [
                [0.5, 0.25, 0.25],
                [0.25, 0.5, 0.25],
                [0.25, 0.25, 0.5],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.3, 0.3, 0.4],
            ]
            for w in mixed_weights:
                weight_sets.append(np.array(w))
            
            # Pick up to n_scalar weight sets
            selected_weights = weight_sets[:n_scalar]
            
            # For each weight vector, pick a diverse starting point and run L-BFGS-B
            F_arr = np.array(F)
            ref = np.array([1705.0, 12.0, 0.3])
            valid_mask = np.all(F_arr < ref, axis=1)
            
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
            else:
                valid_idx = list(range(len(F_arr)))
            
            # Shuffle valid indices for diversity
            rng.shuffle(valid_idx)
            
            for w in selected_weights:
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Pick a starting point (cycle through valid indices)
                start_idx = valid_idx[(int(w.sum() * 100)) % len(valid_idx)]
                x0 = X[start_idx]
                
                def objective(x, _w=w):
                    nonlocal used
                    if used >= self.budget:
                        return 0.0
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    X.append(x)
                    F.append(f)
                    used += 1
                    return float(np.dot(_w, f))
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B',
                                 bounds=list(zip(lo, hi)),
                                 options={'maxiter': 50, 'maxfun': 50, 'ftol': 1e-8})
                except Exception:
                    pass

        # Phase 3: Additional random sampling with local polish if budget remains
        if used < self.budget and time.time() < deadline:
            remaining = self.budget - used
            n_extra = min(remaining, max(5, remaining // 4))
            
            for i in range(n_extra):
                if time.time() > deadline or used >= self.budget:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                f = eval_point(x)
                if f is None:
                    break

        return X
# EVOLVE-BLOCK-END
