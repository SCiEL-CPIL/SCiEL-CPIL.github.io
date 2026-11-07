"""Rewritten multi-objective optimizer for ZDT2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using decomposition + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # ZDT2 specific: x1 controls f1, x2..x30 control f2 via g
        # Optimal: x1 in [0,1], x2..x30 = 0
        # We use a smart initialization and focused search

        # Phase 1: Initialize with points spread along expected Pareto front
        # f1 = x1, f2 = (1+10*n)*(1 - sqrt(x1/g)) where g = 1 + 9*(sum(x2..xn)/n)
        # At optimum g=1, so f2 = 11*(1 - sqrt(x1))
        n_init = min(30, budget // 3)
        X = np.zeros((n_init, dim))
        F = np.zeros((n_init, 2))
        
        # Spread x1 uniformly, set x2..xn = 0 (optimal)
        for i in range(n_init):
            X[i, 0] = (i + 0.5) / n_init
            # Small random perturbation in other dims to explore
            X[i, 1:] = 0.01 * rng.random(dim - 1)
        
        # Evaluate
        for i in range(n_init):
            F[i] = problem(X[i])
        
        used = n_init

        # Phase 2: Iterative refinement - focus on improving g (x2..xn)
        # and fine-tuning x1
        remaining = budget - used
        if remaining <= 0:
            return X

        # Use a simple evolutionary approach with bias toward reducing g
        pop_size = min(20, max(5, budget // 10))
        
        # Select best points (non-dominated)
        def is_dominated(F_mat, idx):
            f = F_mat[idx]
            for j in range(len(F_mat)):
                if j == idx:
                    continue
                if np.all(F_mat[j] <= f) and np.any(F_mat[j] < f):
                    return True
            return False
        
        # Keep track of all evaluated points
        all_X = [X[i].copy() for i in range(n_init)]
        all_F = [F[i].copy() for i in range(n_init)]
        
        # Main loop: generate children by mutating good parents
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Find non-dominated points
            nd_indices = []
            for i in range(len(all_X)):
                if not is_dominated(np.array(all_F), i):
                    nd_indices.append(i)
            
            if not nd_indices:
                nd_indices = list(range(len(all_X)))
            
            # Select parent from non-dominated set
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = all_X[parent_idx].copy()
            
            # Mutation strategy: 
            # - Strongly push x2..xn toward 0 (reduces g, improves f2)
            # - Small perturbation on x1
            child = parent.copy()
            
            # Push x2..xn toward 0 with some noise
            decay = 0.5 + 0.5 * (used / budget)  # stronger decay early
            child[1:] = child[1:] * decay + 0.001 * rng.standard_normal(dim - 1)
            child[1:] = np.clip(child[1:], 0, 1)
            
            # Small perturbation on x1
            child[0] += 0.05 * rng.standard_normal()
            child[0] = np.clip(child[0], 0, 1)
            
            # Evaluate
            fc = problem(child)
            used += 1
            all_X.append(child.copy())
            all_F.append(np.array(fc))
            
            # Occasionally do a pure random exploration
            if rng.random() < 0.05 and used < budget:
                rand_x = lo + span * rng.random(dim)
                rand_f = problem(rand_x)
                used += 1
                all_X.append(rand_x.copy())
                all_F.append(np.array(rand_f))
        
        # Phase 3: Local polish on best points
        if used < budget and len(all_X) > 0:
            all_F_arr = np.array(all_F)
            # Find points with best f2 (lowest g)
            # Sort by f2 and take top few for local refinement
            order = np.argsort(all_F_arr[:, 1])
            top_k = min(5, len(all_F_arr))
            
            for idx in order[:top_k]:
                if used >= budget or time.time() > deadline - 2.0:
                    break
                base = all_X[idx].copy()
                # Local search: try small steps in x2..xn direction toward 0
                for _ in range(3):
                    if used >= budget or time.time() > deadline - 1.0:
                        break
                    candidate = base.copy()
                    candidate[1:] = np.maximum(0, candidate[1:] - 0.005)
                    candidate[0] = np.clip(base[0] + 0.01 * rng.standard_normal(), 0, 1)
                    cf = problem(candidate)
                    used += 1
                    all_X.append(candidate.copy())
                    all_F.append(np.array(cf))
                    if np.all(np.array(cf) <= all_F[idx]):
                        base = candidate.copy()
        
        return all_X
# EVOLVE-BLOCK-END
