"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using HV-guided search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _nondominated_sort(self, F):
        """Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        # S[i] = set of points dominated by i
        # n_dom[i] = number of points dominating i
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        # If some points are not in any front (should not happen for valid data), add them
        if len(np.concatenate(fronts)) < n:
            remaining = [i for i in range(n) if i not in np.concatenate(fronts)]
            if remaining:
                fronts.append(remaining)
                
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for k in range(F.shape[1]):
            sorted_idx = np.argsort(F_front[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            min_val = F_front[sorted_idx[0], k]
            max_val = F_front[sorted_idx[-1], k]
            if max_val > min_val:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_front[sorted_idx[i + 1], k] - F_front[sorted_idx[i - 1], k]) / (max_val - min_val)
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial random sampling
        n_init = min(50, self.budget // 4)
        X_list = []
        F_list = []

        for i in range(n_init):
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            X_list.append(x)
            F_list.append(f)

        used = n_init
        step_size = 0.2

        # Track best hypervolume points for polish
        best_points = []

        while used < self.budget:
            if time.time() > deadline - 60.0:  # Reserve 60s for polish
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute approximate HV contribution for each point
            valid = np.all(F_arr < ref, axis=1)
            if np.sum(valid) == 0:
                parent_idx = rng.integers(n)
            else:
                # For valid points, compute contribution
                Fv = F_arr[valid]
                nv = len(Fv)
                contrib = np.zeros(n)
                if nv == 1:
                    contrib[valid] = np.prod(np.maximum(0, ref - Fv[0]))
                else:
                    # Approximate: box volume * uniqueness weight
                    for i in range(nv):
                        box_vol = np.prod(np.maximum(0, ref - Fv[i]))
                        # Uniqueness: distance to nearest neighbor
                        dists = np.linalg.norm(Fv - Fv[i], axis=1)
                        dists[i] = np.inf
                        min_dist = np.min(dists) if nv > 1 else 1.0
                        scale = np.max(np.linalg.norm(Fv - Fv[0], axis=1)) + 1e-10
                        weight = 1.0 / (1.0 + 0.5 * min_dist / scale)
                        contrib[valid][i] = box_vol * weight

                if np.max(contrib) > 0:
                    top_k = min(5, int(np.sum(contrib > 0)))
                    if top_k > 0:
                        valid_idx = np.where(contrib > 0)[0]
                        top_idx = valid_idx[np.argsort(contrib[valid_idx])][-top_k:]
                        w = contrib[top_idx]
                        w = w / w.sum()
                        parent_idx = rng.choice(top_idx, p=w)
                    else:
                        parent_idx = rng.integers(n)
                else:
                    parent_idx = rng.integers(n)

            parent = X_list[parent_idx]
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.array(problem(child))
            X_list.append(child)
            F_list.append(fc)
            used += 1

            # Adapt step size
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                if not dominated:
                    step_size = max(0.05, step_size * 0.92)
                else:
                    step_size = min(1.0, step_size * 1.08)
            else:
                step_size = min(1.0, step_size * 1.15)

        # Polish phase: use remaining budget to refine best points
        F_arr = np.array(F_list)
        valid = np.all(F_arr < ref, axis=1)
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) > 0 and used < self.budget:
            # Sort by HV contribution (approximate)
            Fv = F_arr[valid_indices]
            nv = len(Fv)
            contrib = np.zeros(nv)
            if nv == 1:
                contrib[0] = np.prod(np.maximum(0, ref - Fv[0]))
            else:
                for i in range(nv):
                    box_vol = np.prod(np.maximum(0, ref - Fv[i]))
                    dists = np.linalg.norm(Fv - Fv[i], axis=1)
                    dists[i] = np.inf
                    min_dist = np.min(dists) if nv > 1 else 1.0
                    scale = np.max(np.linalg.norm(Fv - Fv[0], axis=1)) + 1e-10
                    weight = 1.0 / (1.0 + 0.5 * min_dist / scale)
                    contrib[i] = box_vol * weight
            
            # Take top K points for polish
            top_k_polish = min(10, nv, (self.budget - used) // 3)
            if top_k_polish > 0:
                top_idx = valid_indices[np.argsort(contrib)][-top_k_polish:]
                
                for idx in top_idx:
                    if time.time() > deadline - 10.0:
                        break
                    if used >= self.budget:
                        break
                    
                    parent = X_list[idx]
                    # Local search with smaller steps
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline - 10.0:
                            break
                        child = parent + 0.05 * span * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                        fc = np.array(problem(child))
                        X_list.append(child)
                        F_list.append(fc)
                        used += 1
                        if np.all(fc < ref):
                            parent = child  # Keep improving point as parent

        return X_list
# EVOLVE-BLOCK-END
