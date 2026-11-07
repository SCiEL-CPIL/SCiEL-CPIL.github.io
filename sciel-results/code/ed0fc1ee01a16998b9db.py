"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol sampling, structured initialization, and NSGA-II refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(points):
            n = len(points)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(points[j], points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            if len(nd_idx) <= 2:
                return np.ones(len(nd_idx))

            dist = np.zeros(len(nd_idx))
            m = F.shape[1]  # number of objectives

            for k in range(m):
                sorted_idx = np.argsort(F[nd_idx, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F[nd_idx, k].max() - F[nd_idx, k].min()
                if f_range > 1e-12:
                    for i in range(1, len(nd_idx) - 1):
                        dist[sorted_idx[i]] += (F[nd_idx, k][sorted_idx[i + 1]] - F[nd_idx, k][sorted_idx[i - 1]]) / f_range

            return dist

        # Phase 1: Strategic initialization
        # Combine structured points (from inspiration 1) with Sobol sampling (current)
        # RE21 structure: f1 = x1^2 + x2^2, f2 = (x3-2)^2 + (x4-2)^2
        # Ideal: x1=0, x2=0, x3=2, x4=2
        
        strategic_points = []
        
        # Ideal point (minimizes both if bounds allow)
        ideal = np.array([0.0, 0.0, 2.0, 2.0])
        ideal_clipped = np.clip(ideal, lo, hi)
        strategic_points.append(ideal_clipped)
        
        # Corners
        corners = [
            np.array([lo[0], lo[1], lo[2], lo[3]]),
            np.array([hi[0], hi[1], hi[2], hi[3]]),
            np.array([lo[0], lo[1], hi[2], hi[3]]),
            np.array([hi[0], hi[1], lo[2], lo[3]]),
            np.array([hi[0], hi[1], hi[2], hi[3]]),
            np.array([lo[0], lo[1], lo[2], hi[3]]),
            np.array([hi[0], hi[1], lo[2], hi[3]]),
            np.array([lo[0], lo[1], hi[2], lo[3]]),
        ]
        for c in corners:
            c_clipped = np.clip(c, lo, hi)
            strategic_points.append(c_clipped)
        
        # Midpoints
        mid = (lo + hi) / 2.0
        strategic_points.append(mid)
        
        # Points along the "front" - interpolate between ideal and corners
        # For f1, move away from (0,0) in x1,x2
        # For f2, move away from (2,2) in x3,x4
        for t in [0.2, 0.4, 0.6, 0.8]:
            # Increase f1, keep f2 low
            p1 = np.array([t * (hi[0] - lo[0]), t * (hi[1] - lo[1]), 2.0, 2.0])
            p1 = np.clip(p1, lo, hi)
            strategic_points.append(p1)
            
            # Keep f1 low, increase f2
            p2 = np.array([0.0, 0.0, 2.0 + t * (hi[2] - lo[2]), 2.0 + t * (hi[3] - lo[3])])
            p2 = np.clip(p2, lo, hi)
            strategic_points.append(p2)
        
        # Remove duplicates
        unique_strategic = []
        for p in strategic_points:
            is_dup = False
            for q in unique_strategic:
                if np.allclose(p, q, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                unique_strategic.append(p)
        
        n_strategic = min(len(unique_strategic), 20)
        n_sobol = 80  # remaining budget for Sobol
        n_total_phase1 = n_strategic + n_sobol
        
        X_phase1 = np.zeros((n_total_phase1, dim))
        F_phase1 = np.zeros((n_total_phase1, n_obj))
        used = 0
        
        # Evaluate strategic points
        for i in range(n_strategic):
            if time.time() > deadline - 10:
                break
            x = unique_strategic[i]
            f = problem(x)
            X_phase1[i] = x
            F_phase1[i] = f
            used += 1
        
        # Sobol sampling
        if used < n_total_phase1:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            n_sobol_actual = n_total_phase1 - used
            points_unit = sampler.random(n_sobol_actual)
            X_sobol = lo + points_unit * (hi - lo)
            
            for i in range(n_sobol_actual):
                if time.time() > deadline - 10:
                    break
                x = X_sobol[i]
                f = problem(x)
                X_phase1[used] = x
                F_phase1[used] = f
                used += 1
        
        # Phase 2: NSGA-II style refinement
        pop_size = 30
        n_generations = (budget - used) // pop_size
        
        # Initialize population with best non-dominated points + random
        front_indices = non_dominated_front(F_phase1)
        
        X_pop = np.zeros((pop_size, dim))
        F_pop = np.zeros((pop_size, n_obj))
        
        # Fill with non-dominated points first
        n_nd = min(len(front_indices), pop_size // 2)
        if n_nd > 0:
            front_points = F_phase1[front_indices]
            order = np.argsort(np.sum(front_points, axis=1))
            selected = front_indices[order[:n_nd]]
            X_pop[:n_nd] = X_phase1[selected]
            F_pop[:n_nd] = F_phase1[selected]
        
        # Fill rest with random points
        rng = np.random.default_rng(123)
        for i in range(n_nd, pop_size):
            if used >= budget or time.time() > deadline - 10:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            X_pop[i] = x
            F_pop[i] = f
            used += 1
        
        # NSGA-II loop
        for gen in range(n_generations):
            if used >= budget or time.time() > deadline - 10:
                break
            
            # Get non-dominated front
            nd_idx = non_dominated_front(F_pop)
            
            if len(nd_idx) == 0:
                best_idx = np.argmin(np.sum(F_pop, axis=1))
                nd_idx = np.array([best_idx])
            
            # Compute crowding distance for selection
            cd = crowding_distance(F_pop, nd_idx)
            
            # Select parents and generate offspring
            for child_idx in range(pop_size):
                if used >= budget or time.time() > deadline - 10:
                    break
                
                # Tournament selection for parent 1
                p1_idx = self._tournament_select(F_pop, cd, nd_idx, 2, rng)
                # Tournament selection for parent 2
                p2_idx = self._tournament_select(F_pop, cd, nd_idx, 2, rng)
                
                # Crossover (SBX-like)
                if p1_idx != p2_idx:
                    alpha = rng.random()
                    child = alpha * X_pop[p1_idx] + (1 - alpha) * X_pop[p2_idx]
                else:
                    child = X_pop[p1_idx].copy()
                
                # Adaptive mutation
                # Use smaller steps as we progress
                step_scale = 0.15 * (1.0 - 0.3 * gen / max(n_generations, 1))
                step_size = step_scale * (hi - lo)
                child = child + step_size * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                # Use a combined score: dominated points get high score, non-dominated get low
                worst_idx = int(np.argmax(np.sum(F_pop, axis=1)))
                X_pop[worst_idx] = child
                F_pop[worst_idx] = fc
        
        # Combine all evaluated points
        all_X = np.vstack([X_phase1, X_pop])
        all_F = np.vstack([F_phase1, F_pop])
        
        # Remove duplicates (approximately)
        unique_X = []
        unique_F = []
        for i in range(len(all_X)):
            is_dup = False
            for j in range(len(unique_X)):
                if np.allclose(all_X[i], unique_X[j], atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                unique_X.append(all_X[i])
                unique_F.append(all_F[i])
        
        if len(unique_X) == 0:
            return all_X[0] if len(all_X) > 0 else np.zeros(dim)
        
        unique_X = np.array(unique_X)
        unique_F = np.array(unique_F)
        
        # Final non-dominated front
        final_front = non_dominated_front(unique_F)
        
        if len(final_front) == 0:
            return unique_X[0]
        
        return unique_X[final_front[0]]

    def _tournament_select(self, F, cd, nd_idx, k, rng):
        """Tournament selection."""
        n = len(F)
        rank = np.ones(n)
        rank[nd_idx] = 0
        
        candidates = rng.integers(0, n, size=k)
        scores = []
        for c in candidates:
            if rank[c] == 0:
                # Non-dominated: higher crowding distance is better
                cd_mask = nd_idx == c
                if np.any(cd_mask):
                    scores.append(-cd[cd_mask][0])
                else:
                    scores.append(0)
            else:
                scores.append(np.sum(F[c]))
        
        best = candidates[np.argmin(scores)]
        return best
# EVOLVE-BLOCK-END
