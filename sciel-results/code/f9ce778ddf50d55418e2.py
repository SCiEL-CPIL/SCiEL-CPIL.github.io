"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol quasi-random sweep + local Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n_points, dim):
        """Generate a Sobol quasi-random sequence in [0, 1]^dim."""
        sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
        samples = sampler.random(n=n_points)
        return samples

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        # domination[i] = list of indices dominated by i
        domination = [[] for _ in range(n)]
        # dominance_count[i] = number of points that dominate i
        dominance_count = [0] * n
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination[i].append(j)
                    dominance_count[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination[j].append(i)
                    dominance_count[i] += 1
        
        # First front: points with no dominators
        for i in range(n):
            if dominance_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in domination[i]:
                    dominance_count[j] -= 1
                    if dominance_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        m = self.n_obj
        
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = sorted(range(n), key=lambda i: F[indices[i]][k])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Range of objective k
            obj_range = F[indices[sorted_indices[-1]]][k] - F[indices[sorted_indices[0]]][k]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    F[indices[sorted_indices[i + 1]]][k] - F[indices[sorted_indices[i - 1]]][k]
                ) / obj_range
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Sobol quasi-random sweep for uniform coverage
        # Use a larger portion of budget for initial exploration
        sweep_size = int(self.budget * 0.6)
        sweep_size = min(sweep_size, self.budget)

        if sweep_size >= 2:
            sobol_samples = self._sobol_sequence(sweep_size, self.dim)
        else:
            sobol_samples = np.random.random((sweep_size, self.dim))

        all_X = sobol_samples.copy()
        all_F = np.array([problem(x) for x in all_X])
        used = sweep_size

        # Phase 2: Local Gaussian perturbations around non-dominated points
        remaining = self.budget - used

        if remaining > 0:
            # Find non-dominated points from the sweep using fast non-dominated sort
            fronts = self._fast_non_dominated_sort(all_F)
            
            # Get first front (non-dominated points)
            if fronts and fronts[0]:
                non_dom_indices = fronts[0]
                
                # Adaptive sigma: start moderate, shrink over time
                num_rounds = min(remaining, 80)
                points_per_round = max(1, remaining // num_rounds)

                for round_idx in range(num_rounds):
                    if time.time() > deadline:
                        break
                    if used >= self.budget:
                        break

                    # Progress-based sigma: larger early, smaller late
                    progress = round_idx / max(1, num_rounds - 1)
                    sigma = 0.12 * (1.0 - 0.8 * progress)

                    # Select a subset of non-dominated points to perturb
                    # Focus on points with higher crowding distance (more spread out)
                    n_perturb = min(len(non_dom_indices), points_per_round)
                    if n_perturb > 0:
                        if len(non_dom_indices) > n_perturb:
                            # Compute crowding distance to prioritize diverse points
                            cd = self._crowding_distance(all_F, non_dom_indices)
                            # Select points with highest crowding distance
                            sorted_by_cd = sorted(range(len(non_dom_indices)), key=lambda i: cd[i], reverse=True)
                            selected = [non_dom_indices[i] for i in sorted_by_cd[:n_perturb]]
                        else:
                            selected = list(non_dom_indices)

                        new_points = []
                        for idx in selected:
                            base = all_X[idx]
                            perturbed = base + sigma * np.random.standard_normal(self.dim)
                            perturbed = np.clip(perturbed, lo, hi)
                            new_points.append(perturbed)

                        if new_points:
                            new_points = np.array(new_points)
                            new_F = np.array([problem(x) for x in new_points])
                            all_X = np.vstack([all_X, new_points])
                            all_F = np.vstack([all_F, new_F])
                            used += len(new_points)

                            # Update non-dominated set using fast non-dominated sort
                            fronts = self._fast_non_dominated_sort(all_F)
                            if fronts and fronts[0]:
                                non_dom_indices = fronts[0]

        return all_X
# EVOLVE-BLOCK-END
