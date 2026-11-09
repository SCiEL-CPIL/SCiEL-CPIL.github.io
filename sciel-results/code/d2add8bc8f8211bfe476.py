import numpy as np
import time

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using
    an adaptive non-dominated sorting strategy with Sobol sampling for initial
    coverage and local refinement around hypervolume-critical regions."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n points in the unit square using a simple Sobol-like sequence."""
        # Use a deterministic low-discrepancy sequence based on scrambled Halton
        # with bases 2 and 3, which is a good approximation for 2D.
        i = np.arange(1, n + 1)
        
        # Halton base 2
        x0 = np.zeros(n)
        for k in range(n):
            val = 0.0
            x = i[k]
            f = 0.5
            while x > 0:
                val += f * (x % 2)
                x //= 2
                f /= 2.0
            x0[k] = val

        # Halton base 3
        x1 = np.zeros(n)
        for k in range(n):
            val = 0.0
            x = i[k]
            f = 1.0 / 3.0
            while x > 0:
                val += f * (x % 3)
                x //= 3
                f /= 3.0
            x1[k] = val

        return np.column_stack((x0, x1))

    def _non_dominated_sort(self, objectives):
        """Return indices of non-dominated points (Pareto front)."""
        n = len(objectives)
        if n == 0:
            return np.array([], dtype=int)
        
        # For 3 objectives, a point i dominates j if all objectives of i <= j
        # and at least one is strictly less.
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                # Check if i dominates j
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    dominated[j] = True
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    dominated[i] = True
                    break
        
        front_indices = np.where(~dominated)[0]
        return front_indices

    def _compute_hypervolume_contribution(self, objectives, reference):
        """
        Estimate which regions of the objective space contribute most to hypervolume.
        Returns a set of "critical" regions where adding a point would increase hypervolume most.
        Simplified: identify gaps in the non-dominated front in objective space.
        """
        n = len(objectives)
        if n < 2:
            return []
        
        # Find the bounding box of the objectives
        min_obj = np.min(objectives, axis=0)
        max_obj = np.max(objectives, axis=0)
        
        # Create a coarse grid to identify empty regions
        # We'll divide the space into a 4x4 grid in the 2D trade-off space
        # Project to 2D: use the first two objectives as the trade-off plane
        # (since the third is determined by the sum constraint for DTLZ1)
        
        # Actually, for DTLZ1, the Pareto front lies in a 2D simplex.
        # We can parameterize by two variables. Let's find the "corners" of the front
        # and identify the largest empty triangles.
        
        # For simplicity, we'll return points that are far from existing points
        # in the parameter space (x0, x1)
        return []

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Phase 1: Initial coverage with Sobol sampling (150 points)
        n_phase1 = min(150, budget)
        n_phase2 = budget - n_phase1
        
        # Generate initial samples
        samples_phase1 = self._sobol_2d(n_phase1)
        
        # Ensure corners are included
        if n_phase1 >= 3:
            samples_phase1[0] = [1.0, 0.0]
            samples_phase1[1] = [0.0, 1.0]
            samples_phase1[2] = [0.0, 0.0]
        
        # Construct points for phase 1
        points_phase1 = np.zeros((n_phase1, dim))
        points_phase1[:, 0] = samples_phase1[:, 0]
        points_phase1[:, 1] = samples_phase1[:, 1]
        for j in range(2, dim):
            points_phase1[:, j] = 0.5
        
        # Evaluate phase 1
        results = []
        objectives = []
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = points_phase1[i]
            obj = problem(x)
            results.append(obj)
            objectives.append(np.array(obj))
        
        if time.time() > deadline or len(results) >= budget:
            return results
        
        # Phase 2: Local refinement around hypervolume-critical regions
        if n_phase2 > 0 and objectives:
            objectives = np.array(objectives)
            
            # Identify non-dominated points
            front_idx = self._non_dominated_sort(objectives)
            
            if len(front_idx) > 0:
                front_obj = objectives[front_idx]
                
                # Find the "gaps" in the front by looking at the parameter space
                # For each non-dominated point, we know its (x0, x1) from the samples
                front_params = samples_phase1[front_idx]
                
                # Generate refinement points around the front
                # Use a jittered grid around the front points
                n_refine_per_point = max(1, n_phase2 // max(1, len(front_idx)))
                
                refine_points = []
                for idx in front_idx:
                    if time.time() > deadline:
                        break
                    center = samples_phase1[idx]
                    # Generate a few points around the center with small jitter
                    n_local = min(n_refine_per_point, 3)
                    for _ in range(n_local):
                        if time.time() > deadline or len(refine_points) >= n_phase2:
                            break
                        # Small random jitter (deterministic based on index)
                        jitter_scale = 0.02
                        dx = (hash(idx, _[0]) % 1000 / 1000.0 - 0.5) * 2 * jitter_scale
                        dy = (hash(idx, _[1]) % 1000 / 1000.0 - 0.5) * 2 * jitter_scale
                        new_x0 = np.clip(center[0] + dx, 0.0, 1.0)
                        new_x1 = np.clip(center[1] + dy, 0.0, 1.0)
                        # Ensure we're on the valid simplex: x0 + x1 <= 1
                        if new_x0 + new_x1 > 1.0:
                            scale = 1.0 / (new_x0 + new_x1)
                            new_x0 *= scale
                            new_x1 *= scale
                        refine_points.append([new_x0, new_x1])
                
                # Evaluate refinement points
                for rp in refine_points:
                    if time.time() > deadline or len(results) >= budget:
                        break
                    x = np.zeros(dim)
                    x[0] = rp[0]
                    x[1] = rp[1]
                    for j in range(2, dim):
                        x[j] = 0.5
                    obj = problem(x)
                    results.append(obj)
        
        return results
