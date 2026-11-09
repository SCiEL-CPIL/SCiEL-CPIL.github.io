"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using Sobol initialization and HV-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Helper: Check if point a dominates point b
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        # Helper: Compute 2D Hypervolume for a set of points
        def hv_2d(points, ref_2d):
            if len(points) == 0:
                return 0.0
            # Sort by first coordinate
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            hv = 0.0
            n = len(pts)
            if n == 0:
                return 0.0
            
            # Pre-compute min_x2_from_right
            min_x2_from_right = np.empty(n)
            min_x2_from_right[-1] = pts[-1, 1]
            for i in range(n - 2, -1, -1):
                min_x2_from_right[i] = min(pts[i, 1], min_x2_from_right[i + 1])
                
            for i in range(n):
                x1 = pts[i, 0]
                if i + 1 < n:
                    x1_next = pts[i + 1, 0]
                else:
                    x1_next = ref_2d[0]
                width = x1_next - x1
                if width <= 0:
                    continue
                x2_eff = min_x2_from_right[i]
                if x2_eff >= ref_2d[1]:
                    continue
                height = ref_2d[1] - x2_eff
                hv += width * height
            return hv

        # Helper: Compute 3D Hypervolume
        def hv_3d(points, ref_3d):
            if len(points) == 0:
                return 0.0
            # Filter valid points
            valid_mask = np.all(points < ref_3d, axis=1)
            pts = points[valid_mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by x1
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            n = len(pts)
            
            hv = 0.0
            for i in range(n):
                x1 = pts[i, 0]
                if i + 1 < n:
                    x1_next = pts[i + 1, 0]
                else:
                    x1_next = ref_3d[0]
                width = x1_next - x1
                if width <= 0:
                    continue
                
                # Get points from i to end
                suffix = pts[i:, 1:3]
                area = hv_2d(suffix, ref_3d[1:3])
                hv += width * area
                
            return hv

        # Helper: Find non-dominated set indices
        def get_non_dominated_indices(F):
            n = len(F)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(n):
                    if i == j or is_dominated[j]:
                        continue
                    if dominates(F[j], F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        # Phase 1: Sobol Initialization (128 points)
        # Use a deterministic Sobol sequence for maximal space-filling diversity
        sobol_n = min(128, budget)
        sampler = qmc.Sobol(d=dim, scramble=False)
        # Generate 2^k points where 2^k >= sobol_n
        k = int(np.ceil(np.log2(sobol_n)))
        sobol_points = sampler.random(2**k)[:sobol_n]
        
        # Map to bounds
        X_sobol = lo + (hi - lo) * sobol_points
        F_sobol = np.array([problem(x) for x in X_sobol])
        evals_used = sobol_n
        
        X = X_sobol.copy()
        F = F_sobol.copy()
        
        # Initial HV
        current_hv = hv_3d(F, ref)
        
        # Phase 2: Local Search on Top Marginal HV Points (remaining budget)
        # We will focus on the top 10 non-dominated points with highest marginal contribution
        # To save time, we'll approximate marginal contribution by HV difference when removed
        
        remaining_budget = budget - evals_used
        
        # If we have very little budget left, just return
        if remaining_budget <= 0:
            return X
        
        # Prepare local search
        # We'll maintain a list of active points to refine
        # Start with non-dominated points
        nd_indices = get_non_dominated_indices(F)
        
        if len(nd_indices) == 0:
            nd_indices = np.arange(len(F))
            
        # Calculate marginal HV for each non-dominated point
        # This is expensive, so we do it once at the start of Phase 2
        # And periodically update
        # Actually, let's just pick the top 10 based on a simpler heuristic or just random from ND
        # The directive says "top 10 points with the highest marginal hypervolume contribution"
        
        # Let's compute marginal HV for all points to rank them
        # n is at most 128 initially. O(n^2) HV calcs is 128 * 128^2 ~ 2M ops. Fast enough.
        # But hv_3d is O(n^2 log n). So total is O(n^3 log n). 128^3 * 7 ~ 14M. Still fast.
        
        marginal_hvs = np.zeros(len(F))
        for i in range(len(F)):
            # Remove point i
            mask = np.ones(len(F), dtype=bool)
            mask[i] = False
            F_reduced = F[mask]
            hv_reduced = hv_3d(F_reduced, ref)
            marginal_hvs[i] = current_hv - hv_reduced
            
        # Get top 10 indices by marginal HV
        # Only consider non-dominated points for refinement? Or all?
        # Usually dominated points have 0 marginal HV.
        top_k = 10
        if len(marginal_hvs) > top_k:
            # Get indices of top K marginal HVs
            top_indices = np.argsort(marginal_hvs)[::-1][:top_k]
        else:
            top_indices = np.argsort(marginal_hvs)[::-1]
            
        # Local search loop
        # We will cycle through these top points and try to improve them
        # We will re-rank every 10 evaluations to adapt to changes
        
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        rng = np.random.default_rng(42)
        
        evals_since_rerank = 0
        rerank_interval = 10
        
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            # Pick a point from the top list
            # Randomly choose one of the top points
            parent_idx = top_indices[rng.integers(len(top_indices))]
            parent = X[parent_idx].copy()
            
            # Pick a mutation
            # Adaptive step size: start larger, get smaller
            # Simple schedule based on evals_used
            progress = evals_used / budget
            step_idx = min(int(progress * len(step_sizes)), len(step_sizes) - 1)
            step = step_sizes[step_idx]
            
            dim_idx = rng.integers(dim)
            direction = rng.choice([-1, 1])
            
            child = parent.copy()
            child[dim_idx] += direction * step * (hi[dim_idx] - lo[dim_idx])
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            evals_used += 1
            
            # Check if this improves HV
            # Temporarily replace
            F_old = F[parent_idx].copy()
            F[parent_idx] = fc
            new_hv = hv_3d(F, ref)
            
            if new_hv > current_hv + 1e-9:
                # Accept
                X[parent_idx] = child
                current_hv = new_hv
                evals_since_rerank += 1
            else:
                # Reject
                F[parent_idx] = F_old
            
            # Re-rank every rerank_interval evaluations
            if evals_since_rerank >= rerank_interval:
                evals_since_rerank = 0
                # Recalculate marginal HVs
                # Only for non-dominated points to save time?
                # Let's just do all, it's fast enough for n=200
                marginal_hvs = np.zeros(len(F))
                for i in range(len(F)):
                    mask = np.ones(len(F), dtype=bool)
                    mask[i] = False
                    F_reduced = F[mask]
                    hv_reduced = hv_3d(F_reduced, ref)
                    marginal_hvs[i] = current_hv - hv_reduced
                    
                if len(marginal_hvs) > top_k:
                    top_indices = np.argsort(marginal_hvs)[::-1][:top_k]
                else:
                    top_indices = np.argsort(marginal_hvs)[::-1]
                    
        return X

# EVOLVE-BLOCK-END
