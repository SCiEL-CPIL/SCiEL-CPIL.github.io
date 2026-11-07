"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol Sampling + Local Polish for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Phase 1: Sobol Sampling for broad, uniform coverage
        # Sobol sequences provide better low-discrepancy coverage than LHS in 12D
        n_sobol = min(160, self.budget)
        
        # Use a fixed seed for reproducibility in the sampling sequence itself,
        # but the problem evaluator sets np.random.seed(seed) globally.
        # qmc.Sobol seed is independent of np.random unless we tie them.
        # We'll use a fixed seed for the Sobol sequence to ensure consistent space filling.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=1234)
        sobol_samples = sampler.random(n=n_sobol)
        X_sobol = lo + (hi - lo) * sobol_samples
        
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        # Build initial global archive
        archive_X, archive_F = self._build_archive(X_sobol, F_sobol)
        
        # Phase 2: Local Polish / Refinement
        # Use remaining budget to refine the front by perturbing existing good points
        n_refine = self.budget - used
        
        if n_refine > 0 and len(archive_X) > 0:
            # Select a diverse subset of the archive to serve as parents for local search
            # This ensures we explore the entire front, not just one region
            k_parents = min(10, len(archive_X))
            parent_idx = self._select_diverse(archive_X, archive_F, k=k_parents)
            parents_X = archive_X[parent_idx].copy()
            
            # Adaptive step size for local search
            # Start with a relatively small step, potentially increasing if stagnation occurs?
            # For simplicity and robustness, use a fixed small fraction of the range.
            # WFG4 variables are typically in [0, 1] or similar.
            # Let's use 0.05 * (hi - lo) as the standard deviation for Gaussian noise.
            step_size = 0.05 * (hi - lo)
            
            rng = np.random.default_rng(42)
            
            for _ in range(n_refine):
                if used >= self.budget:
                    break
                if time.time() - start_time > deadline:
                    break

                # Pick a random parent from the diverse set
                p_idx = rng.integers(len(parents_X))
                parent = parents_X[p_idx]
                
                # Generate child via Gaussian perturbation
                noise = rng.standard_normal(self.dim)
                child = parent + step_size * noise
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Update global archive
                archive_X, archive_F = self._update_archive(archive_X, archive_F, child, fc)
                
                # Optional: Update the parent set if the new point is significantly better?
                # For simplicity, we keep the initial diverse parent set.
                # This encourages exploration of the initial front regions.
                # If we wanted to be more adaptive, we could replace the worst parent.

        return archive_X if len(archive_X) > 0 else X_sobol[:1]

    def _build_archive(self, X, F):
        """Build initial archive from non-dominated points."""
        n = len(X)
        if n == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)

        dominated = np.zeros(n, dtype=bool)
        # Vectorized dominance check is tricky for all pairs, but n is small (160)
        for i in range(n):
            # Check if any other point j dominates i
            # j dominates i if F[j] <= F[i] and F[j] < F[i] (strictly better in at least one)
            dom_mask = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dom_mask[i] = False  # Exclude self
            if np.any(dom_mask):
                dominated[i] = True

        mask = ~dominated
        return X[mask], F[mask]

    def _update_archive(self, X, F, new_x, new_f):
        """Update archive with a new point."""
        if len(F) == 0:
            return new_x.reshape(1, -1).copy(), new_f.reshape(1, -1).copy()
        
        # Check if new point is dominated by any existing point
        new_dominated = np.any(
            np.all(F <= new_f, axis=1) & np.any(F < new_f, axis=1)
        )
        
        if new_dominated:
            return X, F
        
        # Remove points dominated by the new point
        dominated_mask = np.array([
            np.all(new_f <= f, axis=0) and np.any(new_f < f)
            for f in F
        ])
        
        if np.any(dominated_mask):
            X = np.delete(X, np.where(dominated_mask)[0], axis=0)
            F = np.delete(F, np.where(dominated_mask)[0], axis=0)
            
        # Add new point
        X = np.vstack([X, new_x.reshape(1, -1)])
        F = np.vstack([F, new_f.reshape(1, -1)])
        
        # Reduce archive if it gets too large to maintain performance
        if len(X) > 200:
            X, F = self._reduce_archive(X, F, max_size=100)
            
        return X, F

    def _select_diverse(self, X, F, k=10):
        """Select k diverse points from archive using crowding distance."""
        n = len(F)
        if n <= k:
            return np.arange(n)

        crowding = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                diff = F[order[i+1], m] - F[order[i-1], m]
                if diff > 1e-10:
                    crowding[order[i]] += diff

        keep = np.argsort(crowding)[-k:]
        return np.sort(keep)

    def _reduce_archive(self, X, F, max_size=100):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F
        keep_idx = self._select_diverse(X, F, k=max_size)
        return X[keep_idx], F[keep_idx]


# EVOLVE-BLOCK-END
