"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Hybrid LHS + Differential Evolution for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Phase 1: Latin Hypercube Sampling for broad, uniform coverage
        # LHS is often superior to Sobol for initial front discovery in high dims
        n_lhs = min(100, self.budget // 2)
        sampler = qmc.LatinHypercube(d=self.dim, seed=0)
        lhs_samples = sampler.random(n=n_lhs)
        X_lhs = lo + (hi - lo) * lhs_samples
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs

        # Build initial global archive
        archive_X, archive_F = self._build_archive(X_lhs, F_lhs)
        
        # Select an active population for DE refinement
        # We want a diverse set of good points to guide DE
        active_size = 20
        if len(archive_X) > active_size:
            active_idx = self._select_diverse(archive_X, archive_F, k=active_size)
            active_X = archive_X[active_idx].copy()
            active_F = archive_F[active_idx].copy()
        else:
            active_X = archive_X.copy()
            active_F = archive_F.copy()

        # Phase 2: Differential Evolution Refinement
        # Use remaining budget to refine the front using DE/rand/1/bin
        n_refine = self.budget - used
        rng = np.random.default_rng(42)
        F_factor = 0.8  # Differential weight
        CR = 0.5        # Crossover rate

        for _ in range(n_refine):
            if used >= self.budget:
                break
            if time.time() - start_time > deadline:
                break

            # Ensure we have at least 3 parents
            if len(active_X) < 3:
                # If active population is too small, just do random local search
                center = active_X[rng.integers(len(active_X))]
                noise = rng.standard_normal(self.dim)
                step = 0.1 * (hi - lo)
                child = center + step * noise
                child = np.clip(child, lo, hi)
            else:
                # DE/rand/1/bin
                # Pick 3 distinct random indices
                idx_a, idx_b, idx_c = rng.choice(len(active_X), size=3, replace=False)
                x_a = active_X[idx_a]
                x_b = active_X[idx_b]
                x_c = active_X[idx_c]
                
                # Vector difference
                v = x_a + F_factor * (x_b - x_c)
                
                # Crossover
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        v[j] = v[j]
                    else:
                        v[j] = x_a[j]
                        
                child = np.clip(v, lo, hi)

            fc = problem(child)
            used += 1

            # Update global archive
            archive_X, archive_F = self._update_archive(archive_X, archive_F, child, fc)

            # Update active population
            # Strategy: If the new point is better than the worst point in the active population, replace it.
            # We define "better" by hypervolume contribution to the current archive.
            if len(active_X) > 0:
                # Calculate HV contribution for all active points and the new point
                # To be efficient, we only check if the new point is non-dominated in the archive.
                # If it is, it's likely a good candidate for the active population.
                
                # Simple check: Is the new point dominated by any active point?
                # If not, it's a strong candidate.
                dominated_by_active = np.any(
                    np.all(active_F <= fc, axis=1) & np.any(active_F < fc, axis=1)
                )
                
                if not dominated_by_active:
                    # Find the worst point in active population based on HV contribution
                    # Approximate by removing each point and checking HV change? Too slow.
                    # Instead, use crowding distance or just replace the worst by "age" or random.
                    # Let's replace the point with the lowest crowding distance in the active set.
                    # Or simpler: if active size is fixed, replace the one with the worst HV contribution.
                    
                    # For speed, let's just replace a random point if the new one is non-dominated in the global archive.
                    # This maintains diversity.
                    if rng.random() < 0.5:
                        replace_idx = rng.integers(len(active_X))
                        active_X[replace_idx] = child
                        active_F[replace_idx] = fc

        return archive_X if len(archive_X) > 0 else X_lhs[:1]

    def _build_archive(self, X, F):
        """Build initial archive from non-dominated points."""
        n = len(X)
        if n == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)

        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            dom_mask = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dom_mask[i] = False
            if np.any(dom_mask):
                dominated[i] = True

        mask = ~dominated
        return X[mask], F[mask]

    def _update_archive(self, X, F, new_x, new_f):
        """Update archive with a new point."""
        if len(F) == 0:
            return new_x.reshape(1, -1).copy(), new_f.reshape(1, -1).copy()
        
        # Check if new point is dominated
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
