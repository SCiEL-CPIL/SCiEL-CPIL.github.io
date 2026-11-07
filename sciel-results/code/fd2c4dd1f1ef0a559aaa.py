"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 using a combination of:
    1. Random initialization
    2. Augmented Tchebycheff decomposition with adaptive weight vectors
    3. Multi-start L-BFGS-B local optimization
    4. Hypervolume-based archive management
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume calculation
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive: list of (x, f) pairs that are non-dominated
        archive_x = []
        archive_f = []
        used = 0

        def is_dominated(f_new, archive_f_list):
            """Check if f_new is dominated by any point in archive_f_list."""
            for f_arch in archive_f_list:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new):
            """Update the archive with a new point, removing dominated points."""
            nonlocal archive_x, archive_f
            # Check if new point is dominated by existing archive
            for i, f_arch in enumerate(archive_f):
                if dominates(f_arch, f_new):
                    return False
            # Remove points dominated by new point
            keep = []
            for i, f_arch in enumerate(archive_f):
                if not dominates(f_new, f_arch):
                    keep.append(i)
            archive_x = [archive_x[i] for i in keep]
            archive_f = [archive_f[i] for i in keep]
            archive_x.append(x_new.copy())
            archive_f.append(f_new.copy())
            return True

        def hv_contribution(f_point, ref):
            """
            Compute the hypervolume contribution of a single point 
            relative to the reference point in 3D.
            Only points strictly better than ref in all objectives count.
            """
            if np.any(f_point >= ref):
                return 0.0
            # For 3 objectives, the contribution is the volume of the box
            # from f_point to ref, but we need to be careful about overlaps.
            # For simplicity, we use a simple volume calculation for the
            # individual contribution (this is an approximation for ranking).
            return np.prod(ref - f_point)

        # Phase 1: Random initialization
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = np.array(problem(x))
            used += 1
            update_archive(x, f)

        # Phase 2: Generate weight vectors for Tchebycheff decomposition
        # Use a set of weight vectors that cover the objective space
        n_weights = min(15, max(5, (self.budget - used) // 10))
        
        # Generate weight vectors on a simplex
        weight_vectors = []
        # Add vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        
        # Add some intermediate points
        if n_obj == 3:
            # Add points along edges and face centers
            for i in range(n_obj):
                for j in range(n_obj):
                    if i != j:
                        for alpha in [0.25, 0.5, 0.75]:
                            w = np.zeros(n_obj)
                            w[i] = alpha
                            w[j] = 1 - alpha
                            weight_vectors.append(w)
            # Face centers
            for i in range(n_obj):
                for j in range(n_obj):
                    for k in range(n_obj):
                        if len(set([i, j, k])) == 3:
                            w = np.array([1/3, 1/3, 1/3])
                            weight_vectors.append(w)
        
        # Normalize and deduplicate
        weight_vectors = np.array(weight_vectors)
        weight_vectors = weight_vectors / weight_vectors.sum(axis=1, keepdims=True)
        
        # Remove duplicates
        unique_weights = []
        for w in weight_vectors:
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        weight_vectors = np.array(unique_weights)

        # Shuffle weight vectors to diversify search
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(weight_vectors))
        weight_vectors = weight_vectors[perm]

        # Phase 3: Local optimization using Tchebycheff decomposition
        # For each weight vector, run L-BFGS-B from multiple starting points
        n_starts_per_weight = 3
        
        # Create starting points: archive points + random points
        start_points = list(archive_x)
        for _ in range(5):
            start_points.append(lo + (hi - lo) * np.random.random(dim))

        for w_idx, w in enumerate(weight_vectors):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            for sp_idx, x_start in enumerate(start_points):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                # Tchebycheff scalarization
                def tcheb_objective(x):
                    f = np.array(problem(x))
                    return np.max(w * (f - np.min(archive_f, axis=0))) if archive_f else np.max(w * f)

                # Run L-BFGS-B
                try:
                    result = minimize(
                        tcheb_objective,
                        x_start,
                        bounds=list(zip(lo, hi)),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'maxfun': 50}
                    )
                    used += result.nfev

                    x_opt = result.x
                    f_opt = np.array(problem(x_opt))
                    update_archive(x_opt, f_opt)
                except Exception:
                    # If optimization fails, just evaluate the start point
                    if used < self.budget:
                        f_sp = np.array(problem(x_start))
                        used += 1
                        update_archive(x_start, f_sp)

        # Phase 4: Fill remaining budget with more random searches and local optimization
        while used < self.budget:
            if time.time() > deadline:
                break

            # Pick a random starting point near the archive
            if archive_x:
                idx = np.random.randint(len(archive_x))
                x_start = archive_x[idx] + 0.1 * (hi - lo) * np.random.randn(dim)
                x_start = np.clip(x_start, lo, hi)
            else:
                x_start = lo + (hi - lo) * np.random.random(dim)

            # Try a few different weight vectors
            w = np.random.dirichlet(np.ones(n_obj))

            def tcheb_objective(x, w=w):
                f = np.array(problem(x))
                if archive_f:
                    ideal = np.min(archive_f, axis=0)
                    return np.max(w * (f - ideal))
                else:
                    return np.max(w * f)

            try:
                result = minimize(
                    tcheb_objective,
                    x_start,
                    bounds=list(zip(lo, hi)),
                    method='L-BFGS-B',
                    options={'maxiter': 20, 'maxfun': 30}
                )
                used += result.nfev

                x_opt = result.x
                f_opt = np.array(problem(x_opt))
                update_archive(x_opt, f_opt)
            except Exception:
                if used < self.budget:
                    f_sp = np.array(problem(x_start))
                    used += 1
                    update_archive(x_start, f_sp)

        # Return the archive (return value is ignored by evaluator)
        if archive_x:
            return np.array(archive_x)
        return np.array([])
# EVOLVE-BLOCK-END
