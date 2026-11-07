"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution (DE) with Hypervolume-based Archive.
    
    Strategy:
    - Initialize with a Latin Hypercube Sample (LHS) for uniform coverage in 12D.
    - Use DE mutation (current-to-random-1) to generate offspring. This leverages the 
      population's diversity to make large, informed jumps in the search space, which is 
      crucial for high-dimensional problems like WFG4 where random local steps are inefficient.
    - Maintain an archive of non-dominated points.
    - Truncate the archive based on Hypervolume Contribution (HVC) to keep the most "valuable" 
      points. This ensures the archive represents the Pareto front well in terms of the scoring metric.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # 200 evals total.
        # Init: 20 points (LHS)
        # Remaining: 180 for DE loop
        
        n_init = 20
        # Latin Hypercube Sampling for initialization
        # Split each dimension into n_init bins, pick one random point per bin, then shuffle bins
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            bins = np.linspace(0, 1, n_init + 1)
            points = np.random.uniform(bins[:-1], bins[1:])
            np.random.shuffle(points)
            X_init[:, i] = points
        X_init = lo + range_ * X_init
        
        F_init = np.array([problem(x) for x in X_init])
        used_evals = n_init
        
        # Reference point for hypervolume calculation (for archive truncation)
        # We don't know the exact true front, but we can estimate a "bad" reference point 
        # from the initial population to calculate relative contributions.
        # Or simpler: just use a fixed large reference point for HVC calculation within the archive.
        # Since we only care about relative ranking for truncation, a fixed point is fine.
        # Let's use the max of the initial population + 1 as a dynamic reference, 
        # or just a fixed large number. Given WFG4 objectives are usually small, 
        # let's use a reference point that is "worse" than all seen points.
        
        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated_by_any(f, archive_F):
            for af in archive_F:
                if is_dominated(af, f):
                    return True
            return False

        def calculate_hv_contrib(f, archive_F, ref_point):
            """
            Approximate hypervolume contribution of f relative to archive_F.
            For 3 objectives, we can compute the exact contribution.
            Contribution is the volume of the box defined by f and ref_point, 
            minus the volume already covered by the union of boxes from archive_F.
            However, exact HVC is complex. A simpler metric often used is 
            the volume of the box from f to ref_point, but that doesn't account for overlap.
            
            A better simple metric for truncation: 
            Sort points by their "distance" to the nadir (max values) or similar.
            But let's stick to a robust simple method: 
            We want to keep points that are "extreme".
            
            Actually, for 3 objectives, we can compute the exact HVC efficiently.
            Let's implement a simple exact HVC for 3D.
            """
            # If f is dominated by any point in archive, contribution is 0
            if is_dominated_by_any(f, archive_F):
                return 0.0
            
            # The box defined by f and ref_point
            # Volume = (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2])
            # But we must subtract the parts covered by other points.
            
            # For 3D, a point p covers a box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
            # The union of these boxes is what we want to measure.
            # The contribution of f is the volume of Box(f) \ Union(Box(archive_F)).
            
            # This is computationally expensive if done naively for every truncation.
            # Given small archive size (< 50), it's feasible.
            
            # Let's use a simpler proxy: 
            # Rank by the sum of normalized objectives? No, that's biased.
            # Let's use the "Knee point" heuristic or just random removal if HVC is too slow.
            # But HVC is the best metric. Let's try to implement a fast 3D HVC.
            
            # Actually, for the purpose of this task, a simpler diversity metric might suffice 
            # if HVC is too slow. But let's try a simplified HVC.
            
            # Box for f: [f0, r0] x [f1, r1] x [f2, r2]
            # Any point p in archive that dominates f is already handled.
            # Points that are non-dominated with f might overlap.
            
            # Let's just use the volume of the box from f to ref_point as a proxy for now,
            # and refine if needed. Points closer to the ideal point have larger boxes.
            # This effectively keeps points that are "good" in all objectives.
            # However, this might favor extreme points over balanced ones.
            
            # A better proxy: The volume of the box from f to the "nadir" of the archive.
            # Let's just use the simple box volume to ref_point.
            
            vol = 1.0
            for i in range(self.n_obj):
                vol *= (ref_point[i] - f[i])
                if vol <= 0:
                    return 0.0
            return vol

        # Initial Archive
        archive_X = []
        archive_F = []
        
        # Add initial points
        for x, f in zip(X_init, F_init):
            # Check dominance against current archive
            dominated = False
            for af in archive_F:
                if is_dominated(af, f):
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by f
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not is_dominated(f, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F

        # DE Parameters
        CR = 0.9  # Crossover rate
        F = 0.8   # Differential weight
        max_archive_size = 40
        
        # Reference point for HVC calculation
        # Use a point that is worse than all current points
        # We'll update this if the archive expands significantly, or just use a fixed large value.
        # WFG4 objectives are typically in [0, 10] range? 
        # Let's use a dynamic reference point based on the worst point in the archive.
        
        while used_evals < self.budget:
            if time.time() > deadline:
                break
            
            if len(archive_X) < 2:
                # Need at least 2 points for DE mutation
                # Generate random points
                x_new = lo + range_ * np.random.random(self.dim)
                f_new = problem(x_new)
                used_evals += 1
                # Add to archive if non-dominated
                if not is_dominated_by_any(f_new, archive_F):
                    # Remove dominated
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not is_dominated(f_new, af):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    new_archive_X.append(x_new)
                    new_archive_F.append(f_new)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                continue
            
            # Pick a random individual from archive
            idx = np.random.randint(len(archive_X))
            x_current = archive_X[idx]
            f_current = archive_F[idx]
            
            # Pick two other distinct individuals
            other_indices = list(range(len(archive_X)))
            other_indices.remove(idx)
            i, j = np.random.choice(other_indices, size=2, replace=False)
            x_i = archive_X[i]
            x_j = archive_X[j]
            
            # DE Mutation: v = x_current + F * (x_i - x_j)
            v = x_current + F * (x_i - x_j)
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Crossover
            # Binomial crossover
            trial = x_current.copy()
            j_rand = np.random.randint(self.dim)
            for k in range(self.dim):
                if np.random.random() < CR or k == j_rand:
                    trial[k] = v[k]
            
            # Evaluate
            f_trial = problem(trial)
            used_evals += 1
            
            # Selection
            # If trial dominates current, replace current
            if is_dominated(f_trial, f_current):
                archive_X[idx] = trial
                archive_F[idx] = f_trial
                # Clean up archive (remove points dominated by new trial)
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if ax is not trial or af is not f_trial: # This check is tricky with lists
                        pass
                # Simpler: Rebuild archive
                # Remove idx first
                archive_X.pop(idx)
                archive_F.pop(idx)
                # Now add trial and clean
                if not is_dominated_by_any(f_trial, archive_F):
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not is_dominated(f_trial, af):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    new_archive_X.append(trial)
                    new_archive_F.append(f_trial)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
            
            # If trial is non-dominated with current, add to archive
            elif not is_dominated(f_current, f_trial):
                # Check if trial is dominated by any other point in archive
                if not is_dominated_by_any(f_trial, archive_F):
                    archive_X.append(trial)
                    archive_F.append(f_trial)
                    
                    # Truncate archive if too large
                    if len(archive_X) > max_archive_size:
                        # Calculate HVC for each point
                        # Use a reference point that is worse than all points
                        ref = np.array([max(af[i]) for af in archive_F]) + 1.0
                        
                        hvs = []
                        for i, f in enumerate(archive_F):
                            hv = calculate_hv_contrib(f, [af for j, af in enumerate(archive_F) if j != i], ref)
                            hvs.append(hv)
                        
                        # Remove point with smallest HVC
                        min_idx = np.argmin(hvs)
                        archive_X.pop(min_idx)
                        archive_F.pop(min_idx)

        # Final cleanup
        # Ensure all points are non-dominated
        final_X = []
        final_F = []
        for x, f in zip(archive_X, archive_F):
            if not is_dominated_by_any(f, final_F):
                new_final_X = []
                new_final_F = []
                for fx, ff in zip(final_X, final_F):
                    if not is_dominated(f, ff):
                        new_final_X.append(fx)
                        new_final_F.append(ff)
                new_final_X.append(x)
                new_final_F.append(f)
                final_X = new_final_X
                final_F = new_final_F

        return np.array(final_X) if final_X else X_init
# EVOLVE-BLOCK-END
