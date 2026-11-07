"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Lattice-Based Local Search.
    Designed for WFG4 (12 dims, 3 objectives) with a budget of 200 evaluations.
    
    Strategy:
    - Initialize with a stratified sample (Latin Hypercube style) to ensure broad coverage.
    - Use a simple archive to store non-dominated points.
    - Perform local search around the best points in the archive using coordinate-wise perturbations.
    - This deterministic/semi-deterministic approach avoids the random walk behavior of PSO/DE
      which can get stuck in local optima or fail to cover the Pareto front effectively in 12D.
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
        # Initialization: 16 points (4^2 for 12D? No, just a reasonable stratified sample).
        # Let's use 16 initial points.
        # Remaining: 184 evals for local search.
        
        # 1. Initialization: Stratified Sampling
        # We want points spread out. A simple way is to divide each dimension into k bins
        # and pick one point per bin combination, but that's 2^12 = 4096 which is too many.
        # Instead, use a Latin Hypercube or just random but ensuring spread.
        # For 12D and 16 points, a simple random start is often okay if local search is strong.
        # However, to be "materially different", let's use a structured grid for the first few dimensions
        # and random for the rest? No, let's use a Halton sequence or similar for better uniformity.
        
        # Simpler: Use numpy random but seed it consistently if needed? 
        # The problem says np.random.seed(seed) is called by evaluator. 
        # So np.random is seeded. We should use np.random directly for reproducibility per seed.
        
        n_init = 16
        X_init = lo + range_ * np.random.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        used_evals = n_init
        
        # Archive management
        # Store (x, f) pairs.
        archive_X = []
        archive_F = []
        
        def is_dominated(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 < f2
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any in archive
            for af in archive_F:
                if is_dominated(af, f):
                    return False
            # Remove any points in archive dominated by new point
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
            return True

        for x, f in zip(X_init, F_init):
            add_to_archive(x, f)

        # 2. Local Search Phase
        # We will iterate through the archive. For each point, we try to improve it.
        # If a neighbor is found that is non-dominated (or dominates the current), we accept it.
        # To maintain diversity, we don't just chase the global best; we refine each archive member.
        
        # Define step sizes for local search
        # Start with a relatively large step, shrink if no improvement?
        # Or use a fixed set of steps.
        # Given 184 evals, and ~10-20 points in archive, we can do ~10-20 steps per point.
        
        # Directions: Coordinate axes + a few random directions
        directions = np.eye(self.dim)
        # Add 4 random directions
        for _ in range(4):
            d = np.random.randn(self.dim)
            d /= np.linalg.norm(d)
            directions = np.vstack([directions, d])
        
        # Local search loop
        # We will perform a series of "refinement rounds".
        # In each round, we pick a point from the archive (cycling or random) and try to move it.
        
        # To ensure we don't waste evals on points that are clearly worse, we prioritize
        # points that are "extreme" in the objective space? 
        # For simplicity, let's just cycle through the archive.
        
        # Step size strategy:
        # Initial step: 0.1 * range
        # If no improvement in a point's local neighborhood, shrink step?
        # Let's keep it simple: try steps [0.2, 0.1, 0.05, 0.02] * range for each direction.
        
        step_factors = [0.2, 0.1, 0.05, 0.02]
        
        # We have ~184 evals left.
        # Let's structure this as:
        # While budget allows:
        #   Pick a random point from archive (or the one with highest "potential")
        #   Try a random direction and step
        #   If improvement (non-dominated), update archive.
        
        # To be more efficient, let's do a "coordinate descent" style search on the top points.
        # But "top" is ambiguous in MO.
        
        # Alternative Strategy: 
        # Take the current archive. For each point in the archive, perform a local search.
        # If we find a better point, replace the original in the archive (or add new).
        
        # Let's iterate through the archive multiple times.
        # Each time, for each point, try to improve it using coordinate steps.
        
        # Since the archive size is small (< 30), we can afford to be thorough.
        
        # We will use a "simulated annealing" like acceptance? No, just non-dominated replacement.
        # If new point dominates old, replace. If new point is non-dominated with old, add new (archive grows).
        # If old dominates new, reject.
        
        # To prevent archive explosion, we cap it. If > 20, remove the "least interesting" point.
        # "Least interesting" could be the one with the smallest hypervolume contribution? 
        # That's expensive. Let's just random removal if > 20.
        
        max_archive_size = 20
        
        while used_evals < self.budget:
            if time.time() > deadline:
                break
            
            # If archive is empty, break (should not happen)
            if len(archive_X) == 0:
                break
            
            # Pick a point to refine. 
            # Strategy: Random selection from archive to maintain diversity.
            idx = np.random.randint(len(archive_X))
            x_current = archive_X[idx]
            f_current = archive_F[idx]
            
            # Try a random direction and step
            dir_idx = np.random.randint(len(directions))
            d = directions[dir_idx]
            
            # Try a random step factor
            step_factor = step_factors[np.random.randint(len(step_factors))]
            step_size = step_factor * range_
            
            # Move in positive and negative direction
            for sign in [1, -1]:
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x_new = x_current + sign * step_size * d
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used_evals += 1
                
                # Check dominance
                # Does f_new dominate f_current?
                if is_dominated(f_new, f_current):
                    # Replace current point in archive
                    archive_X[idx] = x_new
                    archive_F[idx] = f_new
                    # Also, check if it dominates other points in archive
                    # We need to re-verify the whole archive or just add and clean?
                    # Since we replaced, we should check if it dominates others.
                    # Let's just add it to a "pending" list and clean up later?
                    # Simpler: Just add to archive and let the add_to_archive logic handle it?
                    # But we already have it in the archive at idx. 
                    # Let's remove the old one and add the new one properly.
                    # Actually, add_to_archive assumes it's not already there.
                    # Let's just update the list directly and then clean up dominated points.
                    
                    # Remove dominated points from archive
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if ax is x_current: # This check is fragile if objects are identical
                            continue
                        if not is_dominated(f_new, af):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    
                    # Add the new point
                    new_archive_X.append(x_new)
                    new_archive_F.append(f_new)
                    
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    
                    # Cap archive size
                    if len(archive_X) > max_archive_size:
                        # Remove a random point (or the one that was just replaced? No, we removed it)
                        # Remove random
                        remove_idx = np.random.randint(len(archive_X))
                        archive_X.pop(remove_idx)
                        archive_F.pop(remove_idx)
                        
                elif is_dominated(f_current, f_new):
                    # Current is better, reject new
                    pass
                else:
                    # Non-dominated
                    # Add to archive if not already there
                    # Check if x_new is already in archive (unlikely due to random steps)
                    add_to_archive(x_new, f_new)
                    
                    if len(archive_X) > max_archive_size:
                        remove_idx = np.random.randint(len(archive_X))
                        archive_X.pop(remove_idx)
                        archive_F.pop(remove_idx)
                        
        # Final cleanup: ensure all points in archive are non-dominated with each other
        # (The add_to_archive logic should maintain this, but the direct replacement might have introduced issues)
        # Let's do a final non-dominated sort on the archive.
        
        final_X = []
        final_F = []
        
        for x, f in zip(archive_X, archive_F):
            dominated = False
            for af in final_F:
                if is_dominated(af, f):
                    dominated = True
                    break
            if not dominated:
                # Remove any in final_F dominated by f
                new_final_X = []
                new_final_F = []
                for fx, ff in zip(final_X, final_F):
                    if not is_dominated(f, ff):
                        new_final_X.append(fx)
                        new_final_F.append(ff)
                final_X = new_final_X
                final_F = new_final_F
                final_X.append(x)
                final_F.append(f)
        
        # The evaluator records all problem(x) calls, so the archive state doesn't strictly matter for scoring
        # unless we return something that triggers further evaluation? No, return value is ignored.
        # But we must not exceed budget.
        
        return np.array(final_X) if final_X else X_init
# EVOLVE-BLOCK-END
