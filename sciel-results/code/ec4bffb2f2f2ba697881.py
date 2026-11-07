import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def is_dominated(f, archive_F):
            for g in archive_F:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def dominates(f, g):
            return np.all(f <= g) and np.any(f < g)

        # 1. Initialize Archive with random points
        # Use a smaller initial set to leave budget for CMA-ES
        init_size = 20
        archive_X = []
        archive_F = []
        
        for _ in range(init_size):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            archive_X.append(x)
            archive_F.append(f)
        
        # Sort initial archive to remove dominated points
        # Since it's small, simple O(N^2) is fine
        # We want to keep only non-dominated points
        # Let's just keep all for now, CMA-ES will handle diversity via sampling
        # But for the "mean" center, we should use the non-dominated ones or all?
        # Standard MO-CMA-ES uses the whole population or a selected subset.
        # Let's use the entire current pool for the first iteration to get a good center.
        
        # 2. CMA-ES Loop
        # Parameters
        pop_size = 10  # lambda
        max_archive_size = 50
        step_size = 0.5 * np.max(hi - lo)
        sigma = step_size
        
        # Initialize CMA-ES state
        # Center is the mean of the initial non-dominated points (or all if none)
        if archive_F:
            # Filter to non-dominated for a better center
            nd_indices = []
            for i in range(len(archive_F)):
                if not is_dominated(archive_F[i], [archive_F[j] for j in range(len(archive_F)) if j != i]):
                    nd_indices.append(i)
            
            if nd_indices:
                center = np.mean([archive_X[i] for i in nd_indices], axis=0)
            else:
                center = np.mean(archive_X, axis=0)
        else:
            center = (lo + hi) / 2.0

        # Covariance matrix C
        C = np.eye(dim)
        # Inverse square root of C for transformation
        # We can update this less frequently or use a simplified update for speed
        # For dim=4, full Cholesky is fast.
        
        # We will use a simplified "memoryless" or "fast" update for C to save time
        # or just keep C = I and rely on sigma and center movement?
        # No, adaptation is key. Let's do standard CMA-ES update but only on the
        # non-dominated subset or the best subset.
        
        # To keep it simple and robust within time limits:
        # We will sample from N(center, sigma^2 * C).
        # Update center to the mean of the non-dominated offspring + parents.
        # Update sigma based on the spread.
        # Update C using the rank-1/rank-mu update rule on the selected points.
        
        # Given the small budget (200), we might only do ~15-18 generations.
        # Each generation: 10 evals.
        
        while used < budget and time.time() < deadline:
            gen_start = time.time()
            
            # 1. Sample offspring
            offspring_X = []
            offspring_F = []
            
            # Check if we have enough budget for a full population
            if used + pop_size > budget:
                pop_size = budget - used
            
            if pop_size <= 0:
                break
                
            # Update C and sigma if we have history? 
            # For simplicity and speed, let's use a "natural gradient" like update 
            # or just standard CMA-ES.
            
            # Let's implement a lightweight CMA-ES update.
            # We need to store the previous population to update C?
            # Standard CMA-ES updates C using the weighted mean of the selected points.
            
            # To avoid complex state management, let's use a simpler strategy:
            # "Pareto-CMA" where we update the distribution parameters based on the
            # current archive's non-dominated points.
            
            # Actually, a very effective strategy for small budgets is:
            # 1. Sample from current distribution.
            # 2. Combine with archive.
            # 3. Select non-dominated points.
            # 4. Update center to mean of selected.
            # 5. Update sigma to maintain a target spread (e.g. 0.5 * current range).
            # 6. Update C to align with the principal axes of the selected points.
            
            # Let's do that.
            
            # Sample
            for _ in range(pop_size):
                # Standard normal
                z = np.random.randn(dim)
                # Transform using C^0.5
                # Since C is symmetric positive definite, we can use Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z # Fallback if C becomes singular
                
                x = center + sigma * y
                x = np.clip(x, lo, hi)
                
                f = evaluate(x)
                offspring_X.append(x)
                offspring_F.append(f)
            
            # 2. Combine Archive and Offspring
            combined_X = archive_X + offspring_X
            combined_F = archive_F + offspring_F
            
            # 3. Select Non-Dominated
            nd_X = []
            nd_F = []
            
            # Efficient non-dominated sort for small N
            # O(N^2) is fine for N < 100
            for i in range(len(combined_F)):
                f_i = combined_F[i]
                dominated = False
                for j in range(len(combined_F)):
                    if i == j:
                        continue
                    f_j = combined_F[j]
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    nd_X.append(combined_X[i])
                    nd_F.append(combined_F[i])
            
            # If no non-dominated points (shouldn't happen unless all dominated by new ones? 
            # No, at least the best one is non-dominated), keep the best one.
            if not nd_F:
                # Find the point with min sum of objectives
                sums = [np.sum(f) for f in combined_F]
                best_idx = np.argmin(sums)
                nd_X = [combined_X[best_idx]]
                nd_F = [combined_F[best_idx]]
            
            # 4. Update Parameters
            if nd_F:
                # New Center
                new_center = np.mean(nd_X, axis=0)
                
                # New Sigma
                # Estimate spread from the non-dominated points
                if len(nd_F) > 1:
                    # Use the range in objective space or decision space?
                    # Decision space spread is safer for step size.
                    # Let's use the standard deviation of the decision space points.
                    stds = np.std(nd_X, axis=0)
                    # Avoid zero std
                    stds = np.maximum(stds, 1e-6)
                    new_sigma = np.mean(stds)
                    
                    # Adaptive step size: keep it roughly constant or slightly shrinking?
                    # For exploration, maybe keep it similar to previous or slightly smaller.
                    # Let's blend with previous sigma to avoid drastic changes.
                    sigma = 0.7 * sigma + 0.3 * new_sigma
                    
                    # Update C
                    # Align C with the covariance of the non-dominated points
                    # Cov = (1/N) * sum((x_i - mean)(x_i - mean)^T)
                    centered = np.array(nd_X) - new_center
                    if len(centered) > 1:
                        new_C = (centered.T @ centered) / len(centered)
                        # Regularize to ensure positive definite
                        new_C += 1e-6 * np.eye(dim)
                        # Blend with previous C to maintain stability
                        C = 0.5 * C + 0.5 * new_C
                else:
                    # Only one point, keep sigma and C
                    pass
                
                center = new_center
            
            # 5. Update Archive
            # Keep the non-dominated points as the new archive
            # Limit archive size to avoid memory issues (though N is small)
            # If archive is too large, we might need to thin it out.
            # For now, just replace.
            archive_X = nd_X
            archive_F = nd_F
            
            # Safety check
            if time.time() > deadline - 5.0:
                break

        return all_X
