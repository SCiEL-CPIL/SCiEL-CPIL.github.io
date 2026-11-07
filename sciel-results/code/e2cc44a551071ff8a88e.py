"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Use a local RNG for speed and reproducibility
        # The task mentions np.random.seed is called, but using a local one is safer 
        # and the 1.0000 solution did this successfully.
        rng = np.random.default_rng(42)

        # Population size: small enough for fast dominance checks, large enough for coverage
        # 1.0000 solution used 20. Let's try 30 for slightly better coverage, still fast.
        pop_size = 30
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Fast non-dominated check
            # F is (N, M). 
            # dominated[i] is True if there exists j such that F[j] <= F[i] and F[j] != F[i]
            # Vectorized approach:
            # diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]
            # This creates a (N, N, M) array. For N=30, M=3, this is 30*30*3 = 2700 floats. Very fast.
            
            # Check if j dominates i: F[j] <= F[i] for all obj, and F[j] < F[i] for at least one.
            # Let's compute a matrix D where D[j, i] is True if j dominates i.
            
            # F[j, :] <= F[i, :] -> (N, N, M) boolean
            # We want: for each i, is there a j != i such that all(F[j] <= F[i]) and any(F[j] < F[i])?
            
            # Efficient way:
            # less_or_equal = F[:, np.newaxis, :] <= F[np.newaxis, :, :]  # (N, N, M)
            # strictly_less = F[:, np.newaxis, :] < F[np.newaxis, :, :]   # (N, N, M)
            
            # j dominates i if:
            # np.all(less_or_equal[j, i, :], axis=2) and np.any(strictly_less[j, i, :], axis=2)
            
            # Let's compute domination matrix
            # Note: F is (N, M)
            # F_expanded_j = F[:, np.newaxis, :] # (N, 1, M) -> j index
            # F_expanded_i = F[np.newaxis, :, :] # (1, N, M) -> i index
            
            # We want to know for each i if it is dominated by any j.
            
            # dom_matrix[j, i] = True if j dominates i
            # condition: all(F[j] <= F[i]) and any(F[j] < F[i])
            
            # To avoid the full (N,N,M) array if N gets large, but N=30 is tiny.
            # Let's do it.
            
            F_j = F[:, np.newaxis, :]  # (N, 1, M)
            F_i = F[np.newaxis, :, :]  # (1, N, M)
            
            # (N, N, M)
            le = F_j <= F_i
            lt = F_j < F_i
            
            # all_le: (N, N) - True if F[j] <= F[i] for all objectives
            all_le = np.all(le, axis=2)
            any_lt = np.any(lt, axis=2)
            
            # dom[j, i] is True if j dominates i
            dom = all_le & any_lt
            
            # Exclude self-domination (diagonal)
            np.fill_diagonal(dom, False)
            
            # dominated[i] is True if any j dominates i
            # dom is (N, N) where row j, col i.
            # We want to know if any row j has dom[j, i] = True.
            # So sum over axis 0 (j).
            dominated = np.any(dom, axis=0)
            
            # Find non-dominated indices
            nd_indices = np.where(~dominated)[0]
            
            # If all are dominated (shouldn't happen with strict dominance check usually, 
            # but if there are duplicates or issues, fallback to all)
            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)
            
            # Select parent from non-dominated
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]
            
            # Mutation: adaptive step
            # Start with larger steps, shrink over time?
            # 1.0000 used fixed 0.1.
            # Let's try a slight adaptation: 0.1 to 0.05
            progress = used / self.budget
            step_size = 0.1 * (1 - 0.5 * progress)
            
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replacement:
            # 1.0000 replaced the worst by sum(F).
            # This is a simple and effective heuristic for multi-objective optimization 
            # as it tends to remove points that are far from the origin in the objective space.
            
            # However, to be slightly more robust, we can check if the child is non-dominated.
            # If child is non-dominated, replace the worst dominated point if any, else worst by sum.
            # If child is dominated, replace the worst dominated point.
            
            # Let's stick to the simple and proven "replace worst by sum" as it got 1.0.
            # It naturally handles both cases: if child is good, it won't be the worst. 
            # If child is bad, it might be the worst and get replaced? No, we replace 
            # the worst *existing* point with the child.
            # So if child is very bad, it replaces a point that was even worse? 
            # No, we calculate worst on the *current* population F, then replace it.
            # So if child is bad, we might replace a good point with a bad one?
            # Wait. The 1.0000 code does:
            # worst = argmax(sum(F))
            # X[worst], F[worst] = child, fc
            # This means: Find the point in the current population with the highest sum of objectives.
            # Replace it with the new child.
            # This is a "greedy" replacement. It assumes the child is better than the worst.
            # This is NOT always true.
            
            # A safer strategy:
            # 1. Check if child dominates any point in F. If yes, replace the most dominated one.
            # 2. If child is dominated by existing points, replace the worst point only if child is better than that worst point in some sense?
            # Or simply: Replace the point with the highest sum(F) ONLY IF the child has a lower sum(F) than that point?
            # Or just always replace the worst?
            
            # Let's look at the 1.0000 code again. It ALWAYS replaces the worst.
            # This is a risk. If the child is terrible, we lose a good point.
            # However, with random initialization and mutation, and the fact that we start with random points,
            # the "worst" is likely very bad, and the child is likely better on average.
            
            # Let's improve this slightly:
            # Compare child to the worst point.
            # If child is better (in terms of sum) than the worst, replace.
            # Otherwise, maybe replace a random point? Or do nothing?
            # Doing nothing wastes the evaluation.
            
            # Better: 
            # If child is non-dominated, replace the worst dominated point (or worst by sum if no dominated).
            # If child is dominated, replace the worst dominated point only if child is "less dominated" or better in sum?
            
            # Let's try a hybrid:
            # 1. Calculate sum_F for all points.
            # 2. Find index of worst point (max sum).
            # 3. If sum(child) < sum(worst_point), replace worst with child.
            # 4. Else, if child is non-dominated, replace the point with the smallest crowding distance (approx by sum) among dominated?
            #    Actually, if child is non-dominated, it's valuable. We should replace the worst point regardless?
            #    If child is non-dominated, it's likely better than the worst dominated point.
            #    So: If child is non-dominated, replace worst by sum.
            #    If child is dominated, replace worst by sum ONLY if child has lower sum than worst.
            #    If child is dominated and has higher sum than worst, we still have to put it somewhere?
            #    We can replace a random point to maintain diversity?
            
            # Let's stick to the 1.0000 logic but add a check:
            # Only replace if the child is better than the worst in terms of sum of objectives.
            # This prevents degrading the population.
            
            worst_idx = np.argmax(np.sum(F, axis=1))
            worst_sum = np.sum(F[worst_idx])
            child_sum = np.sum(fc)
            
            if child_sum < worst_sum:
                X[worst_idx] = child
                F[worst_idx] = fc
            else:
                # Child is not better than the worst.
                # Maybe replace a random point to explore?
                # Or just keep the population as is? 
                # If we keep it, we are not using the evaluation to improve the set.
                # Let's replace a random point to maintain diversity, but only if the child is non-dominated?
                # If child is dominated and worse than worst, it's useless.
                # Let's check if child is non-dominated.
                
                # Re-calculate domination for the child vs F?
                # This is expensive.
                
                # Simpler: Just replace the worst point anyway? 
                # The 1.0000 solution did that. 
                # Why did it work? Because the mutation is small, so the child is usually close to the parent,
                # and if the parent was non-dominated, the child is likely also good.
                # If the parent was dominated, the child might be better.
                
                # I will revert to the 1.0000 simple replacement to ensure we don't break the winning strategy.
                # The "risk" of replacing a good point with a bad one is mitigated by the fact that 
                # the "worst" is defined by sum of objectives, which is a strong indicator of quality 
                # for minimizing problems with positive objectives (like WFG7).
                
                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
