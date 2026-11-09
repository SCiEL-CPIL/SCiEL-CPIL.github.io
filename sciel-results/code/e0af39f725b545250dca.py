"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive evolutionary multi-objective search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = min(30, self.budget // 3)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation scale
        mut_scale = 0.1

        def is_non_dominated(F):
            """Return boolean array: True if point is non-dominated."""
            n = len(F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom[i] = False
                        break
            return non_dom

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = is_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Select parent from non-dominated set
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]

            # Adaptive mutation
            child = parent + mut_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Local refinement around the new child if it looks promising
            # (i.e., it's not immediately dominated by the current front)
            # We do a quick check: does it dominate any current point?
            # If yes, it's likely on or near the front, so refine it.
            # Simple check: is it non-dominated in the current population + child?
            # For speed, we just check if it's better than the worst point in sum.
            # Or better: check if it's non-dominated against the current non-dominated set.
            
            # To keep it simple and fast, we do a few small steps around the child
            # only if the child is not dominated by any point in the current non-dominated set.
            
            # Check if child is dominated by any non-dominated point
            child_is_dominated = False
            for idx in nd_indices:
                if np.all(F[idx] <= fc) and np.any(F[idx] < fc):
                    child_is_dominated = True
                    break
            
            if not child_is_dominated:
                # Do 2-3 local refinement steps
                local_steps = 2
                for _ in range(local_steps):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Small mutation around the child
                    local_mut = mut_scale * 0.1 * span
                    child_local = child + local_mut * rng.standard_normal(self.dim)
                    child_local = np.clip(child_local, lo, hi)
                    fc_local = problem(child_local)
                    used += 1
                    
                    # Accept if it improves the child in at least one objective
                    # without worsening others too much (simple dominance check)
                    if np.any(fc_local < fc) and not (np.all(fc_local >= fc) and np.any(fc_local > fc)):
                        # This is a bit loose. Let's just accept if it's non-dominated
                        # relative to the current best (the child itself)
                        # Actually, let's just replace the child with the local one
                        # if the local one is better in sum or non-dominated.
                        # For simplicity, we'll just update the child if it's better in sum.
                        if np.sum(fc_local) < np.sum(fc):
                            child = child_local
                            fc = fc_local
                        else:
                            # Still record the evaluation, but maybe keep the original child
                            # We've already evaluated it, so it's in the record.
                            # We just don't use it as the new "best" for the next step.
                            pass
                    # If dominated, we just skip updating the child for the next step
                    # but the evaluation is still recorded.

            # Replace worst individual in population
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

            # Adapt mutation scale based on progress
            if used % 20 == 0:
                progress = used / self.budget
                mut_scale = 0.1 * (1.0 - 0.5 * progress)

        return X
# EVOLVE-BLOCK-END
