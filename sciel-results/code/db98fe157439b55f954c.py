"""Multi-objective optimizer on RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using weighted-sum scalarization with multi-start
    local search for structured coverage of the objective space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # We will track all evaluated points and their objective values
        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            X_list.append(x)
            F_list.append(f)
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for pf in F_set:
                if np.all(pf <= f) and np.any(pf < f):
                    return True
            return False

        def local_search(x0, f0, max_evals):
            """Perform local search around x0 using Nelder-Mead style steps."""
            nonlocal eval_count
            best_x = x0.copy()
            best_f = f0.copy()
            step_size = 0.1 * (hi - lo)
            n_steps = 0

            for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
                if not time_ok() or eval_count >= max_evals:
                    break
                for _ in range(6):
                    if not time_ok() or eval_count >= max_evals:
                        break
                    # Try random direction
                    direction = np.random.randn(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    new_x = best_x + direction * step * (hi - lo)
                    new_x = np.clip(new_x, lo, hi)
                    new_f = evaluate(new_x)

                    # Accept if not dominated by current best
                    if not (np.all(best_f <= new_f) and np.any(best_f < new_f)):
                        # Check if new point dominates best
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x.copy()
                            best_f = new_f.copy()
                        # Otherwise keep both - but for local search we track best
                        # Actually, for hypervolume we want diversity, so accept non-dominated
                        elif not (np.all(new_f <= best_f) and np.any(new_f < best_f)):
                            # Both non-dominated relative to each other; keep the one with better sum
                            if np.sum(new_f) < np.sum(best_f):
                                best_x = new_x.copy()
                                best_f = new_f.copy()
            return best_x, best_f

        # Generate weight vectors on the simplex for n_obj=3
        # Use a grid of weight vectors for structured coverage
        n_weights = 20
        weight_vectors = []
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                w = np.array([i / n_weights, j / n_weights, k / n_weights], dtype=np.float64)
                weight_vectors.append(w)

        # Shuffle weights to get diverse starting points
        np.random.shuffle(weight_vectors)

        # Phase 1: Initial random sampling
        n_random = min(30, budget // 4)
        for i in range(n_random):
            if not time_ok():
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # Phase 2: For each weight vector, do multi-start local search
        # Use remaining budget for local search around weight-based targets
        remaining_budget = budget - eval_count
        n_local_starts = min(len(weight_vectors), max(1, remaining_budget // 15))

        for wi in range(n_local_starts):
            if not time_ok():
                break

            w = weight_vectors[wi]

            # Find starting point: use existing non-dominated point closest to weight direction
            # Or just start from random
            if F_list:
                F_arr = np.array(F_list)
                # Score each point by how well it aligns with weight vector
                # Lower weighted sum = better for this weight
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                start_x = X_list[best_idx].copy()
                start_f = F_list[best_idx].copy()
            else:
                start_x = lo + (hi - lo) * np.random.random(dim)
                start_f = evaluate(start_x)

            # Do local search with a budget for this weight vector
            local_budget = min(15, budget - eval_count)
            if local_budget > 0:
                local_search(start_x, start_f, local_budget)

        # Phase 3: Use remaining budget for additional exploration
        # Try more weight vectors or refine existing solutions
        while time_ok() and eval_count < budget:
            # Pick a random weight vector
            w = np.random.dirichlet(np.ones(n_obj))

            # Start from a random point or existing good point
            if F_list and np.random.random() < 0.5:
                F_arr = np.array(F_list)
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                start_x = X_list[best_idx].copy()
                start_f = F_list[best_idx].copy()
            else:
                start_x = lo + (hi - lo) * np.random.random(dim)
                start_f = evaluate(start_x)

            # Small local search
            local_budget = min(10, budget - eval_count)
            if local_budget > 0:
                local_search(start_x, start_f, local_budget)

        return X_list
# EVOLVE-BLOCK-END
