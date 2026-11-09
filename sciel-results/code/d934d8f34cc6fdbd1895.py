# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-estimation-based island model for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        d = self.dim
        n_obj = self.n_obj

        budget = self.budget
        # Island model: 6 islands with different weight vectors
        n_islands = 6
        # Weight vectors for different regions of the objective space
        # WFG4 has 3 objectives; use diverse weight vectors
        weights = []
        for i in range(n_islands):
            w = np.zeros(n_obj)
            # Different weight patterns for different islands
            if i == 0:
                w[:] = 1.0 / n_obj  # uniform
            elif i == 1:
                w[0] = 0.6
                w[1:] = 0.4 / (n_obj - 1)
            elif i == 2:
                w[1] = 0.6
                w[:1] = 0.4 / (n_obj - 1)
                w[2:] = 0.4 / (n_obj - 1)
            elif i == 3:
                w[2] = 0.6
                w[:2] = 0.4 / (n_obj - 1)
            elif i == 4:
                w[0] = 0.5
                w[1] = 0.3
                w[2] = 0.2
            else:
                w[0] = 0.3
                w[1] = 0.3
                w[2] = 0.4
            weights.append(w)

        # Per-island population size
        pop_size = max(10, budget // (4 * n_islands))
        # Mutation intensity per island (higher for regions that need more exploration)
        mutation_rates = [0.15, 0.12, 0.12, 0.12, 0.10, 0.10]

        # Initialize islands
        islands_X = []
        islands_F = []
        for i in range(n_islands):
            X_i = lo + span * rng.random((pop_size, d))
            F_i = np.array([problem(x) for x in X_i])
            islands_X.append(X_i)
            islands_F.append(F_i)

        used = n_islands * pop_size

        # Helper: compute weighted scalarization for an island
        def scalarize(F_i, w):
            return F_i @ w

        # Helper: get non-dominated count in an island
        def non_dominated_mask(F_i):
            n = len(F_i)
            dominated = np.zeros(n, dtype=bool)
            for j in range(n):
                for k in range(n):
                    if j != k and np.all(F_i[k] <= F_i[j]) and np.any(F_i[k] < F_i[j]):
                        dominated[j] = True
                        break
            return ~dominated

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Evolve each island
            for i in range(n_islands):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                X_i = islands_X[i]
                F_i = islands_F[i]
                w = weights[i]
                mut_rate = mutation_rates[i]

                # Find non-dominated individuals
                nd_mask = non_dominated_mask(F_i)
                nd_indices = np.where(nd_mask)[0]

                # If no non-dominated, use all
                if len(nd_indices) == 0:
                    parents = X_i
                    parent_F = F_i
                else:
                    parents = X_i[nd_indices]
                    parent_F = F_i[nd_indices]

                # Select a parent and mutate
                p_idx = rng.integers(len(parents))
                parent = parents[p_idx]

                # Weight-specific mutation: larger steps in directions where weight is low
                # (explore more in under-weighted objective directions)
                w_norm = w / (np.sum(w) + 1e-12)
                # Inverse weights give higher mutation in less important directions
                inv_w = 1.0 / (w_norm + 0.1)
                inv_w = inv_w / np.sum(inv_w)

                # Multi-objective aware mutation: perturb each dimension with intensity
                # modulated by the weight vector
                noise = rng.standard_normal(d)
                # Scale mutation by inverse weight to explore under-emphasized regions
                mutation_step = mut_rate * span * noise * (0.5 + inv_w[np.arange(d) % n_obj])
                child = parent + mutation_step
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst individual in this island (by scalarization)
                scores = scalarize(F_i, w)
                worst_idx = int(np.argmax(scores))
                islands_X[i][worst_idx] = child
                islands_F[i][worst_idx] = fc

            # Migration: every few generations, swap best individuals between islands
            # (Simple ring migration)
            # Track generation count roughly
            if used % (n_islands * pop_size) < n_islands:
                for i in range(n_islands):
                    j = (i + 1) % n_islands
                    # Find best in island i by scalarization with island j's weight
                    w_j = weights[j]
                    scores_i = scalarize(islands_F[i], w_j)
                    best_i = int(np.argmin(scores_i))

                    # Find worst in island j by its own scalarization
                    scores_j = scalarize(islands_F[j], weights[j])
                    worst_j = int(np.argmax(scores_j))

                    # Migrate
                    islands_X[j][worst_j] = islands_X[i][best_i].copy()
                    islands_F[j][worst_j] = islands_F[i][best_i].copy()

        # After budget exhausted, collect all points
        all_X = np.vstack(islands_X)
        all_F = np.vstack(islands_F)

        # Return (the return value is ignored by evaluator, but keep interface)
        return all_X
# EVOLVE-BLOCK-END
