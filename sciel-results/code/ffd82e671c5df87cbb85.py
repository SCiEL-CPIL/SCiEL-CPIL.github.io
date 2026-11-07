"""Multi-objective optimizer for ZDT1 using MOEA/D (decomposition with neighborhood mating)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D-style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance diversity and convergence
        # For 2 objectives, we want enough points to cover the front well
        pop_size = min(91, max(11, budget // 2))

        # Generate uniformly spaced weight vectors for 2 objectives
        # Weights are (w1, w2) where w1 + w2 = 1, w1, w2 >= 0
        # Use pop_size equally spaced weights from 0 to 1
        weights = np.array([[i / (pop_size - 1), 1 - i / (pop_size - 1)] for i in range(pop_size)])

        # Build neighborhood: for each weight, find the closest K neighbors
        # For 2D weights on a line, neighbors are simply adjacent indices
        K = 5  # neighborhood size
        neighbors = []
        for i in range(pop_size):
            # Get K closest neighbors (circular)
            idxs = []
            for d in range(1, K + 1):
                idxs.append((i - d) % pop_size)
                idxs.append((i + d) % pop_size)
            neighbors.append(idxs)

        # Initialize population randomly
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive sigma for mutation
        sigma = 0.2
        sigma_min = 0.01
        sigma_decay = 0.995

        # Track best Tchebycheff value per weight for replacement decision
        # We'll use the normalized objectives for Tchebycheff aggregation

        def tchebycheff(f, w, f_min, f_max):
            """Compute Tchebycheff aggregation value."""
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            f_norm = (f - f_min) / f_range
            # Avoid division by zero in weights
            w_safe = np.where(w > 1e-10, w, 1.0)
            val = np.max(f_norm / w_safe)
            return val

        # Main loop: each iteration, pick a weight, create one child
        # We'll cycle through weights in a shuffled order each generation
        generation = 0
        while used < budget:
            # Shuffle weight order for this generation
            weight_order = np.random.permutation(pop_size)

            for w_idx in weight_order:
                if used >= budget:
                    break

                w = weights[w_idx]

                # Select parent from neighborhood of w_idx
                # Use tournament selection based on Tchebycheff value for weight w
                neigh = neighbors[w_idx]
                # Add the index itself
                candidates = list(set(neigh + [w_idx]))

                # Compute Tchebycheff values for candidates
                f_min = np.min(F, axis=0)
                f_max = np.max(F, axis=0)
                tcheb_vals = np.array([tchebycheff(F[i], w, f_min, f_max) for i in candidates])

                # Select parent: lowest Tchebycheff value (best for this weight)
                best_local = np.argmin(tcheb_vals)
                parent_idx = candidates[best_local]

                # SBX crossover with a second parent from neighborhood
                # Or just mutation for simplicity and speed
                # Let's use mutation with adaptive sigma

                child = X[parent_idx].copy()
                # Gaussian mutation
                mutation = np.random.standard_normal(dim) * sigma * (hi - lo)
                child += mutation
                child = np.clip(child, lo, hi)

                # Evaluate child
                fc = problem(child)
                used += 1

                # Replacement: replace the point with the worst Tchebycheff value
                # among the neighborhood of w_idx (including w_idx itself)
                # Only replace if child has better Tchebycheff value for weight w

                # Compute Tchebycheff value for child
                child_tcheb = tchebycheff(fc, w, f_min, f_max)

                # Find the worst point in the neighborhood
                worst_idx = w_idx
                worst_val = tchebycheff(F[w_idx], w, f_min, f_max)

                for i in neighbors[w_idx]:
                    val_i = tchebycheff(F[i], w, f_min, f_max)
                    if val_i > worst_val:
                        worst_val = val_i
                        worst_idx = i

                # Replace if child is better
                if child_tcheb < worst_val:
                    X[worst_idx] = child
                    F[worst_idx] = fc

            # Decay sigma
            sigma = max(sigma_min, sigma * sigma_decay)
            generation += 1

        return X

# EVOLVE-BLOCK-END
