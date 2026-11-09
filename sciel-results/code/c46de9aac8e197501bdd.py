"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-estimation-based island model for WFG4 multi-objective optimization."""

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

        # Create islands with different weight vectors
        # For 3 objectives, use weight vectors that cover the simplex
        n_islands = 8
        weights = self._generate_weights(n_islands, n_obj)

        # Initialize populations
        pop_size = max(5, min(20, budget // (2 * n_islands)))
        total_init = pop_size * n_islands
        if total_init > budget:
            pop_size = max(2, budget // n_islands)
            total_init = pop_size * n_islands

        rng = np.random.RandomState(0)
        all_X = []
        all_F = []
        used = 0

        # Initialize each island with random points
        islands_X = []
        islands_F = []
        for i in range(n_islands):
            X_i = lo + (hi - lo) * rng.random((pop_size, dim))
            F_i = np.array([problem(x) for x in X_i])
            used += pop_size
            islands_X.append(X_i)
            islands_F.append(F_i)
            all_X.extend(X_i)
            all_F.extend(F_i)

        # Main optimization loop
        evals_left = budget - used

        # Use a combination of local search (L-MADS style) and global moves
        # For each island, focus on points that are good in the weighted sense
        while used < budget:
            # Pick an island
            island_idx = rng.randint(n_islands)
            X_i = islands_X[island_idx]
            F_i = islands_F[island_idx]
            w = weights[island_idx]

            # Compute weighted scalarization
            scores = F_i @ w
            # Pick a good parent (low score)
            parent_idx = np.argmin(scores)
            parent = X_i[parent_idx]
            parent_F = F_i[parent_idx]

            # Mutation: adaptive step size
            # Use a step size that depends on how well we're doing
            # Smaller steps for exploitation, larger for exploration
            base_step = 0.05
            # If we have many evaluations left, use larger steps early
            progress = 1.0 - (used / budget)
            step_size = base_step * (0.5 + 0.5 * progress)

            # Generate mutant
            mutant = parent + step_size * (hi - lo) * rng.standard_normal(dim)
            mutant = np.clip(mutant, lo, hi)

            f_mutant = problem(mutant)
            used += 1
            evals_left -= 1

            # Update the island: replace the worst point in this island
            worst_idx = np.argmax(scores)
            X_i[worst_idx] = mutant
            F_i[worst_idx] = f_mutant
            islands_X[island_idx] = X_i
            islands_F[island_idx] = F_i

            all_X.append(mutant)
            all_F.append(f_mutant)

            # Occasionally do a global restart in a random island
            if evals_left > 0 and rng.random() < 0.1:
                global_idx = rng.randint(n_islands)
                if global_idx != island_idx:
                    X_g = islands_X[global_idx]
                    F_g = islands_F[global_idx]
                    w_g = weights[global_idx]
                    scores_g = F_g @ w_g
                    parent_g = X_g[np.argmin(scores_g)]
                    mutant_g = parent_g + 0.15 * (hi - lo) * rng.standard_normal(dim)
                    mutant_g = np.clip(mutant_g, lo, hi)
                    f_mutant_g = problem(mutant_g)
                    used += 1
                    evals_left -= 1
                    worst_g = np.argmax(scores_g)
                    X_g[worst_g] = mutant_g
                    F_g[worst_g] = f_mutant_g
                    islands_X[global_idx] = X_g
                    islands_F[global_idx] = F_g
                    all_X.append(mutant_g)
                    all_F.append(f_mutant_g)

        return all_X

    def _generate_weights(self, n_islands, n_obj):
        """Generate weight vectors for islands covering the simplex."""
        # Use a grid of weight vectors
        if n_obj == 3:
            weights = []
            # Create a 2D grid in the simplex
            n_per_axis = int(np.ceil(n_islands ** 0.5))
            for i in range(n_per_axis):
                for j in range(n_per_axis):
                    if len(weights) >= n_islands:
                        break
                    w1 = i / max(1, n_per_axis - 1)
                    w2 = j / max(1, n_per_axis - 1)
                    w3 = 1.0 - w1 - w2
                    if w1 >= 0 and w2 >= 0 and w3 >= 0:
                        # Add small noise to avoid exact zeros
                        weights.append([w1 + 0.01, w2 + 0.01, w3 + 0.01])
            # Normalize
            weights = np.array(weights)
            weights = weights / weights.sum(axis=1, keepdims=True)
            # If we don't have enough, add random ones
            while len(weights) < n_islands:
                w = rng_dirichlet(n_obj)
                weights = np.vstack([weights, w])
            return weights[:n_islands]
        else:
            # For general case, use random Dirichlet
            weights = []
            for _ in range(n_islands):
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)
            return np.array(weights)


def rng_dirichlet(n_obj):
    """Generate a random Dirichlet-distributed vector."""
    x = np.random.exponential(1.0, size=n_obj)
    return x / x.sum()
# EVOLVE-BLOCK-END
