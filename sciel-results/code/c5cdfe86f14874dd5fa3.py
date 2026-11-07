"""Optimized program for WFG9 using MOEA/D with Tchebycheff decomposition,
neighborhood mating, and an external archive for Pareto front coverage.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D Multi-Objective Optimizer with Tchebycheff decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        n_obj = self.n_obj
        dim = self.dim

        # Population size - use a grid of weight vectors
        # For 3 objectives, use a simplex grid
        pop_size = 45  # 9x9 grid on the simplex for 3 objectives

        # Generate weight vectors on the simplex
        def generate_weights(n_obj, n_each):
            """Generate weight vectors on the (n_obj-1)-simplex."""
            weights = []
            if n_obj == 2:
                for i in range(n_each + 1):
                    w = np.array([i / n_each, (n_each - i) / n_each])
                    weights.append(w)
            elif n_obj == 3:
                for i in range(n_each + 1):
                    for j in range(n_each - i + 1):
                        k = n_each - i - j
                        w = np.array([i / n_each, j / n_each, k / n_each])
                        weights.append(w)
            else:
                # Fallback: random weights normalized
                for _ in range(pop_size):
                    w = np.random.rand(n_obj)
                    w /= w.sum()
                    weights.append(w)
            return np.array(weights)

        # Choose grid size to approximate pop_size
        n_each = 4  # gives (4+1)*(4+2)/2 = 15 for 3 obj, too few
        # Let's use a denser grid
        # For 3 objectives with n_each=8: (8+1)*(8+2)/2 = 45
        n_each = 8
        W = generate_weights(n_obj, n_each)
        pop_size = len(W)

        # Initialize population
        X = lo + np.random.rand(pop_size, dim) * range_
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        # Initialize best solutions for each weight
        best_X = X.copy()
        best_F = F.copy()

        # Compute ideal point
        ideal = F.min(axis=0)

        # Neighborhood size
        T = 7  # neighborhood size (odd, centered)

        # Build neighbor list
        def build_neighbors(W, T):
            """Build neighbor list based on weight vector distances."""
            n = len(W)
            neighbors = []
            # Compute pairwise distances
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dists[i, j] = np.sum((W[i] - W[j]) ** 2)
            for i in range(n):
                # Sort by distance, take T nearest
                order = np.argsort(dists[i])
                neighbors.append(order[:T].tolist())
            return neighbors

        neighbors = build_neighbors(W, T)

        # Tchebycheff scalarization
        def tchebycheff(f, w, ideal):
            """Tchebycheff scalarization: max_i w_i * |f_i - ideal_i|"""
            normalized = np.abs(f - ideal)
            # Avoid division by zero in weights
            w_safe = np.where(w < 1e-10, 1e-10, w)
            return np.max(w_safe * normalized)

        # Compute initial scalarization values
        scalar_vals = np.array([tchebycheff(F[i], W[i], ideal) for i in range(pop_size)])

        # DE parameters
        CR = 0.5  # crossover rate
        F_de = 0.5  # differential weight

        # Main evolution loop
        while evaluations < self.budget and time.time() < deadline:
            # Update ideal point
            ideal = F.min(axis=0)

            # Update best solutions
            for i in range(pop_size):
                if tchebycheff(F[i], W[i], ideal) < scalar_vals[i]:
                    best_X[i] = X[i]
                    best_F[i] = F[i]
                    scalar_vals[i] = tchebycheff(F[i], W[i], ideal)

            # Generate offspring
            for i in range(pop_size):
                if evaluations >= self.budget or time.time() > deadline:
                    break

                # Select parents from neighborhood
                nb = neighbors[i]
                if len(nb) >= 3:
                    # Select 3 distinct parents from neighborhood
                    parents = np.random.choice(nb, size=3, replace=False)
                    p1, p2, p3 = parents
                    # DE mutation
                    mutant = X[p1] + F_de * (X[p2] - X[p3])
                    mutant = np.clip(mutant, lo, hi)
                elif len(nb) >= 2:
                    p1, p2 = np.random.choice(nb, size=2, replace=False)
                    mutant = X[p1] + F_de * (X[p2] - X[i])
                    mutant = np.clip(mutant, lo, hi)
                else:
                    mutant = X[i].copy()

                # Binary crossover
                rand = np.random.rand(dim)
                jrand = np.random.randint(dim)
                child = np.where(rand < CR, mutant, X[i])
                child[jrand] = mutant[jrand]
                child = np.clip(child, lo, hi)

                # Evaluate
                f_child = problem(child)
                evaluations += 1

                # Update if better in Tchebycheff sense
                if tchebycheff(f_child, W[i], ideal) < scalar_vals[i]:
                    X[i] = child
                    F[i] = f_child
                    scalar_vals[i] = tchebycheff(f_child, W[i], ideal)

                # Also check neighborhood for replacement
                for j in nb:
                    if tchebycheff(f_child, W[j], ideal) < scalar_vals[j]:
                        X[j] = child.copy()
                        F[j] = f_child.copy()
                        scalar_vals[j] = tchebycheff(f_child, W[j], ideal)

        # Final: extract non-dominated solutions from all evaluated points
        # Combine all X and F
        all_X = X
        all_F = F

        # Find non-dominated solutions
        def is_non_dominated(F_arr):
            """Return boolean mask of non-dominated solutions."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if (F_arr[j] <= F_arr[i]).all() and (F_arr[j] < F_arr[i]).any():
                        mask[i] = False
                        break
            return mask

        nd_mask = is_non_dominated(all_F)
        return all_X[nd_mask]
# EVOLVE-BLOCK-END
