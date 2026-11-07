"""Decomposition-based multi-objective optimizer using Tchebycheff scalarization for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Decomposition-based multi-objective optimizer using Tchebycheff scalarization.
    Maps the multi-objective problem onto a set of single-objective subproblems
    defined by diverse weight vectors, then evolves a population where each member
    is associated with one subproblem.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Generate weight vectors for Tchebycheff scalarization
        # Use a grid of weight vectors in the simplex
        n_sub = 20  # number of subproblems
        weights = self._generate_weights(n_sub, self.n_obj)

        # Initialize population
        pop_size = n_sub
        X = lo + range_ * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()
        best_tcheb = self._compute_tchebycheff(F, weights, best_F)

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Check stagnation: if no improvement in last 30 evals, do a random restart
            # For now, just continue evolving

            # Select parents based on neighborhood
            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Choose a neighbor
                j = (i + np.random.choice([-1, 1])) % pop_size

                # Crossover: blend of parent i and j
                alpha = np.random.rand()
                child = alpha * X[i] + (1 - alpha) * X[j]

                # Mutation
                if np.random.rand() < 0.5:
                    sigma = 0.1 * np.random.rand(self.dim)
                    child = child + sigma * range_ * np.random.choice([-1, 1], self.dim)
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Evaluate child against subproblem i
                # Tchebycheff scalarization: min max_j (w_j * |f_j(x) - z_j^i|)
                # Use current best as ideal point approximation
                tcheb_val = self._tchebycheff_value(fc, weights[i], best_F)

                # Update best for subproblem i if improved
                if tcheb_val < best_tcheb[i]:
                    best_tcheb[i] = tcheb_val
                    best_X[i] = child.copy()
                    best_F[i] = fc.copy()

                # Also update the main population if child is better than X[i] in its subproblem
                tcheb_i = self._tchebycheff_value(fc, weights[i], best_F)
                if tcheb_i < self._tchebycheff_value(F[i], weights[i], best_F):
                    X[i] = child
                    F[i] = fc

                # Additionally, consider updating neighboring subproblems
                for k in [i, (i + 1) % pop_size, (i - 1) % pop_size]:
                    if k == i:
                        continue
                    tcheb_k = self._tchebycheff_value(fc, weights[k], best_F)
                    if tcheb_k < best_tcheb[k]:
                        best_tcheb[k] = tcheb_k
                        best_X[k] = child.copy()
                        best_F[k] = fc.copy()

        # Return the best non-dominated solutions found
        # Collect all best_X and find non-dominated
        all_X = best_X
        all_F = best_F

        # Find non-dominated solutions
        non_dominated_idx = self._find_non_dominated(all_F)
        result_X = all_X[non_dominated_idx]
        return result_X

    def _generate_weights(self, n_sub, n_obj):
        """Generate weight vectors on the simplex for Tchebycheff scalarization."""
        if n_obj == 1:
            return np.ones((n_sub, 1)) / n_sub

        # Use Dirichlet distribution for diverse weights
        weights = np.random.dirichlet(np.ones(n_obj), size=n_sub)
        return weights

    def _tchebycheff_value(self, f, w, ideal):
        """Compute Tchebycheff scalarization value."""
        # f: objective vector, w: weight vector, ideal: ideal point
        # T(x, w, z) = max_i (w_i * |f_i(x) - z_i|)
        return np.max(w * np.abs(f - ideal))

    def _compute_tchebycheff(self, F, weights, ideal):
        """Compute Tchebycheff values for all solutions against all weight vectors."""
        n = len(F)
        vals = np.zeros((n, len(weights)))
        for i in range(n):
            for j in range(len(weights)):
                vals[i, j] = self._tchebycheff_value(F[i], weights[j], ideal)
        return np.min(vals, axis=1)  # Best tchebycheff value per solution

    def _find_non_dominated(self, F):
        """Find indices of non-dominated solutions."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]
# EVOLVE-BLOCK-END
