"""Multi-objective optimizer on WFG7 using decomposition-based evolutionary search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Weight-based decomposition (MOEA/D-style) multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 40

        # Generate weight vectors (simplex-lattice)
        weights = self._generate_weights(pop_size, n_obj)

        # Initial population: WFG7-aware initialization
        # WFG7 has shape variables x_1..x_5 and position variables x_6..x_12
        # Known Pareto front: shape variables should be near 0.5
        X = np.empty((pop_size, dim))
        for i in range(pop_size):
            x = np.empty(dim)
            # Shape variables (first 5-6 dims) biased toward 0.5
            shape_dims = min(6, dim)
            x[:shape_dims] = 0.5 + 0.3 * np.random.randn(shape_dims)
            x[:shape_dims] = np.clip(x[:shape_dims], lo[:shape_dims], hi[:shape_dims])
            # Position variables random
            pos_dims = dim - shape_dims
            if pos_dims > 0:
                x[shape_dims:] = lo[shape_dims:] + (hi[shape_dims:] - lo[shape_dims:]) * np.random.rand(pos_dims)
            X[i] = x

        # Evaluate initial population
        F = np.array([problem(X[i]) for i in range(pop_size)])
        used = pop_size

        # Assign each individual to the nearest weight
        assignments = self._assign_to_weights(X, F, weights)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Adaptive mutation rate
            progress = used / budget
            mut_rate = 0.3 * (1.0 - 0.5 * progress)  # Start at 0.3, decay to 0.15

            # Generate children using decomposition
            for i in range(min(pop_size, budget - used)):
                if time.time() > deadline:
                    break

                w = weights[i]
                # Pick a parent: the one assigned to this weight or a neighbor
                parent_idx = assignments[i]

                # Mutation
                child = X[parent_idx].copy()
                # Gaussian mutation with adaptive step
                step = mut_rate * span
                child += step * np.random.randn(dim)
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1

                # Update: replace if it improves the weighted scalarized objective
                # or if it's non-dominated in the local neighborhood
                current_scalar = np.dot(F[i], w)
                new_scalar = np.dot(fc, w)

                # Check if new point is better in scalarized sense
                if new_scalar < current_scalar:
                    X[i] = child
                    F[i] = fc
                else:
                    # Try to maintain diversity: replace worst in neighborhood
                    # Find the worst individual among those assigned to nearby weights
                    neighborhood = [j for j in range(pop_size) if np.dot(weights[j], w) > 0.5]
                    if neighborhood:
                        worst_in_nb = max(neighborhood, key=lambda j: np.dot(F[j], w))
                        if new_scalar < np.dot(F[worst_in_nb], w):
                            X[worst_in_nb] = child
                            F[worst_in_nb] = fc

        return X

    def _generate_weights(self, pop_size, n_obj):
        """Generate weight vectors on a simplex lattice."""
        if n_obj == 3:
            weights = []
            # Use a grid approach for 3 objectives
            n_per_dim = max(1, int(round((pop_size / 6) ** 0.5)))
            for i in range(n_per_dim + 1):
                for j in range(n_per_dim + 1 - i):
                    k = n_per_dim - i - j
                    w = np.array([i, j, k], dtype=float)
                    w = w / (w.sum() + 1e-10)
                    weights.append(w)
            # Ensure we have exactly pop_size weights
            if len(weights) < pop_size:
                while len(weights) < pop_size:
                    w = np.random.dirichlet(np.ones(n_obj))
                    weights.append(w)
            else:
                weights = weights[:pop_size]
            return np.array(weights)
        else:
            return np.random.dirichlet(np.ones(n_obj), size=pop_size)

    def _assign_to_weights(self, X, F, weights):
        """Assign each individual to the nearest weight based on objective space."""
        assignments = []
        for i in range(len(X)):
            f = F[i]
            # Find weight that minimizes the angle/distance
            best_w = 0
            best_dist = float('inf')
            for j, w in enumerate(weights):
                # Use dot product as similarity (higher = better alignment)
                # But we want the weight that best "covers" this point
                dist = np.linalg.norm(f * w)
                if dist < best_dist:
                    best_dist = dist
                    best_w = j
            assignments.append(best_w)
        return assignments
# EVOLVE-BLOCK-END
