"""MOEA/D with Tchebycheff decomposition and neighborhood mating for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 295.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 80

        # Generate weight vectors using the simplex-lattice method
        weights = self._generate_weights(pop_size, n_obj)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        evals = 0
        all_F_list = []

        for i in range(pop_size):
            F[i] = problem(X[i])
            evals += 1
            all_F_list.append(F[i].copy())

        # Compute ideal point (minimum of each objective)
        ideal = np.min(F, axis=0)

        # Build neighborhood structure
        neighbor_list = self._build_neighbors(weights, pop_size, n_neighbors=20)

        # Main MOEA/D loop
        while evals < budget:
            if time.perf_counter() > deadline - 5:
                break

            # Update ideal point
            ideal = np.min(F, axis=0)

            # Generate offspring
            new_X = np.empty((pop_size, dim))
            new_F = np.empty((pop_size, n_obj))
            n_new = 0

            for i in range(pop_size):
                if evals >= budget or time.perf_counter() > deadline - 5:
                    break

                # Select parents from neighborhood
                neighbors = neighbor_list[i]
                # Use a probability to select from neighborhood vs. entire population
                if rng.random() < 0.9:
                    p1_idx = neighbors[rng.integers(len(neighbors))]
                    p2_idx = neighbors[rng.integers(len(neighbors))]
                else:
                    p1_idx = rng.integers(pop_size)
                    p2_idx = rng.integers(pop_size)

                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # Differential Evolution style crossover
                child = p1.copy()
                for j in range(dim):
                    if rng.random() < 0.5:
                        child[j] = p1[j] + 0.5 * (p2[j] - p1[j])
                    else:
                        child[j] = p1[j]

                # Polynomial mutation
                mutation_rate = 0.1
                eta_mut = 20.0
                for j in range(dim):
                    if rng.random() < mutation_rate:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta_mut + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_mut + 1))
                        child[j] += delta * span[j]

                # Clip to bounds
                child = np.clip(child, lo, hi)

                new_X[n_new] = child
                new_F[n_new] = problem(child)
                evals += 1
                all_F_list.append(new_F[n_new].copy())
                n_new += 1

            if n_new == 0:
                break

            # Combine and select
            # For each subproblem i, update with the best child according to Tchebycheff
            # We need to match children to subproblems. In MOEA/D, each subproblem gets one child.
            # But we generated pop_size children. Let's assign child i to subproblem i.
            # Actually, the standard MOEA/D generates one child per subproblem.
            # Let's restructure: generate children for each subproblem.

            # Let's redo the offspring generation to be per-subproblem
            # But we already did that above. Let's just use the first pop_size children.
            # Actually, we need to be careful. The loop above generates up to pop_size children.
            # Let's assume n_new == pop_size for simplicity, or handle partial.

            if n_new < pop_size:
                # Pad with copies of existing solutions if needed
                for k in range(n_new, pop_size):
                    new_X[k] = X[k]
                    new_F[k] = F[k]
                n_new = pop_size

            # Selection: for each subproblem i, keep the better of (X[i], new_X[i])
            # according to the Tchebycheff scalarizing function with weight w_i
            for i in range(pop_size):
                w = weights[i]
                # Normalize F using ideal point
                # Avoid division by zero
                norm_F_old = (F[i] - ideal) / (1e-10 + np.ones(n_obj))
                norm_F_new = (new_F[i] - ideal) / (1e-10 + np.ones(n_obj))

                # Tchebycheff: max_j w_j * |f_j - z_j^*|
                # Since we're minimizing, and ideal is the min, we use (f_j - ideal_j)
                t_old = np.max(w * np.abs(norm_F_old))
                t_new = np.max(w * np.abs(norm_F_new))

                if t_new < t_old:
                    X[i] = new_X[i]
                    F[i] = new_F[i]
                # else keep old

        return X

    def _generate_weights(self, n, n_obj):
        """Generate weight vectors using simplex-lattice method."""
        if n_obj == 1:
            return np.ones((n, 1)) / 1.0

        # For 3 objectives, we can use a 2D simplex
        # Generate points on the 2-simplex: w1 + w2 + w3 = 1, w_i >= 0
        # Use a grid approach
        weights = []
        # Number of divisions
        # We need approximately n points
        # For a 2-simplex, the number of points with H divisions is (H+1)(H+2)/2
        # Solve for H: H^2 + 3H + 2 ~ 2n => H ~ sqrt(2n)
        H = int(np.sqrt(2 * n))
        while H > 0:
            count = (H + 1) * (H + 2) // 2
            if count <= n:
                break
            H -= 1

        for w1 in range(H + 1):
            for w2 in range(H + 1 - w1):
                w3 = H - w1 - w2
                weights.append([w1 / H, w2 / H, w3 / H])

        weights = np.array(weights)

        # If we have more than n, randomly subsample
        if len(weights) > n:
            idx = rng.choice(len(weights), n, replace=False)
            weights = weights[idx]
        # If we have fewer, pad with random weights
        elif len(weights) < n:
            extra = np.random.dirichlet(np.ones(n_obj), size=n - len(weights))
            weights = np.vstack([weights, extra])

        return weights

    def _build_neighbors(self, weights, n, n_neighbors=20):
        """Build neighborhood structure based on weight vector similarity."""
        # Compute pairwise distances
        diff = weights[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        neighbor_list = []
        for i in range(n):
            # Sort by distance, exclude self
            sorted_idx = np.argsort(dist[i])
            sorted_idx = sorted_idx[1:]  # exclude self
            neighbor_list.append(sorted_idx[:n_neighbors])

        return neighbor_list
# EVOLVE-BLOCK-END
