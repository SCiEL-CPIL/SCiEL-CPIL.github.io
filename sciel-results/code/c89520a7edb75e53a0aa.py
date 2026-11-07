"""Multi-objective optimizer on RE34: Decomposition-based MOEA/D with adaptive weight vectors."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """MOEA/D optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X = []
        F = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # Generate weight vectors for MOEA/D
        # For 3 objectives, use a grid of weight vectors on the simplex
        # We want about 100 subproblems
        # Use a systematic grid
        H = 8  # divisions
        weights = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                if np.sum(w) > 1e-10:
                    weights.append(w)

        # Filter out duplicate weights (shouldn't happen with this grid, but just in case)
        unique_weights = []
        for w in weights:
            if not any(np.allclose(w, uw, atol=1e-10) for uw in unique_weights):
                unique_weights.append(w)

        weights = np.array(unique_weights)
        N = len(weights)  # Number of subproblems

        # Initialize population with random points
        rng = np.random.default_rng(42)
        for _ in range(N):
            x = rng.uniform(lo, hi)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        # Helper: Compute non-dominated front
        def get_nondominated_indices(F):
            n = len(F)
            nondom = set()
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom.add(i)
            return list(nondom)

        # Helper: Update best solution for each subproblem using Tchebycheff
        def tchebycheff(f, w, f_min):
            return np.max(w * np.abs(f - f_min))

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()
        best_dist = np.zeros(N)
        f_min = np.min(F, axis=0)

        for i in range(N):
            best_dist[i] = tchebycheff(F[i], weights[i], f_min)

        # Helper: Find neighbors (within distance 2 in the weight space)
        # For simplicity, use a fixed neighborhood size
        T = 20  # Neighborhood size

        def get_neighbors(i, weights, T):
            dists = np.zeros(len(weights))
            for j in range(len(weights)):
                dists[j] = np.linalg.norm(weights[i] - weights[j])
            order = np.argsort(dists)
            return list(order[:T])

        # Precompute neighbors
        neighbors = [get_neighbors(i, weights, T) for i in range(N)]

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select a subproblem to evolve
            # Use a probability based on how much it has improved recently
            # For simplicity, random selection with some bias towards recently improved
            i = rng.integers(N)

            # Select a parent from the neighborhood
            neigh = neighbors[i]
            # Tournament selection: pick 2 random neighbors, choose the one with better Tchebycheff distance
            p1, p2 = rng.choice(neigh, size=2, replace=False)
            d1 = tchebycheff(F[p1], weights[i], f_min)
            d2 = tchebycheff(F[p2], weights[i], f_min)
            if d1 < d2:
                parent = p1
            else:
                parent = p2

            # Crossover and mutation
            x_parent = X[parent]
            x_child = x_parent.copy()

            # Gaussian mutation
            sigma = 0.1
            x_child += rng.normal(0, sigma, dim)
            x_child = np.clip(x_child, lo, hi)

            x_child, f_child = evaluate(x_child)

            # Update the best solution for subproblem i
            d_child = tchebycheff(f_child, weights[i], f_min)
            if d_child < best_dist[i]:
                best_X[i] = x_child
                best_F[i] = f_child
                best_dist[i] = d_child

            # Also check if the child is better than the current solution in the population
            # for other subproblems in the neighborhood
            for j in neigh:
                d_j = tchebycheff(f_child, weights[j], f_min)
                if d_j < best_dist[j]:
                    best_X[j] = x_child
                    best_F[j] = f_child
                    best_dist[j] = d_j

            # Update population: replace the worst solution in the neighborhood
            # Find the worst neighbor
            worst = max(neigh, key=lambda k: tchebycheff(F[k], weights[i], f_min))
            if tchebycheff(f_child, weights[i], f_min) < tchebycheff(F[worst], weights[i], f_min):
                X[worst] = x_child
                F[worst] = f_child

            # Update f_min
            f_min = np.min(F, axis=0)

            # Every 50 evaluations, do a local search on the best non-dominated points
            if used % 50 == 0 and time.time() < deadline - 30:
                nondom = get_nondominated_indices(F)
                # Pick a few non-dominated points and do local search
                for idx in nondom[:3]:
                    if time.time() > deadline - 10:
                        break
                    x0 = X[idx]
                    f0 = F[idx]

                    def objective(x):
                        x = np.clip(x, lo, hi)
                        f = np.asarray(problem(x), dtype=np.float64)
                        return np.sum(f)  # Simple sum for local search

                    try:
                        res = minimize(objective, x0, method='L-BFGS-B',
                                       bounds=list(zip(lo, hi)), options={'maxiter': 100})
                        if res.success:
                            x_new = res.x
                            f_new = np.asarray(problem(x_new), dtype=np.float64)
                            used += 1
                            # Check if it's better
                            if not any(np.all(F[k] <= f_new) and np.any(F[k] < f_new) for k in range(len(F))):
                                # It's non-dominated, replace the worst
                                worst = np.argmax(np.sum(F, axis=1))
                                X[worst] = x_new
                                F[worst] = f_new
                                f_min = np.min(F, axis=0)
                    except:
                        pass

        return X
# EVOLVE-BLOCK-END
