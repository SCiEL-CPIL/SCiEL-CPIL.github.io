"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4: Pareto front is the unit simplex in objective space.
        # For 3 objectives, the front is a triangle in [0,1]^3 where f1+f2+f3=1.
        # We can sample points on this front directly and use local search to refine.

        # Strategy:
        # 1. Sample a diverse set of points on the known Pareto front (analytic).
        # 2. Use remaining budget for local search / refinement around these points.
        # 3. Track all evaluated points and return the best set.

        # Generate points on the Pareto front: f = [sin(pi*x1/2), sin(pi*x2/2), 1] scaled
        # Actually for DTLZ4 with 3 objectives, the Pareto-optimal solutions satisfy:
        # g(x) = 0 and the objective values are:
        # f1 = 0.5 * (1+g) * x1^beta * sin(pi*x1/2)
        # f2 = 0.5 * (1+g) * x1^beta * sin(pi*x2/2) * (1-x1^beta)
        # f3 = 0.5 * (1+g) * (1-x1^beta) * (1-x2^beta)
        # When g=0: f1 = x1^beta * sin(pi*x1/2), f2 = x1^beta * sin(pi*x2/2) * (1-x1^beta), f3 = (1-x1^beta)*(1-x2^beta)
        # The Pareto front in objective space is the unit simplex: f1+f2+f3 = 1, fi >= 0.

        # Let's sample uniformly on the simplex and find preimages via local search.
        # For 3 objectives, we can parameterize the front as:
        # f1 = a, f2 = b, f3 = 1-a-b, where a,b >= 0, a+b <= 1

        # A simpler approach: use a population-based evolutionary algorithm with
        # good initialization on the front and mutation for exploration.

        # Phase 1: Initialize with points sampled from the Pareto front region
        # The decision variables that give g=0 in DTLZ4:
        # x_{k+1}...x_{M} must satisfy sum((xi - 0.5)^2) = 0 for i = k+1 to M
        # So xi = 0.5 for i = k+1 to M, and x1, x2 free in [0,1].
        # With beta=10, x1^10 and x2^10 compress the range.

        # Let's use a two-phase approach:
        # Phase 1: Evaluate a set of points on the analytic front (by setting g=0)
        # Phase 2: Use remaining budget for local refinement

        # For DTLZ4 with dim=12, k=3 (since n_obj=3, k=n_obj=3):
        # g depends on x4...x12 (9 variables), set them to 0.5
        # x1, x2, x3 are the "position" variables on the front
        # Wait, let me re-check: for DTLZ4, k = n_obj = 3
        # x1, x2, x3 are position variables, x4...x12 are distance variables
        # g(x) = 100 * (n_g + sum((x_i - 0.5)^2 - sin(20*pi*(x_i-0.5))) for i in x4...x12)
        # n_g = 12 - 3 = 9

        # When g=0, we need each (x_i - 0.5)^2 - sin(20*pi*(x_i-0.5)) = 0
        # This is satisfied when x_i = 0.5 for all i in 4..12

        # So for points on the front: x4=x5=...=x12=0.5
        # f1 = 0.5*(1+0)*x1^10 * sin(pi*x1/2) = 0.5 * x1^10 * sin(pi*x1/2)
        # f2 = 0.5 * x1^10 * sin(pi*x2/2) * (1-x1^10)
        # f3 = 0.5 * (1-x1^10) * (1-x2^10)

        # Hmm, that doesn't sum to 1. Let me reconsider.

        # Actually for DTLZ4:
        # f1 = 0.5 * (1+g) * x1^beta * sin(pi*x1/2)
        # f2 = 0.5 * (1+g) * x1^beta * sin(pi*x2/2) * (1-x1^beta)  
        # f3 = 0.5 * (1+g) * (1-x1^beta) * (1-x2^beta)
        # Wait, that's not standard. Let me think again.

        # Standard DTLZ4 with M objectives:
        # f1 = 0.5*(1+g)*x1^beta*sin(pi*x1/2)
        # f2 = 0.5*(1+g)*x1^beta*sin(pi*x2/2)*(1-x1^beta)  -- no, this is wrong

        # Correct DTLZ4:
        # For M objectives, k = M position variables x1..xM
        # f_i = 0.5*(1+g) * prod_{j=1}^{i-1} x_j^beta * sin(pi*x_i/2) * (1-x_i^beta) for i=1..M-1
        # Wait no. Let me look at this more carefully.

        # DTLZ4 standard form for M=3:
        # f1 = 0.5 * (1+g) * x1^beta * sin(pi*x1/2)
        # f2 = 0.5 * (1+g) * x1^beta * sin(pi*x2/2) * (1 - x1^beta)
        # f3 = 0.5 * (1+g) * (1 - x1^beta) * (1 - x2^beta)

        # Hmm, but these don't sum to 1 in general. The Pareto front is still the simplex.

        # Let me just use a robust evolutionary approach with good initialization.

        # Use NSGA-II-like approach:
        # - Initialize with diverse points
        # - Use tournament selection, mutation, and crowding distance

        # Given the budget of 200 evaluations, let's use a small population.
        # pop_size = 20, generations = (200-20)/20 = 9 generations

        pop_size = 20
        X = np.zeros((pop_size, dim))
        # Initialize: set distance variables to 0.5, randomize position variables
        X[:, 3:] = 0.5  # x4..x12
        X[:, :3] = rng.random((pop_size, 3))

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def dominated_count(F_arr):
            """Return array of dominance counts for each point."""
            n = len(F_arr)
            counts = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        counts[i] += 1
            return counts

        def crowding_distance(F_arr):
            """Compute crowding distance for each point."""
            n = len(F_arr)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(F_arr.shape[1]):
                sorted_idx = np.argsort(F_arr[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_arr[sorted_idx[-1], m] - F_arr[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_arr[sorted_idx[i+1], m] - F_arr[sorted_idx[i-1], m]) / f_range
            return cd

        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Compute non-dominated fronts
            n = pop_size
            non_dominated = dominated_count(F) == 0
            parents_idx = np.where(non_dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(n)

            # Generate children
            n_children = min(pop_size, budget - used)
            children = np.zeros((n_children, dim))
            for i in range(n_children):
                # Tournament selection
                p1_idx = parents_idx[rng.integers(len(parents_idx))]
                p2_idx = parents_idx[rng.integers(len(parents_idx))]
                # Simple crossover
                mask = rng.random(dim) < 0.5
                child = np.where(mask, X[p1_idx], X[p2_idx])
                # Mutation
                child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                children[i] = child

            # Evaluate children
            child_F = np.array([problem(x) for x in children])
            used += n_children
            all_X.append(children.copy())
            all_F.append(child_F.copy())

            # Combine parents and children
            combined_X = np.vstack([X, children])
            combined_F = np.vstack([F, child_F])
            total = len(combined_X)

            # Non-dominated sorting
            remaining = np.arange(total)
            fronts = []
            while len(remaining) > 0:
                sub_F = combined_F[remaining]
                nd = dominated_count(sub_F) == 0
                front = remaining[nd]
                fronts.append(front)
                remaining = remaining[~nd]

            # Select next population
            X_new = np.zeros((pop_size, dim))
            F_new = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X_new[count] = combined_X[idx]
                        F_new[count] = combined_F[idx]
                        count += 1
                else:
                    # Use crowding distance to fill remaining slots
                    sub_F = combined_F[front]
                    cd = crowding_distance(sub_F)
                    sorted_front = front[np.argsort(-cd)]
                    for idx in sorted_front:
                        if count < pop_size:
                            X_new[count] = combined_X[idx]
                            F_new[count] = combined_F[idx]
                            count += 1
                if count >= pop_size:
                    break

            X = X_new
            F = F_new

        # Collect all evaluated points and return
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)
        return all_X
# EVOLVE-BLOCK-END
