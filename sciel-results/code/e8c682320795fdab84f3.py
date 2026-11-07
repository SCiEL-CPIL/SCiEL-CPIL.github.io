import numpy as np


class Optimizer:
    """NSGA-III style decomposition-based multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        pop_size = min(30, self.budget // 6)
        pop_size = max(10, pop_size)

        # Generate reference points on simplex for 2 objectives
        # For 2 objectives, reference points are just points on the line
        n_ref = pop_size
        ref_points = np.linspace(0, 1, n_ref).reshape(-1, 1)
        ref_points = np.hstack([ref_points, 1 - ref_points])

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best known Pareto front
        best_front = F.copy()

        # Adaptive mutation parameters
        base_sigma = 0.15
        min_sigma = 0.01
        sigma = base_sigma

        # Main loop
        while used < self.budget:
            # Non-dominated sorting
            ranks = self._non_dominated_sort(F)

            # Assign each individual to nearest reference point
            # and select parents based on reference point coverage
            parents = self._select_parents(X, F, ranks, ref_points, rng)

            # Generate children
            n_children = min(pop_size, self.budget - used)
            children_X = []
            children_F = []

            for i in range(n_children):
                # Pick parent with some probability of picking from different ref points
                parent_idx = rng.integers(len(parents))
                parent = parents[parent_idx]

                # Adaptive mutation: Gaussian with decreasing sigma
                current_sigma = sigma * (0.95 ** (used / max(1, self.budget)))
                current_sigma = max(current_sigma, min_sigma)

                # Mix of large and small steps
                if rng.random() < 0.2:
                    # Large step for exploration
                    step = current_sigma * 3 * (hi - lo) * rng.standard_normal(n)
                else:
                    # Normal step
                    step = current_sigma * (hi - lo) * rng.standard_normal(n)

                child = parent + step
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

                if used >= self.budget:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            all_X = np.vstack([X, children_X])
            all_F = np.vstack([F, children_F])

            # Select next generation using non-dominated sorting + crowding
            X, F = self._select_next_gen(all_X, all_F, pop_size, ref_points, rng)

        return X

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting and return ranks."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        unassigned = set(range(n))

        while unassigned:
            front = []
            for i in unassigned:
                dominated = False
                for j in unassigned:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)

            if not front:
                # All remaining are mutually non-dominated
                front = list(unassigned)

            for i in front:
                ranks[i] = len(fronts)
            fronts.append(front)
            unassigned -= set(front)

        return ranks

    def _select_parents(self, X, F, ranks, ref_points, rng):
        """Select parents using reference point-based selection."""
        n = len(X)
        # Assign each individual to its nearest reference point
        # Compute distance in objective space (normalized)
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1
        F_norm = (F - f_min) / f_range

        # Assign to nearest reference point
        assignments = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.sum((F_norm[i] - ref_points) ** 2, axis=1)
            assignments[i] = np.argmin(dists)

        # Prefer lower ranks, then prefer individuals with good reference point coverage
        parent_indices = []
        for ref_idx in range(len(ref_points)):
            candidates = np.where(assignments == ref_idx)[0]
            if len(candidates) > 0:
                # Pick the best (lowest rank) from this reference point
                min_rank = np.min(ranks[candidates])
                best_in_ref = candidates[ranks[candidates] == min_rank]
                parent_indices.append(rng.choice(best_in_ref))

        # Fill up to pop_size
        while len(parent_indices) < len(X):
            parent_indices.append(rng.integers(len(X)))

        return X[parent_indices]

    def _select_next_gen(self, X, F, pop_size, ref_points, rng):
        """Select next generation using non-dominated sorting + reference point crowding."""
        n = len(X)
        ranks = self._non_dominated_sort(F)

        # Sort by rank, then by crowding distance
        order = np.argsort(ranks)

        # Select top pop_size by rank
        selected = []
        for i in order:
            if len(selected) >= pop_size:
                break
            selected.append(i)

        # If we need more, add from lower ranks
        if len(selected) < pop_size:
            remaining = [i for i in order if i not in selected]
            rng.shuffle(remaining)
            for i in remaining[:pop_size - len(selected)]:
                selected.append(i)

        return X[selected], F[selected]
