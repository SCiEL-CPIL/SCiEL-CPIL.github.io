# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG4 with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondom_sort(self, F):
        """Return list of fronts (each a list of indices) and rank array."""
        n = len(F)
        if n == 0:
            return [], np.array([], dtype=int)

        S = [set() for _ in range(n)]
        ncount = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    ncount[j] += 1

        current_front = [i for i in range(n) if ncount[i] == 0]
        for i in current_front:
            rank[i] = 0

        fronts = [current_front]
        k = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    ncount[j] -= 1
                    if ncount[j] == 0:
                        rank[j] = k
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            current_front = next_front
            k += 1

        # Assign rank to any unranked (shouldn't happen but safety)
        for i in range(n):
            if rank[i] == -1:
                rank[i] = k
                if not fronts or fronts[-1] is not None:
                    if not fronts:
                        fronts.append([])
                    fronts[-1].append(i)

        return fronts, rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for individuals in indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            vals = F_sub[:, m]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            min_val = vals[order[0]]
            max_val = vals[order[-1]]
            if max_val > min_val:
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)

        return cd

    def _sbx_crossover(self, x1, x2, lo, hi):
        """Simulated binary crossover."""
        if np.random.random() < 0.5:
            return x1, x2

        eta = 20.0
        beta1 = 1.0
        beta2 = 1.0
        if np.random.random() < 0.5:
            u = np.random.random()
            if u <= 0.5:
                beta1 = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta1 = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        if np.random.random() < 0.5:
            u = np.random.random()
            if u <= 0.5:
                beta2 = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta2 = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        c1 = 0.5 * ((x1 + x2) - beta1 * (x2 - x1))
        c2 = 0.5 * ((x1 + x2) + beta2 * (x2 - x1))
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial mutation."""
        eta = 20.0
        delta = np.zeros_like(x)
        for i in range(len(x)):
            u = np.random.random()
            if u < 0.5:
                delta[i] = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta[i] = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        span = hi - lo
        x = x + delta * span * 0.5
        return np.clip(x, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size
        pop_size = min(50, max(20, self.budget // 4))

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts, rank = self._nondom_sort(F)

            # Create offspring
            offspring = []
            offspring_F = []

            # Selection via tournament + crowding
            while len(offspring) < pop_size:
                # Select parents from best fronts
                # Get crowding distances for the first front (or as many as needed)
                # For simplicity, select from the best front with crowding
                # But we need to handle multiple fronts

                # Tournament selection: pick 2 random individuals, prefer lower rank, then higher crowding
                # First compute crowding for all individuals
                # Actually, let's do a simpler approach: select from fronts in order

                # Pick a front
                # We'll do binary tournament based on (rank, crowding)
                # Compute crowding for each front separately and store

                # Simpler: just pick randomly from the best front, with some probability from next
                if len(fronts) > 0:
                    # With probability 0.8 pick from front 0, else from later fronts
                    if np.random.random() < 0.8 or len(fronts) == 1:
                        front_idx = 0
                    else:
                        front_idx = np.random.randint(0, len(fronts))

                    front = fronts[front_idx]
                    # Compute crowding for this front
                    cd = self._crowding_distance(F, front)

                    # Binary tournament within this front
                    i1 = np.random.randint(0, len(front))
                    i2 = np.random.randint(0, len(front))
                    if cd[i1] >= cd[i2]:
                        p1_idx = front[i1]
                    else:
                        p1_idx = front[i2]

                    # Second parent
                    i1 = np.random.randint(0, len(front))
                    i2 = np.random.randint(0, len(front))
                    if cd[i1] >= cd[i2]:
                        p2_idx = front[i1]
                    else:
                        p2_idx = front[i2]

                    x1 = X[p1_idx]
                    x2 = X[p2_idx]

                    # Crossover
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi)

                    # Mutation
                    c1 = self._polynomial_mutation(c1, lo, hi)
                    c2 = self._polynomial_mutation(c2, lo, hi)

                    offspring.append(c1)
                    offspring.append(c2)

                    # Evaluate
                    f1 = problem(c1)
                    f2 = problem(c2)
                    used += 2
                    offspring_F.append(f1)
                    offspring_F.append(f2)

                    if used >= self.budget:
                        break

            if len(offspring) == 0:
                break

            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring)])
            F_all = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort on combined
            fronts_all, rank_all = self._nondom_sort(F_all)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    cd = self._crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Track all evaluated
            for i in range(len(offspring)):
                all_X.append(offspring[i])
                all_F.append(offspring_F[i])

        return X
# EVOLVE-BLOCK-END
