import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        # Population size: N=15 gives ~13 generations with budget=200
        pop_size = min(15, budget)

        if pop_size < 2:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # NSGA-II Loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            ranks, fronts = self.non_dominated_sort(F)

            # Crowding distance
            cd = self.crowding_distance(F, fronts)

            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                # Tournament selection for two parents
                p1_idx = self.tournament_select(ranks, cd, 3)
                p2_idx = self.tournament_select(ranks, cd, 3)

                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self.sbx_crossover(x1, x2, lo, hi, 0.8, 20)
                c1 = self.polynomial_mutation(c1, lo, hi, 2.0/dim, 20)
                c2 = self.polynomial_mutation(c2, lo, hi, 2.0/dim, 20)

                offspring_X[i*2] = c1
                offspring_X[i*2+1] = c2

            # If pop_size is odd, handle the last one
            if pop_size % 2 == 1:
                p1_idx = self.tournament_select(ranks, cd, 3)
                x1 = X[p1_idx]
                c1 = x1.copy()
                c1 = self.polynomial_mutation(c1, lo, hi, 2.0/dim, 20)
                offspring_X[-1] = c1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort on combined population
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            cd_all = self.crowding_distance(F_all, fronts_all)

            # Select next population
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

            # Update archive: keep only non-dominated points from combined set
            combined_X = np.vstack([archive_X, X])
            combined_F = np.vstack([archive_F, F])
            # Find non-dominated points in the combined archive
            n_combined = len(combined_F)
            dominated = np.zeros(n_combined, dtype=bool)
            for i in range(n_combined):
                if dominated[i]:
                    continue
                for j in range(i + 1, n_combined):
                    if dominated[j]:
                        continue
                    if self.dominates(combined_F[i], combined_F[j]):
                        dominated[j] = True
                    elif self.dominates(combined_F[j], combined_F[i]):
                        dominated[i] = True
                        break
            mask = ~dominated
            archive_X = combined_X[mask]
            archive_F = combined_F[mask]

        return X

    def _dominates(self, a, b):
        """Returns True if a dominates b."""
        return np.all(a <= b) and np.any(a < b)

    def _select_diverse_front(self, F, max_candidates):
        """Select diverse non-dominated candidates."""
        n = F.shape[0]
        if n <= max_candidates:
            return list(range(n))

        # Find non-dominated points
        non_dom = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j and self._dominates(F[j], F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom.append(i)

        if len(non_dom) <= max_candidates:
            return non_dom

        # Select diverse subset using greedy farthest-point sampling in objective space
        f_min = F[non_dom].min(axis=0)
        f_max = F[non_dom].max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-14] = 1.0
        F_norm = (F[non_dom] - f_min) / f_range

        selected = [non_dom[0]]
        min_dists = np.full(len(non_dom), np.inf)
        min_dists[0] = 0

        for _ in range(max_candidates - 1):
            idx = np.argmax(min_dists)
            if min_dists[idx] <= 1e-10:
                break
            selected.append(non_dom[idx])

            for j in range(len(non_dom)):
                dist = np.linalg.norm(F_norm[j] - F_norm[idx])
                if dist < min_dists[j]:
                    min_dists[j] = dist

        return selected
