"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOEA/D on ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        pop_size = 20
        remaining = self.budget - pop_size
        n_gen = max(1, remaining // pop_size)

        # Weight vectors for 2 objectives: (w, 1-w)
        weights = np.linspace(0.0, 1.0, pop_size)
        W = np.column_stack([weights, 1.0 - weights])

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Ideal point
        ideal = F.min(axis=0)

        # Neighborhood: fixed window in weight space
        neigh_size = 5
        neighbors = []
        for i in range(pop_size):
            dists = np.abs(W[:, 0] - W[i, 0])
            idx = np.argsort(dists)[:neigh_size]
            neighbors.append(idx.tolist())

        for gen in range(n_gen):
            # Generate offspring: one per sub-problem
            offspring_X = []
            offspring_F = []
            offspring_parent = []

            for i in range(pop_size):
                if used >= self.budget:
                    break

                # Select parents from neighborhood of i
                if len(neighbors[i]) >= 2:
                    p1_idx, p2_idx = rng.choice(neighbors[i], size=2, replace=False)
                else:
                    p1_idx = rng.integers(0, pop_size)
                    p2_idx = rng.integers(0, pop_size)

                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # SBX crossover
                child = p1.copy()
                if rng.random() < 0.9:
                    eta = 20.0
                    for d in range(self.dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])

                # Polynomial mutation
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        eta_m = 20.0
                        u = rng.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child[d] += delta_q * span[d]

                # Large jump for exploration (ZDT4 has many local optima)
                if rng.random() < 0.15:
                    n_jumps = rng.integers(1, 4)
                    jump_dims = rng.choice(self.dim, size=n_jumps, replace=False)
                    for jd in jump_dims:
                        child[jd] = lo[jd] + rng.random() * span[jd]

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                offspring_parent.append(i)

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            offspring_parent = np.array(offspring_parent)

            # Update ideal
            all_F = np.vstack([F, offspring_F])
            ideal = all_F.min(axis=0)

            # Selection: for each sub-problem i, pick best offspring from neighborhood
            new_X = []
            new_F = []

            for i in range(pop_size):
                cand_mask = np.isin(offspring_parent, neighbors[i])
                if not np.any(cand_mask):
                    cand_mask = np.ones(len(offspring_F), dtype=bool)

                cand_idx = np.where(cand_mask)[0]
                best_score = np.inf
                best_local = -1

                for j in cand_idx:
                    diff = np.abs(offspring_F[j] - ideal)
                    tcheb = np.max(W[i] * diff)
                    if tcheb < best_score:
                        best_score = tcheb
                        best_local = j

                if best_local >= 0:
                    new_X.append(offspring_X[best_local])
                    new_F.append(offspring_F[best_local])

            if len(new_X) == 0:
                break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
