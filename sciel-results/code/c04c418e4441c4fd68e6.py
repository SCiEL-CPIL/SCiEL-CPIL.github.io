"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """MOEA/D-style decomposition-based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Phase 1: Sobol sequence for initial coverage
        sobol_size = min(100, self.budget // 3)
        sobol_n = 1
        while sobol_n < sobol_size:
            sobol_n *= 2
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        sobol_samples = sampler.random(n=sobol_n)
        X_sobol = lo + (hi - lo) * sobol_samples[:sobol_size]
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = sobol_size

        # Build initial archive
        archive_idx = self._nondominated_indices(F_sobol)
        archive_X = X_sobol[archive_idx].copy()
        archive_F = F_sobol[archive_idx].copy()

        # Phase 2: MOEA/D with weighted-sum decomposition
        # Generate weight vectors on simplex
        M = 30  # number of subproblems
        weights = self._generate_weights(M, self.n_obj)

        # Initialize population: assign best solution to each weight
        pop_size = M
        X_pop = np.zeros((pop_size, self.dim))
        F_pop = np.zeros((pop_size, self.n_obj))

        if len(archive_X) > 0:
            for i in range(pop_size):
                # Find archive point that best matches this weight direction
                scores = weights[i] @ archive_F.T
                best_j = np.argmin(scores)
                X_pop[i] = archive_X[best_j]
                F_pop[i] = archive_F[best_j]
        else:
            X_pop = lo + (hi - lo) * rng.random((pop_size, self.dim))
            F_pop = np.array([problem(x) for x in X_pop])
            used += pop_size

        # MOEA/D parameters
        CR = 0.5
        F_scale = 0.5
        neighborhood_size = 5  # number of neighbors to use

        # Precompute neighborhoods based on weight similarity
        weight_dists = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                weight_dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        neighborhoods = np.zeros((M, neighborhood_size), dtype=int)
        for i in range(M):
            order = np.argsort(weight_dists[i])
            neighborhoods[i] = order[1:neighborhood_size + 1]  # exclude self

        # Main MOEA/D loop
        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Select a subproblem to evolve
            i = rng.integers(0, M)
            neighbors = neighborhoods[i]

            # Select parents from neighborhood (with some probability use global best)
            if len(neighbors) >= 2 and rng.random() < 0.7:
                p1_idx = neighbors[rng.integers(0, len(neighbors))]
                p2_idx = neighbors[rng.integers(0, len(neighbors))]
                if p1_idx == p2_idx:
                    p2_idx = (p2_idx + 1) % M
            else:
                p1_idx = rng.integers(0, M)
                p2_idx = rng.integers(0, M)
                if p1_idx == p2_idx:
                    p2_idx = (p2_idx + 1) % M

            parent1 = X_pop[p1_idx]
            parent2 = X_pop[p2_idx]

            # DE mutation
            v = parent1 + F_scale * (parent2 - parent1)
            j_rand = rng.integers(0, self.dim)
            mask = rng.random(self.dim) < CR
            mask[j_rand] = True
            child = np.where(mask, v, parent1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            if len(archive_F) > 0:
                child_dominated = np.any(
                    (archive_F <= fc).all(axis=1) & (archive_F < fc).any(axis=1)
                )
            else:
                child_dominated = False

            if not child_dominated:
                if len(archive_F) > 0:
                    dominated = (
                        (fc <= archive_F).all(axis=1) & (fc < archive_F).any(axis=1)
                    )
                    if dominated.any():
                        keep_mask = ~dominated
                        archive_X = archive_X[keep_mask]
                        archive_F = archive_F[keep_mask]

                archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

                if len(archive_X) > 200:
                    archive_X, archive_F = self._reduce_archive(archive_X, archive_F)

            # Update population using decomposition (weighted sum)
            # Compare child with subproblem i and its neighbors
            w = weights[i]
            child_score = w @ fc
            i_score = w @ F_pop[i]

            if child_score <= i_score:
                X_pop[i] = child
                F_pop[i] = fc
            else:
                # Try to replace a neighbor if child is better for that neighbor's weight
                for j in neighbors:
                    wj = weights[j]
                    child_score_j = wj @ fc
                    j_score = wj @ F_pop[j]
                    if child_score_j < j_score:
                        X_pop[j] = child
                        F_pop[j] = fc
                        break

        return X_pop

    def _generate_weights(self, M, n_obj):
        """Generate weight vectors uniformly distributed on the simplex."""
        if n_obj == 2:
            # Uniform on line segment
            weights = np.zeros((M, 2))
            for i in range(M):
                weights[i, 0] = i / (M - 1)
                weights[i, 1] = 1 - weights[i, 0]
            return weights
        elif n_obj == 3:
            # Uniform on 2-simplex using stratified sampling
            weights = np.zeros((M, 3))
            # Use a grid approach for 3 objectives
            n_per_axis = int(np.ceil(np.sqrt(M)))
            count = 0
            for i in range(n_per_axis):
                for j in range(n_per_axis):
                    if count >= M:
                        break
                    a = i / (n_per_axis - 1) if n_per_axis > 1 else 0
                    b = j / (n_per_axis - 1) if n_per_axis > 1 else 0
                    if a + b <= 1.0:
                        weights[count] = [a, b, 1 - a - b]
                        count += 1
            # Fill remaining if needed
            if count < M:
                for k in range(count, M):
                    weights[k] = [1.0/(n_obj+1)] * n_obj
                    weights[k] /= weights[k].sum()
            return weights[:M]
        else:
            # General case: Dirichlet sampling
            weights = rng = np.random.default_rng(42)
            weights = rng.dirichlet(np.ones(n_obj), size=M)
            return weights

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F (vectorized)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        leq = (F[:, np.newaxis, :] <= F[np.newaxis, :, :]).all(axis=2)
        lt = (F[:, np.newaxis, :] < F[np.newaxis, :, :]).any(axis=2)
        dom_matrix = leq & lt
        np.fill_diagonal(dom_matrix, False)
        dominated = dom_matrix.any(axis=1)
        return np.where(~dominated)[0]

    def _reduce_archive(self, X, F, max_size=200):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F

        n = len(F)
        crowding = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                denom = F[order[i+1], m] - F[order[i-1], m]
                if denom > 1e-10:
                    crowding[order[i]] += 1.0 / denom

        keep = np.argsort(crowding)[-max_size:]
        keep = np.sort(keep)
        return X[keep], F[keep]


# EVOLVE-BLOCK-END
