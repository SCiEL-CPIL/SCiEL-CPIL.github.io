# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using weighted-sum scalarization with multi-start local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        def sample():
            return lo + span * np.random.random(dim)

        def sample_lhs(n):
            X = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.permutation(n)
                X[:, d] = (perm + np.random.random(n)) / n
            return lo + span * X

        # Phase 1: LHS initialization for diverse coverage
        n_init = min(40, self.budget)
        X_init = sample_lhs(n_init)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        all_X = list(X_init)
        all_F = list(F_init)

        # Phase 2: Weighted-sum scalarization with local refinement
        # Generate diverse weight vectors on the simplex
        n_weights = 20
        # Use Dirichlet distribution for diverse weights
        weights = np.random.dirichlet(np.ones(n_obj), size=n_weights)
        # Add some extreme weights (near axes)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 0.9
            w[(i + 1) % n_obj] = 0.1
            weights = np.vstack([weights, w])

        # For each weight vector, do a local search
        n_local_starts = 3  # multi-start per weight
        local_step_size = 0.15
        local_iters = 8

        for w in weights:
            if used >= self.budget or time.time() > deadline:
                break

            for s in range(n_local_starts):
                if used >= self.budget or time.time() > deadline:
                    break

                # Start from a random point or from best known for this weight
                if s == 0:
                    x_best = sample()
                else:
                    # Perturb a random existing point
                    idx = np.random.randint(len(all_X))
                    x_best = all_X[idx] + local_step_size * (np.random.random(dim) * 2 - 1)
                    x_best = np.clip(x_best, lo, hi)

                f_best = problem(x_best)
                used += 1
                all_X.append(x_best)
                all_F.append(f_best)

                # Local search: gradient-free coordinate descent with random steps
                for it in range(local_iters):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Try a random perturbation
                    x_trial = x_best + local_step_size * (np.random.random(dim) * 2 - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = problem(x_trial)
                    used += 1
                    all_X.append(x_trial)
                    all_F.append(f_trial)

                    # Accept if better in weighted sum
                    if np.dot(w, f_trial) < np.dot(w, f_best):
                        x_best = x_trial
                        f_best = f_trial

        # Phase 3: Use remaining budget for random sampling to fill gaps
        while used < self.budget:
            if time.time() > deadline:
                break
            x = sample()
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1

        return np.array(all_X) if all_X else X_init
# EVOLVE-BLOCK-END
