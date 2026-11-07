"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE37 using:
    1. Latin Hypercube Sampling for initial coverage.
    2. Differential Evolution with structured mutation/crossover.
    3. Nelder-Mead local polish on non-dominated points to tighten Pareto front.
    4. Deadline guard to stay within 330s wall-clock.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Latin Hypercube Sampling for initialization ---
        n_init = min(40, budget // 5)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + rng.random(n_init)) / n_init
        X = lo + (hi - lo) * X

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init

        # --- Archive: store all non-dominated points ---
        archive_X = list(X)
        archive_F = list(F)

        def is_dominated(f):
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            """Add point if non-dominated; remove dominated archive points."""
            if is_dominated(f):
                return False
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue  # dominated by new point
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        # --- Differential Evolution loop ---
        pop_size = max(20, min(50, budget // 4))
        # Ensure we have a population of size pop_size
        while len(archive_X) < pop_size:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used += 1
            update_archive(x, f)

        # DE parameters
        F_mut = 0.8  # mutation scale
        CR = 0.9     # crossover rate

        def de_step():
            nonlocal used
            if len(archive_X) < 3:
                # Fallback: random perturbation
                i = rng.integers(len(archive_X))
                x = archive_X[i] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                update_archive(x, f)
                return

            # Pick 3 distinct parents
            idx = rng.choice(len(archive_X), size=3, replace=False)
            x1, x2, x3 = archive_X[idx[0]], archive_X[idx[1]], archive_X[idx[2]]
            v = x1 + F_mut * (x2 - x3)
            v = np.clip(v, lo, hi)

            # Crossover with a target
            target_idx = rng.integers(len(archive_X))
            target = archive_X[target_idx]
            jrand = rng.integers(dim)
            trial = np.where(rng.random(dim) < CR, v, target)
            trial[jrand] = v[jrand]
            trial = np.clip(trial, lo, hi)

            f_trial = problem(trial)
            used += 1
            update_archive(trial, f_trial)

        # Run DE steps
        while used < budget:
            if time.time() > deadline:
                break
            de_step()

        # --- Nelder-Mead polish on non-dominated archive points ---
        # Polish the best (lowest sum) non-dominated points to tighten the front
        if len(archive_F) > 0:
            sums = np.array([np.sum(f) for f in archive_F])
            # Sort by sum of objectives, polish top candidates
            order = np.argsort(sums)
            n_polish = min(10, len(archive_F))
            polished = set()
            for idx in order[:n_polish]:
                if time.time() > deadline:
                    break
                x0 = np.array(archive_F and archive_X[idx])
                if idx in polished:
                    continue
                polished.add(idx)

                # Objective: minimize weighted sum (equal weights)
                def obj(x):
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    return np.sum(f)

                try:
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                    x_best = np.clip(res.x, lo, hi)
                    f_best = problem(x_best)
                    used += 1
                    update_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X
# EVOLVE-BLOCK-END
