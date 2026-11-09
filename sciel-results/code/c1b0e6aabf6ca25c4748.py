"""Rewritten program: L-SHADE global search + Nelder-Mead local refinement, with an
explicit wall-clock deadline guard. Minimizes f16 (Weierstrass) on [-5, 5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Portfolio: L-SHADE global search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        DEADLINE = 300.0  # safety margin well under the 330 s hard kill

        # ---------- time budgeting ----------
        # Reserve a guaranteed local-refinement window near the end.
        local_time_budget = 25.0
        global_deadline = t0 + (DEADLINE - local_time_budget)
        # Final polish pass gets the last sliver of time.
        polish_deadline = t0 + DEADLINE - 0.5

        def now():
            return time.perf_counter()

        def time_ok(deadline):
            return (now() - t0) < deadline

        rng = np.random.default_rng(12345)

        # ---------- L-SHADE ----------
        # Population size (small for 5-D and 500 budget, but enough for diversity).
        NP = 30
        F_min, F_max = 0.4, 1.2
        CR_min, CR_max = 0.0, 1.0
        K = 5  # archive size
        K0 = 10  # initial archive size

        # Initial population
        pop = rng.uniform(-5, 5, size=(NP, dim))
        fitness = np.empty(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        # Archive
        archive = pop[:K0].copy()
        archive_size = K0
        best_x = pop[0].copy()
        best_f = float(fitness[0])

        # L-SHADE parameters
        n_gen = 0
        max_gen = budget  # hard cap on generations (each gen = NP evals)
        # We will be limited by the budget and the deadline, whichever comes first.

        # Precompute the "random" CR and F distributions are handled inline.
        # Track best fitness for stall detection.
        best_f_history = [best_f]

        while time_ok(global_deadline):
            n_gen += 1
            if n_gen > max_gen:
                break

            # L-SHADE adaptation
            H = min(archive_size, max(K, K0 - (K0 - K) * n_gen // max(1, max_gen)))
            # Random CR
            CR = CR_min + (CR_max - CR_min) * rng.random()
            # Random F
            F = F_min + (F_max - F_min) * rng.random()
            # Update archive
            if archive_size < K0:
                # grow archive
                idx = rng.integers(0, NP)
                archive = np.vstack([archive, pop[idx][None, :]])
                archive_size += 1
            else:
                # replace worst in archive with best of population
                worst_in_archive = archive[np.argmax(np.array([func(a) for a in archive]))]
                # Actually recompute archive fitness is expensive; instead use a simpler
                # strategy: keep the K best of (current archive union current best pop).
                # To keep it cheap, we just insert the current best and drop the oldest.
                archive = np.vstack([archive, best_x[None, :]])
                archive = archive[-K:]
                archive_size = K

            # Generate trial vectors for each member
            new_pop = np.empty_like(pop)
            new_fit = np.empty(NP)
            for i in range(NP):
                # pick two distinct random members
                r1, r2 = rng.integers(0, NP, size=2)
                while r2 == r1:
                    r2 = rng.integers(0, NP)
                # pick a random archive member
                ra = rng.integers(0, archive_size)
                # mutation: L-SHADE uses (best + F*(r1 - r2)) or (r1 + F*(r2 - r3))
                # We use a mix: with 50% use best-based, else random-based.
                if rng.random() < 0.5:
                    v = best_x + F * (pop[r1] - pop[r2])
                else:
                    v = pop[r1] + F * (pop[r2] - archive[ra])
                # crossover
                j_rand = rng.integers(0, dim)
                crosser = rng.random(dim)
                trial = np.where(crosser <= CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                trial = np.clip(trial, -5, 5)
                new_pop[i] = trial
                new_fit[i] = func(trial)

            # selection
            for i in range(NP):
                if new_fit[i] <= fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fit[i]

            # re-sort
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]
            if fitness[0] < best_f:
                best_f = float(fitness[0])
                best_x = pop[0].copy()
            best_f_history.append(best_f)

            # Stall detection: if no improvement in last ~15 generations, restart around best.
            if n_gen > 15:
                recent = best_f_history[-16:-1]
                if best_f_history[-1] >= min(recent) - 1e-12:
                    # no improvement; perturb population around best to escape
                    for i in range(NP):
                        if i == 0:
                            continue
                        pop[i] = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                    # re-evaluate
                    for i in range(NP):
                        fitness[i] = func(pop[i])
                    order = np.argsort(fitness)
                    pop = pop[order]
                    fitness = fitness[order]
                    if fitness[0] < best_f:
                        best_f = float(fitness[0])
                        best_x = pop[0].copy()
                    best_f_history.append(best_f)

        # ---------- Local refinement (Nelder-Mead) ----------
        # Run a fast, low-function-cost local refinement from the best point,
        # and from a few nearby points, within the reserved time window.
        local_deadline = t0 + DEADLINE - 1.0  # leave a tiny margin

        def nelder_mead(func, x0, max_evals=40, tol=1e-9):
            dim = len(x0)
            # initial simplex
            simplex = [x0.copy() for _ in range(dim + 1)]
            for i in range(dim):
                step = 0.05 if abs(x0[i]) < 1e-8 else 0.05 * abs(x0[i])
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step
            fvals = np.array([func(s) for s in simplex])
            for _ in range(max_evals):
                # sort
                order = np.argsort(fvals)
                simplex = [simplex[j] for j in order]
                fvals = fvals[order]
                if fvals[0] < best_f:
                    best_f = float(fvals[0])
                    best_x = simplex[0].copy()
                # convergence
                if abs(fvals[-1] - fvals[0]) < tol:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue
                # expansion
                gamma = 2.0
                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue
                # contraction
                rho = 0.5
                if fr < fvals[-1]:
                    # outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                else:
                    # inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                fc = func(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                    continue
                # shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    fvals[i] = func(simplex[i])
            # final best
            idx = int(np.argmin(fvals))
            if fvals[idx] < best_f:
                best_f = float(fvals[idx])
                best_x = simplex[idx].copy()
            return best_x, best_f

        # Run local refinement from the best point first.
        if time_ok(local_deadline):
            best_x, best_f = nelder_mead(func, best_x, max_evals=60)

        # Then from a few jittered neighbours, if time remains.
        for _ in range(4):
            if not time_ok(local_deadline):
                break
            cand = np.clip(best_x + rng.standard_normal(dim) * 0.1, -5, 5)
            cand_f = func(cand)
            if cand_f < best_f:
                best_x = cand
                best_f = cand_f
            if time_ok(local_deadline):
                best_x, best_f = nelder_mead(func, best_x, max_evals=30)

        # ---------- Final polish (tiny steps) ----------
        # A few very small coordinate perturbations to squeeze out last bits.
        while time_ok(polish_deadline):
            improved = False
            for d in range(dim):
                for sign in (-1.0, 1.0):
                    step = 1e-3
                    cand = best_x.copy()
                    cand[d] += sign * step
                    cand = np.clip(cand, -5, 5)
                    cf = func(cand)
                    if cf < best_f:
                        best_x = cand
                        best_f = cf
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
