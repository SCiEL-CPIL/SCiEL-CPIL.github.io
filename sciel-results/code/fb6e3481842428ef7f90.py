"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based global search with adaptive step + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, calls_left, deadline):
        """Nelder-Mead with adaptive step size."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx
        step = 0.5

        # Build initial simplex
        simplex = [x.copy()]
        for i in range(d):
            xi = x.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            calls += 1
            if calls >= calls_left or time.time() > deadline:
                break
        fvals = np.array([fx] + [func(s) for s in simplex[1:]] if len(simplex) > 1 else [fx])
        calls += max(0, len(simplex) - 1)

        if len(simplex) < 2:
            return best_x, best_f, calls

        # Ensure simplex is valid
        fvals = np.zeros(len(simplex))
        fvals[0] = fx
        for i in range(1, len(simplex)):
            if calls < calls_left and time.time() <= deadline:
                fvals[i] = func(simplex[i])
                calls += 1
            else:
                fvals[i] = fx + i  # dummy

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        max_iter = 200

        for _ in range(max_iter):
            if calls >= calls_left or time.time() > deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if calls >= calls_left or time.time() > deadline:
                if fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                break

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if calls >= calls_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if calls >= calls_left or time.time() > deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls < calls_left and time.time() <= deadline:
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        else:
                            break
                    if calls >= calls_left or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0
        rng = np.random.default_rng(12345)

        # Phase 1: Differential Evolution with adaptive F and CR
        # Rastrigin has many local minima; DE's population diversity helps escape them
        pop_size = 40  # larger population for Rastrigin
        F = 0.8
        CR = 0.5
        calls = 0
        max_generations = 30

        # Initial population: mix of structured and random starts
        pop = np.zeros((pop_size, dim))
        # First few: structured starts
        pop[0] = np.zeros(dim)
        pop[1] = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        pop[2] = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        pop[3] = np.array([2.0, -2.0, 2.0, -2.0, 2.0])
        pop[4] = np.array([-2.0, 2.0, -2.0, 2.0, -2.0])
        # Rest: random in [-4, 4]
        for i in range(5, pop_size):
            pop[i] = rng.uniform(-4, 4, dim)

        # Evaluate initial population
        fvals = np.zeros(pop_size)
        for i in range(pop_size):
            if calls >= budget or time.time() > deadline:
                break
            fvals[i] = func(pop[i])
            calls += 1

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE iterations
        no_improve = 0
        for gen in range(max_generations):
            if calls >= budget or time.time() > deadline:
                break

            # Check if we're converging
            if np.ptp(fvals) < 1e-6:
                break

            # Adaptive F: decrease if convergence is slow
            if no_improve > 5:
                F = max(0.3, F * 0.85)
                CR = min(0.8, CR + 0.05)

            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Select distinct a, b, c
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if len({a, b, c, i}) == 4:
                        break

                # DE/rand/1 mutation
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = func(trial)
                calls += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

        # Phase 2: Nelder-Mead polish from best point
        if calls < budget and time.time() < deadline:
            polish_budget = min(60, budget - calls)
            if polish_budget >= 10:
                x_pol, f_pol, c_pol = self._nm_polish(
                    func, best_x, polish_budget, deadline
                )
                calls += c_pol
                if f_pol < best_f:
                    best_f = f_pol
                    best_x = x_pol.copy()

        # Phase 3: If budget remains, short DE from current best with small F
        if calls < budget and time.time() < deadline and (budget - calls) >= 30:
            small_pop = 20
            F2 = 0.3
            CR2 = 0.8
            gen2 = 0
            max_gen2 = 10
            no_improve2 = 0

            pop2 = np.zeros((small_pop, dim))
            pop2[0] = best_x.copy()
            for i in range(1, small_pop):
                pop2[i] = best_x + rng.normal(0, 0.5, dim)
                pop2[i] = np.clip(pop2[i], -5, 5)

            fvals2 = np.zeros(small_pop)
            fvals2[0] = best_f
            for i in range(1, small_pop):
                if calls >= budget or time.time() > deadline:
                    break
                fvals2[i] = func(pop2[i])
                calls += 1
                if fvals2[i] < best_f:
                    best_f = fvals2[i]
                    best_x = pop2[i].copy()

            for gen2 in range(max_gen2):
                if calls >= budget or time.time() > deadline:
                    break
                for i in range(small_pop):
                    if calls >= budget or time.time() > deadline:
                        break
                    while True:
                        a, b, c = rng.choice(small_pop, 3, replace=False)
                        if len({a, b, c, i}) == 4:
                            break
                    v = pop2[a] + F2 * (pop2[b] - pop2[c])
                    v = np.clip(v, -5, 5)
                    trial = pop2[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR2 or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls += 1
                    if f_trial <= fvals2[i]:
                        pop2[i] = trial
                        fvals2[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                            no_improve2 = 0
                        else:
                            no_improve2 += 1
                    else:
                        no_improve2 += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
