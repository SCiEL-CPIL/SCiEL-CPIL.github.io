# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with Halton-based initialization, adaptive restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(123)

        # Phase 1: Diverse initialization using Halton sequence for uniform coverage
        # Plus strategic points
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Halton sequence for uniform coverage of [-5, 5]^dim
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Generate Halton points for initial exploration
        n_halton = 25
        halton_points = np.zeros((n_halton, dim))
        for i in range(1, n_halton + 1):
            for j in range(dim):
                base = primes[j % len(primes)]
                halton_points[i - 1, j] = halton_sequence(i, base)
        # Scale to [-5, 5]
        halton_points = halton_points * 10.0 - 5.0

        # Strategic points
        strategic_points = [
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]

        # Evaluate Halton points first (uniform coverage)
        for pt in halton_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Evaluate strategic points
        for pt in strategic_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and smart restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size

        # Track success rate for 1/5th rule adaptation
        success_window = []
        window_size = 15

        # Stagnation tracking
        stagnation = 0
        no_improve_count = 0

        # Adaptive parameters
        sigma_min = 0.005
        sigma_max = 4.0

        # Budget allocation: use last 15% for polish
        polish_start = int(budget * 0.85)

        while calls < budget:
            if time.time() > deadline:
                break

            # Determine if we're in polish phase
            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.1)

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track success
            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule for sigma adaptation
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.7, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart conditions
            # Stagnation limit decreases as we progress (more patient early, less later)
            progress_ratio = calls / budget
            stagnation_limit = int(40 * (1.0 - 0.5 * progress_ratio))
            no_improve_limit = int(60 * (1.0 - 0.5 * progress_ratio))

            if in_polish:
                stagnation_limit = 15
                no_improve_limit = 25

            # Restart if stagnated
            if stagnation >= stagnation_limit:
                # Adaptive restart: more likely to explore globally as we've been stuck longer
                explore_prob = min(0.3 + 0.01 * no_improve_count, 0.8)

                if rng.random() < explore_prob:
                    # Global exploration using Halton-inspired point
                    idx = int(rng.integers(1, n_halton + 1))
                    pt = np.zeros(dim)
                    for j in range(dim):
                        base = primes[j % len(primes)]
                        pt[j] = halton_sequence(idx + calls, base)
                    x = np.clip(pt * 10.0 - 5.0, -5, 5)
                    sigma = 2.0
                else:
                    # Local exploration around best
                    local_sigma = max(sigma * 2.0, 0.5)
                    x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement
            if no_improve_count >= no_improve_limit:
                # Try multiple random restarts and pick the best
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 3
                for _ in range(n_restarts):
                    if calls >= budget:
                        break
                    cand = rng.uniform(-5, 5, dim)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < best_restart_f:
                        best_restart_f = cand_f
                        best_restart_x = cand
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.5
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if calls >= budget:
                    break

        # Phase 3: Nelder-Mead polish for final precision
        if calls < budget and time.time() < deadline:
            nm_budget = min(budget - calls, 60)  # Use up to 60 calls for NM polish
            if nm_budget >= dim + 2:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)
                calls = budget  # Mark as done

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, deadline):
        """Nelder-Mead simplex method for final polish."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]
        calls = 1

        # Create initial simplex with small perturbations
        step = 0.1
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < f0:
                f0 = fi
                x0 = xi.copy()

        if calls >= budget:
            return x0, f0

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Try expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < f0:
                        f0 = fe
                        x0 = xe.copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < f0:
                        f0 = fr
                        x0 = xr.copy()
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
                if fr < f0:
                    f0 = fr
                    x0 = xr.copy()
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    # Inside contraction
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                fc = func(xc)
                calls += 1

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    if fc < f0:
                        f0 = fc
                        x0 = xc.copy()
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(x0 + sigma_nm * (simplex[i] - x0), -5, 5)
                        fi = func(simplex[i])
                        calls += 1
                        fvals[i] = fi
                        if fi < f0:
                            f0 = fi
                            x0 = simplex[i].copy()

            if calls >= budget:
                break

        return x0, f0
# EVOLVE-BLOCK-END
