"""Optimizer for BBOB f22 (Gallagher 21 peaks): random explore + deep NM + ES refine."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase: random sampling → deep Nelder-Mead → diverse NM → adaptive ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        budget = self.budget
        dim = self.dim
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Broad random sampling to identify promising basins
        n_random = 30
        for i in range(n_random):
            if time.time() > deadline - 30.0 or calls_made >= budget - 50:
                break
            x = rng.uniform(-4.8, 4.8, dim)
            eval_func(x)

        if best_x is None:
            best_x = np.zeros(dim)
            eval_func(best_x)

        # Phase 2: Deep Nelder-Mead from the global best
        if time.time() < deadline - 20.0 and calls_made < budget - 20:
            self._nelder_mead(eval_func, best_x.copy(), dim,
                              max_iter=150, deadline=deadline)

        # Phase 3: Diverse NM restarts around best with decreasing scale
        scales = [3.0, 2.0, 1.5, 1.0]
        for scale in scales:
            if time.time() > deadline - 15.0 or calls_made >= budget - 10:
                break
            x_start = best_x + scale * rng.standard_normal(dim)
            x_start = np.clip(x_start, -4.9, 4.9)
            self._nelder_mead(eval_func, x_start, dim,
                              max_iter=80, deadline=deadline)

        # Phase 4: Adaptive (1+1)-ES refinement
        if time.time() < deadline - 5.0 and calls_made < budget - 5:
            self._es_refine(eval_func, best_x.copy(), dim,
                            max_steps=budget - calls_made - 1,
                            deadline=deadline)

        # Phase 5: If budget remains, a few more random probes + quick NM
        if calls_made < budget - 15 and time.time() < deadline - 10.0:
            for _ in range(5):
                if calls_made >= budget - 10 or time.time() > deadline - 5.0:
                    break
                x = rng.uniform(-4.8, 4.8, dim)
                eval_func(x)
                if time.time() < deadline - 10.0 and calls_made < budget - 10:
                    self._nelder_mead(eval_func, best_x.copy(), dim,
                                      max_iter=30, deadline=deadline)

        return best_x, best_f

    def _nelder_mead(self, eval_func, x0, dim, max_iter, deadline):
        """Nelder-Mead simplex with adaptive step and convergence detection."""
        x = np.array(x0, dtype=float)
        f0 = eval_func(x)
        if not np.isfinite(f0):
            return

        # Build simplex with adaptive initial step
        simplex = [x.copy()]
        simplex_f = [f0]
        base_step = 0.5
        for i in range(dim):
            xi = x.copy()
            step = base_step if xi[i] < 4.5 else -base_step
            xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            simplex_f.append(eval_func(xi))

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        min_f = min(simplex_f)
        best_x_local = simplex[np.argmin(simplex_f)].copy()

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            # Sort
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Convergence check
            if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = eval_func(xr)

            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = eval_func(xc)
                if fc < min(simplex_f[-1], fr):
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = eval_func(simplex[i])

            # Update local best
            cur_best_idx = np.argmin(simplex_f)
            if simplex_f[cur_best_idx] < min_f:
                min_f = simplex_f[cur_best_idx]
                best_x_local = simplex[cur_best_idx].copy()

        return best_x_local, min_f

    def _es_refine(self, eval_func, x0, dim, max_steps, deadline):
        """Adaptive (1+1)-ES with success-rate-based step control."""
        x = np.array(x0, dtype=float)
        fx = eval_func(x)
        sigma = 0.15
        rng_local = np.random.default_rng(777)
        successes = 0
        window = 10

        for step in range(max_steps):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng_local.standard_normal(dim), -5.0, 5.0)
            fy = eval_func(y)

            if fy < fx:
                x, fx = y, fy
                successes += 1
            else:
                pass

            # Adaptive sigma based on success rate over last window
            if (step + 1) % window == 0:
                rate = successes / window
                if rate > 0.5:
                    sigma *= 1.25
                elif rate < 0.2:
                    sigma *= 0.7
                successes = 0

            if sigma < 1e-9:
                break

        return x, fx
# EVOLVE-BLOCK-END
