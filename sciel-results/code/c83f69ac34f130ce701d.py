"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        def pso_search(x0, budget_left, rng):
            """Run PSO with adaptive inertia and velocity clamping."""
            nonlocal best_x, best_f, total_calls

            n = dim
            # PSO parameters
            swarm_size = 12
            w_min, w_max = 0.4, 0.9  # inertia weight range
            c1, c2 = 1.5, 1.5        # cognitive and social coefficients
            v_max_ratio = 0.5        # max velocity as fraction of search space

            # Initialize swarm
            positions = np.zeros((swarm_size, n))
            velocities = np.zeros((swarm_size, n))
            pbests = np.zeros((swarm_size, n))
            pbest_f = np.full(swarm_size, np.inf)

            # Initial positions: mix of x0 and random
            positions[0] = x0.copy()
            for i in range(1, swarm_size):
                positions[i] = x0 + rng.uniform(-3, 3, n)
                positions[i] = np.clip(positions[i], -5, 5)

            # Evaluate initial positions
            for i in range(swarm_size):
                if total_calls >= budget or time.time() > deadline:
                    break
                f = safe_call(positions[i])
                if f is not None:
                    pbest_f[i] = f
                    pbests[i] = positions[i].copy()
                    if f < best_f:
                        best_f = f
                        best_x = positions[i].copy()

            gbest = best_x.copy() if best_x is not None else x0.copy()
            gbest_f = best_f

            # Velocity bounds
            v_bound = v_max_ratio * 10.0  # search space is [-5,5], width 10

            # Adaptive inertia: decrease over time
            total_iter_budget = max(1, budget_left // swarm_size)
            iter_count = 0

            while iter_count < total_iter_budget:
                if time.time() > deadline or total_calls >= budget:
                    break

                # Update inertia weight
                progress = iter_count / total_iter_budget
                w = w_max - (w_max - w_min) * progress

                # Update global best
                for i in range(swarm_size):
                    if pbest_f[i] < gbest_f:
                        gbest_f = pbest_f[i]
                        gbest = pbests[i].copy()
                        if gbest_f < best_f:
                            best_f = gbest_f
                            best_x = gbest.copy()

                # Update velocities and positions
                for i in range(swarm_size):
                    if time.time() > deadline or total_calls >= budget:
                        break

                    # Velocity update
                    r1 = rng.random(n)
                    r2 = rng.random(n)
                    velocities[i] = (w * velocities[i] +
                                     c1 * r1 * (pbests[i] - positions[i]) +
                                     c2 * r2 * (gbest - positions[i]))

                    # Clamp velocity
                    velocities[i] = np.clip(velocities[i], -v_bound, v_bound)

                    # Position update
                    positions[i] = positions[i] + velocities[i]
                    positions[i] = np.clip(positions[i], -5, 5)

                    # Evaluate
                    f = safe_call(positions[i])
                    if f is not None:
                        if f < pbest_f[i]:
                            pbest_f[i] = f
                            pbests[i] = positions[i].copy()
                        if f < gbest_f:
                            gbest_f = f
                            gbest = positions[i].copy()
                            if f < best_f:
                                best_f = f
                                best_x = positions[i].copy()

                iter_count += 1

                # Stagnation check: if no improvement in last 5 iterations, reinitialize some particles
                if iter_count % 5 == 0:
                    # Check if gbest has improved recently
                    # Reinitialize worst particles with small perturbation around gbest
                    worst_idx = np.argsort(pbest_f)[-swarm_size // 4:]
                    for idx in worst_idx:
                        if total_calls >= budget or time.time() > deadline:
                            break
                        positions[idx] = gbest + rng.normal(0, 0.5, n)
                        positions[idx] = np.clip(positions[idx], -5, 5)
                        velocities[idx] *= 0.5
                        f = safe_call(positions[idx])
                        if f is not None:
                            if f < pbest_f[idx]:
                                pbest_f[idx] = f
                                pbests[idx] = positions[idx].copy()
                            if f < gbest_f:
                                gbest_f = f
                                gbest = positions[idx].copy()
                                if f < best_f:
                                    best_f = f
                                    best_x = gbest.copy()

            return best_x, best_f

        def nelder_mead_polish(x0, f0, budget_left, rng):
            """Nelder-Mead for final polishing."""
            nonlocal best_x, best_f, total_calls

            n = dim
            simplex = [x0.copy()]
            fs = [f0]
            for i in range(n):
                xi = x0.copy()
                step = 0.1
                xi[i] += step if x0[i] < 4.9 else -step
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fs.append(fi)

            if len(simplex) < 2:
                return x0, f0

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iters = min(budget_left, 100)

            for _ in range(max_iters):
                if time.time() > deadline or total_calls >= budget:
                    break

                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                if abs(fs[0] - fs[-1]) < 1e-14:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fs[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1], fs[-1] = xe, fe
                    else:
                        simplex[-1], fs[-1] = xr, fr
                elif fr < fs[-2]:
                    simplex[-1], fs[-1] = xr, fr
                else:
                    if fr < fs[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= fr:
                        simplex[-1], fs[-1] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if fi is None:
                            break

                min_idx = np.argmin(fs)
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Main search strategy
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with PSO from center
        x0 = np.zeros(dim)
        pso_budget = budget // 2
        pso_search(x0, pso_budget, rng)

        # Phase 2: Local refinement with PSO from best
        if total_calls < budget and time.time() < deadline and best_x is not None:
            remaining = budget - total_calls
            local_budget = min(remaining, budget // 3)
            if local_budget > 20:
                pso_search(best_x.copy(), local_budget, rng)

        # Phase 3: Nelder-Mead polish
        if total_calls < budget and time.time() < deadline and best_x is not None:
            nm_budget = min(budget - total_calls, 100)
            if nm_budget > 10:
                nelder_mead_polish(best_x.copy(), best_f, nm_budget, rng)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
