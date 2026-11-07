"""BBOB f14 (Different powers) optimizer.

Best-First Multi-Start Nelder-Mead ensemble with adaptive step sizes,
stagnation detection, and a final coordinate-descent polish phase.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Best-First Multi-Start Nelder-Mead for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # Number of concurrent Nelder-Mead agents
        n_agents = 5

        # Latin Hypercube Sampling for diverse initial points
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_agents + 1)
            for i in range(n_agents):
                lhs_points.append(rng.uniform(strata[i], strata[i+1]))

        lhs_points = np.array(lhs_points).reshape(n_agents, dim)

        # Initialize agents
        agents = []
        for i in range(n_agents):
            x0 = lhs_points[i]
            f0 = call(x0)
            record(x0, f0)

            # Build initial simplex
            simp = [x0.copy()]
            fs = [f0]
            step = 0.5
            for d in range(dim):
                if not ok():
                    break
                xi = x0.copy()
                xi[d] += step
                xi = np.clip(xi, -5, 5)
                fi = call(xi)
                record(xi, fi)
                simp.append(xi.copy())
                fs.append(fi)

            agents.append({
                'simp': simp,
                'fs': fs,
                'last_improve': 0,
                'id': i,
                'step_scale': 1.0
            })

        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sig = 0.5     # contraction
        shr = 0.5     # shrink

        # Main loop: Best-First selection
        global_iter = 0
        stagnation_threshold = 40

        while ok():
            global_iter += 1

            # Select the agent with the best current minimum
            min_fs = [min(a['fs']) for a in agents]
            best_agent_idx = np.argmin(min_fs)
            agent = agents[best_agent_idx]

            simp = agent['simp']
            fs = agent['fs']

            # Sort simplex
            order = np.argsort(fs)
            simp = [simp[i] for i in order]
            fs = [fs[i] for i in order]

            x_best = simp[0]
            f_best = fs[0]
            x_worst = simp[-1]
            f_worst = fs[-1]
            f_second_worst = fs[-2]
            centroid = np.mean(simp[:-1], axis=0)

            # Check for stagnation or convergence before spending a call
            spread = f_worst - f_best
            if spread < 1e-10 or (global_iter - agent['last_improve']) > stagnation_threshold:
                # Restart this agent
                if rng.random() < 0.4 and best_x is not None:
                    # Perturb around global best
                    scale = max(0.1, 1.0 / (1 + global_iter * 0.01))
                    x_new = best_x + rng.normal(0, scale, dim)
                else:
                    x_new = rng.uniform(-5, 5, dim)
                x_new = np.clip(x_new, -5, 5)

                f_new = call(x_new)
                record(x_new, f_new)

                # Rebuild simplex
                simp = [x_new.copy()]
                fs = [f_new]
                step = 0.5
                for d in range(dim):
                    if not ok():
                        break
                    xi = x_new.copy()
                    xi[d] += step
                    xi = np.clip(xi, -5, 5)
                    fi = call(xi)
                    record(xi, fi)
                    simp.append(xi.copy())
                    fs.append(fi)

                agent['simp'] = simp
                agent['fs'] = fs
                agent['last_improve'] = global_iter
                agent['step_scale'] = 1.0
                continue

            # Adaptive step size based on recent performance
            if spread > 1.0:
                agent['step_scale'] = min(2.0, agent['step_scale'] * 1.1)
            elif spread < 0.01:
                agent['step_scale'] = max(0.1, agent['step_scale'] * 0.9)

            # Standard Nelder-Mead Step
            # Reflection
            xr = centroid + gamma * (centroid - x_worst) * agent['step_scale']
            xr = np.clip(xr, -5, 5)
            if not ok():
                break
            fr = call(xr)
            record(xr, fr)

            improved = False
            if fr < f_best:
                # Expansion
                xe = centroid + rho * (xr - centroid) * agent['step_scale']
                xe = np.clip(xe, -5, 5)
                if not ok():
                    break
                fe = call(xe)
                record(xe, fe)
                if fe < fr:
                    simp[-1] = xe
                    fs[-1] = fe
                    improved = True
                else:
                    simp[-1] = xr
                    fs[-1] = fr
                    improved = True
            elif fr < f_second_worst:
                # Accept reflection
                simp[-1] = xr
                fs[-1] = fr
                improved = True
            else:
                # Contraction
                if fr < f_worst:
                    xc = centroid + sig * (xr - centroid) * agent['step_scale']
                else:
                    xc = centroid - sig * (x_worst - centroid) * agent['step_scale']
                xc = np.clip(xc, -5, 5)
                if not ok():
                    break
                fc = call(xc)
                record(xc, fc)
                if fc < min(fr, f_worst):
                    simp[-1] = xc
                    fs[-1] = fc
                    improved = True
                else:
                    # Shrink
                    for i in range(1, len(simp)):
                        if not ok():
                            break
                        simp[i] = x_best + shr * (simp[i] - x_best)
                        simp[i] = np.clip(simp[i], -5, 5)
                        fs[i] = call(simp[i])
                        record(simp[i], fs[i])
                        if fs[i] < f_best:
                            improved = True

            if improved:
                agent['last_improve'] = global_iter

            # Update agent state
            agent['simp'] = simp
            agent['fs'] = fs

        # Final polish phase: coordinate descent on the best point
        if best_x is not None and ok():
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1

            for _ in range(10):  # up to 10 passes of coordinate descent
                if not ok():
                    break
                improved_cd = False
                for d in range(dim):
                    if not ok():
                        break
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        x_try = x_polish.copy()
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, -5, 5)
                        f_try = call(x_try)
                        if f_try < f_polish - 1e-12:
                            x_polish = x_try
                            f_polish = f_try
                            record(x_polish, f_polish)
                            improved_cd = True
                            break
                    if improved_cd:
                        break
                if not improved_cd:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
