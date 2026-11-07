"""
Optimizer using coordinate-wise 1-D search + joint Nelder-Mead refinement
for BBOB f20 (Schwefel). Structural change: decouples the rugged landscape
into 1-D subproblems via ternary search per coordinate to estimate local centers,
then applies joint Nelder-Mead for final refinement.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin for 300s limit

        n = self.dim
        budget = self.budget

        # Track global best
        best_x = np.zeros(n)
        best_fx = float('inf')
        calls = 0

        def check_deadline():
            return calls >= budget or time.time() > deadline

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Coordinate-wise 1-D ternary search to estimate local centers
        # For each coordinate, fix others at current estimate and do ternary search
        x_est = np.zeros(n)  # initial estimate for each coordinate center

        # Allocate ~40% of budget to 1-D phase, 60% to joint NM
        phase1_budget = int(budget * 0.4)
        phase1_deadline_calls = min(phase1_budget, budget)

        # Do 2 passes of 1-D search for better convergence
        for pass_num in range(2):
            if check_deadline():
                break
            for d in range(n):
                if check_deadline():
                    break

                # Ternary search on dimension d, fixing others
                lo = -5.0
                hi = 5.0
                # Number of ternary steps: use ~8-12 steps per coordinate
                num_steps = 10

                for _ in range(num_steps):
                    if check_deadline():
                        break
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0

                    x1 = x_est.copy()
                    x2 = x_est.copy()
                    x1[d] = m1
                    x2[d] = m2

                    f1 = func(x1)
                    calls += 1
                    update_best(x1, f1)

                    if check_deadline():
                        break

                    f2 = func(x2)
                    calls += 1
                    update_best(x2, f2)

                    if f1 < f2:
                        hi = m2
                    else:
                        lo = m1

                # Set estimate to midpoint of final interval
                x_est[d] = (lo + hi) / 2.0

        # Phase 2: Joint Nelder-Mead refinement around the estimated center
        if not check_deadline():
            x_start = x_est.copy()
            x_start = np.clip(x_start, -5, 5)

            # Initialize simplex around the estimated center
            step = 1.5
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            for i in range(n):
                simplex[i + 1] = x_start
                simplex[i + 1, i] += step
            simplex = np.clip(simplex, -5, 5)

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            init_ok = True
            for i in range(n + 1):
                if check_deadline():
                    init_ok = False
                    break
                fx = func(simplex[i])
                fvals[i] = fx
                calls += 1
                update_best(simplex[i], fx)

            if init_ok and calls > 0:
                # Nelder-Mead parameters
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                # Run NM until budget or deadline
                while not check_deadline():
                    # Sort simplex
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]

                    update_best(simplex[0], fvals[0])

                    # Convergence check
                    spread = np.max(np.abs(simplex - simplex[0]), axis=0)
                    if np.all(spread < 1e-5):
                        break

                    worst_idx = n
                    worst_x = simplex[worst_idx]
                    worst_f = fvals[worst_idx]
                    centroid = np.mean(simplex[:n], axis=0)

                    # Reflection
                    x_ref = centroid + alpha * (centroid - worst_x)
                    x_ref = np.clip(x_ref, -5, 5)

                    if check_deadline():
                        break
                    f_ref = func(x_ref)
                    calls += 1
                    update_best(x_ref, f_ref)

                    if f_ref < fvals[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)

                        if check_deadline():
                            simplex[worst_idx] = x_ref
                            fvals[worst_idx] = f_ref
                            break

                        f_exp = func(x_exp)
                        calls += 1
                        update_best(x_exp, f_exp)

                        if f_exp < f_ref:
                            simplex[worst_idx] = x_exp
                            fvals[worst_idx] = f_exp
                        else:
                            simplex[worst_idx] = x_ref
                            fvals[worst_idx] = f_ref

                    elif f_ref < fvals[n - 1]:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref

                    else:
                        # Contraction
                        x_con = centroid + rho * (worst_x - centroid)
                        x_con = np.clip(x_con, -5, 5)

                        if check_deadline():
                            break

                        f_con = func(x_con)
                        calls += 1
                        update_best(x_con, f_con)

                        if f_con < worst_f:
                            simplex[worst_idx] = x_con
                            fvals[worst_idx] = f_con
                        else:
                            # Shrinkage
                            for i in range(1, n + 1):
                                if check_deadline():
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls += 1
                                update_best(simplex[i], fvals[i])

        return best_x, best_fx
