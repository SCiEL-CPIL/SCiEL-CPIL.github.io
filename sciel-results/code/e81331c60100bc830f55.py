"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with periodic restarts.
    
    Uses the Nelder-Mead simplex method which adapts to local curvature.
    Periodic restarts from the global best prevent getting stuck in local minima
    of the multimodal Schaffers F7 landscape.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        def check_deadline():
            return n_calls >= budget or time.time() > deadline

        # ---- Nelder-Mead simplex implementation ---------------------------
        def nelder_mead(x0, max_evals):
            """Run Nelder-Mead from x0, using up to max_evals evaluations."""
            nonlocal n_calls, x_best, f_best, deadline
            
            n = dim
            # Initialize simplex: x0 plus n perturbed points
            simplex = [x0.copy()]
            fvals = [func(x0)]
            record(x0, fvals[0])
            
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                fvals.append(func(xi))
                record(xi, fvals[-1])
                if check_deadline():
                    return

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            while n_calls < budget and time.time() < deadline:
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                record(xr, fr)
                if check_deadline():
                    return

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    record(xe, fe)
                    if check_deadline():
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func(xc)
                    record(xc, fc)
                    if check_deadline():
                        return
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func(simplex[i])
                            record(simplex[i], fvals[i])
                            if check_deadline():
                                return

                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Early termination if simplex is very small
                spread = np.max([np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, n + 1)])
                if spread < 1e-6:
                    break

        # ---- Strategy: Nelder-Mead with periodic restarts ------------------
        # Start with a few diverse random points to find a good region
        n_initial = min(8, budget)
        for _ in range(n_initial):
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            record(x, f)
            if check_deadline():
                return x_best, f_best

        # Now use Nelder-Mead from the best point found, with periodic restarts
        # Each Nelder-Mead run uses about dim+1 evaluations for initialization
        # and then iterates. We'll run for a fixed number of iterations, then restart.
        
        # Estimate: each NM iteration uses ~1-2 evaluations on average
        # Let's run NM for ~40 evaluations per phase, then restart from best
        nm_phase_budget = 40
        max_phases = (budget - n_calls) // nm_phase_budget + 1

        for phase in range(max_phases):
            if check_deadline():
                break
            
            # Start NM from current global best
            x_start = x_best.copy()
            # Add small perturbation to avoid exact same starting point
            if phase > 0:
                x_start = x_start + rng.standard_normal(dim) * 0.1
                x_start = np.clip(x_start, -5.0, 5.0)
            
            # Run Nelder-Mead with a limited budget
            nelder_mead(x_start, nm_phase_budget)
            
            if check_deadline():
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
