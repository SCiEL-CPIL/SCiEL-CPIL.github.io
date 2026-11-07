"""Rewritten optimizer: Multi-start Nelder-Mead with final polish for Sharp Ridge."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start initialization, Nelder-Mead refinement,
    and a final coordinate-descent polish pass for high-precision convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        # Sample to locate the basin of the sharp ridge.
        n_starts = 25
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: Nelder-Mead with adaptive simplex ----------
        def nelder_mead(start_x, max_evals=100):
            """Run Nelder-Mead from start_x, using up to max_evals function calls."""
            nonlocal evaluated, elapsed_ok, clip, eval_point, dim, budget

            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma_shrink = 0.5  # Shrink

            # Build initial simplex with adaptive step size
            # Use a step that's reasonable for the box [-5, 5]^5
            step_size = 0.5
            simplex = [start_x.copy()]
            for i in range(dim):
                x_pert = start_x.copy()
                # Choose direction that stays in bounds
                if start_x[i] > 4.0:
                    step = -step_size
                elif start_x[i] < -4.0:
                    step = step_size
                else:
                    step = step_size
                x_pert[i] += step
                simplex.append(clip(x_pert))

            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                if evaluated >= budget or not elapsed_ok():
                    return
                simplex_f.append(eval_point(x))

            if len(simplex_f) < dim + 1:
                return

            # Nelder-Mead iteration loop
            iters = 0
            while evaluated < budget and elapsed_ok() and iters < 200:
                iters += 1

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence: small spread in function values
                f_spread = simplex_f[-1] - simplex_f[0]
                if f_spread < 1e-10 * (1.0 + abs(simplex_f[0])):
                    break

                # Also check if simplex is very small in space
                if np.max(np.array(simplex)) - np.min(np.array(simplex)) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = clip(x_ref)
                f_ref = eval_point(x_ref)

                if f_ref < simplex_f[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = clip(x_exp)
                    f_exp = eval_point(x_exp)

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])

                    x_con = clip(x_con)
                    f_con = eval_point(x_con)

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink all points toward best
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            simplex_f[i] = eval_point(simplex[i])
                            if evaluated >= budget or not elapsed_ok():
                                return
                        if evaluated >= budget or not elapsed_ok():
                            return

        # Run Nelder-Mead from the best point found
        nelder_mead(best_x.copy(), max_evals=150)

        # ---------- Phase 3: Final Polish with Coordinate Descent ----------
        # If we still have budget, do a fine-grained coordinate descent
        # to squeeze out additional precision on the sharp ridge.
        if evaluated < budget and elapsed_ok() and best_x is not None:
            # Coordinate descent with adaptive step size
            x_current = best_x.copy()
            f_current = best_f
            base_step = 0.1
            min_step = 1e-6

            # Try multiple coordinate descent passes with decreasing steps
            for pass_num in range(3):
                step = base_step * (0.5 ** pass_num)
                for coord in range(dim):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    # Try positive and negative steps
                    for direction in [1.0, -1.0]:
                        x_trial = x_current.copy()
                        x_trial[coord] += direction * step
                        x_trial = clip(x_trial)
                        f_trial = eval_point(x_trial)
                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
                            # Accept and try to go further in same direction
                            # (simple: just continue to next coordinate)
                    if evaluated >= budget or not elapsed_ok():
                        break

            # Update best if improved
            if f_current < best_f:
                best_f = f_current
                best_x = x_current.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
