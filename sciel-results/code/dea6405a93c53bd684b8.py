# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """SPSA (2-eval/iter) gradient-free optimizer for BBOB f14, with a
    time-deadline guard and a final coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0          # hard safety margin (evaluator kills at 330s)
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0

        def remaining():
            return budget - self._used

        self._used = 0

        def f(x):
            self._used += 1
            return float(func(x))

        # --- Start from the origin: a strong prior given min in [-4,4]^dim ---
        x = np.zeros(dim)
        fx = f(x)
        best_x, best_fx = x.copy(), fx

        # --- SPSA gains (time-decaying) ---
        a = 0.5
        c = 0.4
        gamma_a = 0.6
        gamma_c = 0.2
        # Cap the perturbation so x +/- c*delta stays inside the box.
        c_max = (hi - lo) / (2.0 * dim)
        c = min(c, c_max)

        k = 1
        while self._used < budget:
            if time.time() > deadline - 5.0:
                break
            delta = np.sign(np.random.uniform(-1.0, 1.0, dim))
            # Re-draw any zero signs (measure zero) so all coords are perturbed.
            z = delta == 0
            if z.any():
                delta[z] = np.sign(np.random.uniform(-1.0, 1.0, int(z.sum())))
            xp = np.clip(x + c * delta, lo, hi)
            xm = np.clip(x - c * delta, lo, hi)
            fp = f(xp)
            fm = f(xm)
            # Gradient estimate: (f(xp)-f(xm)) / (2 c delta)
            grad = (fp - fm) / (2.0 * c * delta)
            a_k = min(a, 1.0 / (k + 10.0))
            x = np.clip(x - a_k * grad, lo, hi)
            fx = f(x)
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()
            k += 1
            a = a * gamma_a
            c = max(c * gamma_c, 1e-6)

        # --- Coordinate-descent polish (separability of f14) ---
        for _ in range(4):
            if self._used >= budget or time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(dim):
                if self._used >= budget or time.time() > deadline - 2.0:
                    break
                step = 0.4
                for _ in range(12):
                    if self._used >= budget or time.time() > deadline - 2.0:
                        break
                    cand = best_x.copy()
                    cand[i] = min(hi, best_x[i] + step)
                    fv = f(cand)
                    if fv < best_fx:
                        best_x, best_fx = cand, fv
                        improved = True
                    else:
                        cand = best_x.copy()
                        cand[i] = max(lo, best_x[i] - step)
                        fv = f(cand)
                        if fv < best_fx:
                            best_x, best_fx = cand, fv
                            improved = True
                        else:
                            step *= 0.5
                            if step < 1e-5:
                                break
            if not improved:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
