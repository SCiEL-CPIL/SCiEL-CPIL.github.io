"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer: integer-basin exploitation + multi-start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before the 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        exhausted = False

        def call_func(x):
            nonlocal call_count, best_x, best_f, exhausted
            if call_count >= budget or time.time() > deadline:
                exhausted = True
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(-1)[:dim]
            val = float(func(x))
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def budget_ok():
            return call_count < budget and time.time() < deadline and not exhausted

        # ------------------------------------------------------------------
        # Phase 1: Exploit Rastrigin structure.
        # The global minimum of Rastrigin sits at integer coordinates. With the
        # hidden shift/rotation the minimum lands in [-4,4]^5, so the basin we
        # must reach is an integer lattice point. Evaluate a structured integer
        # grid (cheap, ~100 calls) to locate the basin of the global minimum.
        # ------------------------------------------------------------------
        int_range = list(range(-4, 5))

        # 1a: origin + low-Hamming-weight integer points (cheap, high yield)
        candidates = [np.zeros(dim)]
        for d in range(dim):
            for s in (1, -1):
                pt = np.zeros(dim); pt[d] = s; candidates.append(pt)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        pt = np.zeros(dim); pt[d1] = s1; pt[d2] = s2; candidates.append(pt)
        for pt in candidates:
            if not budget_ok():
                break
            call_func(np.clip(pt, -5, 5))

        # 1b: broader random integer sampling to catch a distant basin
        for _ in range(60):
            if not budget_ok():
                break
            pt = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
            call_func(np.clip(pt, -5, 5))

        # ------------------------------------------------------------------
        # Phase 2: Multi-start Nelder-Mead ensemble.
        # Initialize independent simplexes at the space corners + center + the
        # best integer point, so we simultaneously explore different basins and
        # escape Rastrigin local minima. Each simplex refines its basin toward
        # the true (near-integer) minimum.
        # ------------------------------------------------------------------
        remaining = budget - call_count
        if remaining > 20:
            # seeds: 5 corners + center + best integer point
            seeds = []
            for combo in itertools.product([-4.0, 4.0], repeat=dim):
                seeds.append(np.array(combo, dtype=float))
            seeds.append(np.zeros(dim))
            seeds.append(best_x.copy())

            # cap the ensemble so each start gets a useful Nelder-Mead budget
            min_per = 40
            n_starts = min(len(seeds), max(1, remaining // min_per))
            starts = seeds[:n_starts]
            per = remaining // n_starts

            for s in starts:
                if not budget_ok():
                    break
                x0 = np.clip(s, -5, 5)
                try:
                    minimize(
                        call_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxfev': per, 'xatol': 1e-8, 'fatol': 1e-9,
                                 'adaptive': True, 'maxiter': per * 2},
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Phase 3: Fine coordinate-descent polish around the best point found.
        # Snaps toward the integer basin and tightens each coordinate to
        # machine-level precision, converting basin-finding into exact minimum.
        # ------------------------------------------------------------------
        if budget_ok():
            x = best_x.copy()
            fx = best_f

            # coordinate scan over nearby integers (snap to basin center)
            for d in range(dim):
                if not budget_ok():
                    break
                lo = int(np.floor(x[d])) - 2
                hi = int(np.ceil(x[d])) + 2
                for ic in range(lo, hi + 1):
                    if not budget_ok():
                        break
                    pt = x.copy(); pt[d] = float(ic); pt = np.clip(pt, -5, 5)
                    v = call_func(pt)
                    if v < fx - 1e-12:
                        fx = v; x[d] = float(ic)

            # progressive fine-tuning
            for step in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4):
                if not budget_ok():
                    break
                improved = True
                while improved and budget_ok():
                    improved = False
                    for d in range(dim):
                        if not budget_ok():
                            break
                        for direction in (1, -1):
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            v = call_func(x_new)
                            if v < fx - 1e-12:
                                x = x_new; fx = v; improved = True
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
