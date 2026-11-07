"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f22 (Gallagher 21 peaks).

    Design: front-load a strong point quickly, then interleave global
    exploration with local refinement so that the AOCC curve (which is the
    mean over all 500 calls) drops as fast as possible. A final high-precision
    polish pass squeezes the tail.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Explicit deadline guard: 330s wall-clock kill, keep a safety margin.
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        box = (-5.0, 5.0)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                return best_f
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        rng = np.random.default_rng(42)
        bounds = [(box, box)] * dim

        # ------------------------------------------------------------------
        # Phase 0: Immediate front-loading. Evaluate a small set of structured
        # points right away so that the very first n_i values are already low.
        # ------------------------------------------------------------------
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([4.0, -4.0, 4.0, -4.0, 4.0])[:dim],
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0])[:dim],
            np.array([3.0, 3.0, -3.0, -3.0, 3.0])[:dim],
            np.array([-3.0, -3.0, 3.0, 3.0, -3.0])[:dim],
        ]
        for x in structured:
            if calls >= budget or time.time() > deadline:
                break
            x = np.clip(x, box[0], box[1])
            eval_and_track(x)

        # ------------------------------------------------------------------
        # Phase 1: Interleave global exploration with local refinement.
        # We keep a list of promising candidates and, as soon as a new good
        # point appears, we launch a quick local refinement from it. This
        # keeps the best-so-far dropping throughout the budget rather than
        # only at the end.
        # ------------------------------------------------------------------
        candidates = []  # list of (f_val, x)
        # Seed with whatever we already have.
        if best_f < float('inf'):
            candidates.append((best_f, best_x.copy()))

        # Quick local refinement helper.
        def quick_refine(x0, maxiter=30):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            if budget - calls < 4:
                return
            try:
                minimize(
                    eval_and_track,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'maxls': 10},
                )
            except Exception:
                pass

        # Exploration loop: draw random points, keep the best few, and
        # periodically refine the current best to keep progress going.
        n_explore_total = 90
        explore_done = 0
        refine_every = 8  # refine the best every few new points

        while explore_done < n_explore_total:
            if calls >= budget or time.time() > deadline:
                break
            if budget - calls < 6:
                break

            # Draw a batch of random candidates.
            batch = 4
            for _ in range(batch):
                if calls >= budget or time.time() > deadline:
                    break
                x = rng.uniform(box[0], box[1], size=dim)
                f_val = eval_and_track(x)
                candidates.append((f_val, x.copy()))
                explore_done += 1

            # Keep only the top-K candidates for diversity selection.
            candidates.sort(key=lambda t: t[0])
            candidates = candidates[:20]

            # Periodic refinement of the best candidate found so far.
            if explore_done % refine_every == 0 and candidates:
                quick_refine(candidates[0][1].copy(), maxiter=25)

        # ------------------------------------------------------------------
        # Phase 2: Multi-start local refinement on diverse good candidates.
        # ------------------------------------------------------------------
        candidates.sort(key=lambda t: t[0])

        remaining = budget - calls
        if remaining > 10:
            # Choose diverse starting points.
            selected = []
            min_dist = 1.2
            for f_val, x in candidates:
                if len(selected) >= 10:
                    break
                too_close = False
                for _, xs in selected:
                    if np.linalg.norm(x - xs) < min_dist:
                        too_close = True
                        break
                if not too_close:
                    selected.append(x)
            # Top up if not enough.
            if len(selected) < 10:
                for f_val, x in candidates:
                    if len(selected) >= 10:
                        break
                    if x not in selected:
                        selected.append(x)

            # Allocate the remaining budget across the local runs.
            n_runs = len(selected)
            if n_runs > 0:
                per_run_iter = max(15, min(40, int(remaining / (n_runs * 2))))
                for x0 in selected:
                    if calls >= budget or time.time() > deadline:
                        break
                    if budget - calls < 5:
                        break
                    quick_refine(x0.copy(), maxiter=per_run_iter)

        # ------------------------------------------------------------------
        # Phase 3: Final high-precision polish from the best point.
        # ------------------------------------------------------------------
        if calls < budget and time.time() < deadline and (budget - calls) >= 5:
            try:
                minimize(
                    eval_and_track,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 30, 'ftol': 1e-15, 'gtol': 1e-12},
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
