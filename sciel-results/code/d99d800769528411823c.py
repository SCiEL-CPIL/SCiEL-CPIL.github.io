import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin for 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0
        fbest = np.inf
        xbest = np.zeros(n)

        def time_ok():
            return time.time() < deadline

        def make_wrapped():
            def wrapped(x):
                nonlocal calls, fbest, xbest
                if calls >= budget or not time_ok():
                    return fbest if np.isfinite(fbest) else 0.0
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < fbest:
                    fbest = fx
                    xbest = x.copy()
                return fx
            return wrapped

        # Phase 1: Broad multi-start sampling to find promising regions
        n_phase1 = 60
        phase1_points = []
        # Include center and structured points
        phase1_points.append(np.zeros(n))
        # Random points at various scales
        for scale in [0.5, 1.0, 2.0, 3.0, 4.0]:
            for _ in range(10):
                x0 = rng.uniform(-1, 1, n) * scale
                x0 = np.clip(x0, -5, 5)
                phase1_points.append(x0)
        
        # Evaluate all phase 1 points
        phase1_results = []
        for x0 in phase1_points:
            if calls >= budget or not time_ok():
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            calls += 1
            phase1_results.append((x0, fx))
            if fx < fbest:
                fbest = fx
                xbest = x0.copy()

        if not phase1_results:
            return xbest, fbest

        # Sort by fitness and take top candidates for local refinement
        phase1_results.sort(key=lambda t: t[1])
        
        # Phase 2: Multi-start Nelder-Mead from diverse good starting points
        # Use top 8 diverse starts
        n_starts_local = 8
        # Ensure diversity: pick from different regions
        selected_starts = []
        min_dist = 0.8  # minimum distance between selected starts
        
        for x0, fx in phase1_results:
            if len(selected_starts) >= n_starts_local:
                break
            # Check if sufficiently different from already selected
            is_diverse = True
            for xs in selected_starts:
                if np.linalg.norm(x0 - xs) < min_dist:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append(x0)
        
        # If we didn't get enough diverse starts, fill with best remaining
        if len(selected_starts) < n_starts_local:
            for x0, fx in phase1_results:
                if len(selected_starts) >= n_starts_local:
                    break
                if x0 not in selected_starts:
                    selected_starts.append(x0)

        # Allocate remaining budget across local optimizations
        remaining_calls = budget - calls
        calls_per_start = max(30, remaining_calls // max(1, len(selected_starts)))
        
        wrapped = make_wrapped()
        
        for i, x0 in enumerate(selected_starts):
            if calls >= budget or not time_ok():
                break
            
            # Adaptive maxiter based on remaining budget
            maxiter = min(calls_per_start, budget - calls)
            if maxiter < 10:
                break
                
            try:
                result = scipy.optimize.minimize(
                    wrapped, x0, method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
                if result.fun < fbest:
                    fbest = result.fun
                    xbest = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final intensive local refinement from the best point
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 20:
                try:
                    result = scipy.optimize.minimize(
                        wrapped, xbest, method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 200),
                            'xatol': 1e-9,
                            'fatol': 1e-11,
                            'adaptive': True
                        }
                    )
                    if result.fun < fbest:
                        fbest = result.fun
                        xbest = result.x.copy()
                except Exception:
                    pass

        # Phase 4: If still have budget, try Powell method from best point
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 20:
                try:
                    result = scipy.optimize.minimize(
                        wrapped, xbest, method='Powell',
                        options={
                            'maxiter': min(remaining, 100),
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                    if result.fun < fbest:
                        fbest = result.fun
                        xbest = result.x.copy()
                except Exception:
                    pass

        return xbest, fbest
