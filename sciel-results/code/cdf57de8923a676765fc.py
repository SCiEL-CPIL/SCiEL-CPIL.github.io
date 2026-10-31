"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    # First: standard SLSQP on the raw max-overlap objective
    try:
        def objective_raw(h):
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return float(np.max(overlap))

        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        bounds = [(0.0, 1.0)] * N

        result = minimize(
            objective_raw,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
        )

        if result.fun < current_score:
            h_opt = result.x.copy()
            # Enforce exact sum
            s = np.sum(h_opt)
            diff = N/2.0 - s
            if abs(diff) > 1e-15:
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
            # Re-normalize precisely
            for _ in range(3):
                s = np.sum(h_opt)
                diff = N/2.0 - s
                if abs(diff) > 1e-15:
                    h_opt = h_opt + diff / N
                    h_opt = np.clip(h_opt, 0.0, 1.0)

            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                h_final = h_opt.copy()
                current_score = new_score

    except Exception:
        pass

    # Second: smooth surrogate with epigraph variable t
    # Minimize t subject to h_i * (1 - h_{i+1}) <= t for all i, sum(h) = N/2, 0<=h<=1
    # Use smooth max: t >= smooth_max(overlap)
    try:
        if time.time() < deadline - 10.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=200.0)
                return t + 1000.0 * np.maximum(0.0, smax - t)**2

            # Variables: h (N vars) + t (1 var)
            x0 = np.concatenate([h_final, [current_score + 0.1]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # t - overlap_i >= 0 for all i
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result2 = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )

            if result2.fun < 100.0:  # sanity check
                h_opt2 = result2.x[:N].copy()
                s = np.sum(h_opt2)
                diff = N/2.0 - s
                if abs(diff) > 1e-15:
                    h_opt2 = h_opt2 + diff / N
                    h_opt2 = np.clip(h_opt2, 0.0, 1.0)
                for _ in range(3):
                    s = np.sum(h_opt2)
                    diff = N/2.0 - s
                    if abs(diff) > 1e-15:
                        h_opt2 = h_opt2 + diff / N
                        h_opt2 = np.clip(h_opt2, 0.0, 1.0)

                new_score2 = _score(h_opt2)
                if new_score2 < current_score and np.all(h_opt2 >= 0.0) and np.all(h_opt2 <= 1.0):
                    h_final = h_opt2.copy()
                    current_score = new_score2

    except Exception:
        pass

    # Third: high-precision polish using the smooth objective with tighter alpha
    try:
        if time.time() < deadline - 5.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2

            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )

            if result3.fun < 100.0:
                h_opt3 = result3.x[:N].copy()
                s = np.sum(h_opt3)
                diff = N/2.0 - s
                if abs(diff) > 1e-15:
                    h_opt3 = h_opt3 + diff / N
                    h_opt3 = np.clip(h_opt3, 0.0, 1.0)
                for _ in range(3):
                    s = np.sum(h_opt3)
                    diff = N/2.0 - s
                    if abs(diff) > 1e-15:
                        h_opt3 = h_opt3 + diff / N
                        h_opt3 = np.clip(h_opt3, 0.0, 1.0)

                new_score3 = _score(h_opt3)
                if new_score3 < current_score and np.all(h_opt3 >= 0.0) and np.all(h_opt3 <= 1.0):
                    h_final = h_opt3.copy()
                    current_score = new_score3

    except Exception:
        pass

    # Fourth: random restarts if time permits
    try:
        if time.time() < deadline - 10.0:
            np.random.seed(42)
            for trial in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                # Create a structured random initial point
                h_rand = np.random.rand(N)
                # Add some structure: alternating blocks
                pattern = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], N)
                h_rand = 0.5 * h_rand + 0.5 * pattern
                h_rand = h_rand / np.sum(h_rand) * (N / 2.0)
                h_rand = np.clip(h_rand, 0.0, 1.0)
                for _ in range(3):
                    s = np.sum(h_rand)
                    diff = N/2.0 - s
                    if abs(diff) > 1e-15:
                        h_rand = h_rand + diff / N
                        h_rand = np.clip(h_rand, 0.0, 1.0)

                def objective_raw(h):
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return float(np.max(overlap))

                constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                bounds = [(0.0, 1.0)] * N

                result_r = minimize(
                    objective_raw,
                    h_rand.copy(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )

                if result_r.fun < current_score:
                    h_opt_r = result_r.x.copy()
                    for _ in range(3):
                        s = np.sum(h_opt_r)
                        diff = N/2.0 - s
                        if abs(diff) > 1e-15:
                            h_opt_r = h_opt_r + diff / N
                            h_opt_r = np.clip(h_opt_r, 0.0, 1.0)

                    new_score_r = _score(h_opt_r)
                    if new_score_r < current_score and np.all(h_opt_r >= 0.0) and np.all(h_opt_r <= 1.0):
                        h_final = h_opt_r.copy()
                        current_score = new_score_r

    except Exception:
        pass

    return h_final
