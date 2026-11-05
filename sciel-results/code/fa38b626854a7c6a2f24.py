"""Optimized program for the Third Autocorrelation Inequality task.

Strategy: Minimize the peak of the autoconvolution relative to the square of the sum.
By normalizing sum(h)=1, we simply minimize max(conv(h,h)).

Approach:
1. Structured initializations (constant, cosine/sine modes, chirps, two-frequency combos)
2. Differential evolution with a parametric representation (sum of cosines)
3. Local polishing with Nelder-Mead and L-BFGS-B on a smooth-max surrogate
4. Multiple random restarts with perturbations
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Diverse structured initializations
    candidate_list = []

    # Constant
    candidate_list.append(np.ones(N))

    # Linear ramp
    candidate_list.append(np.linspace(0.5, 1.5, N))

    # Cosine modes with various amplitudes
    for k in range(1, 20):
        for amp in [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]:
            h = 1.0 + amp * np.cos(2.0 * np.pi * k * idx / N)
            s = np.sum(h)
            if abs(s) > 1e-10:
                candidate_list.append(h / s)

    # Sine modes
    for k in range(1, 20):
        for amp in [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]:
            h = 1.0 + amp * np.sin(2.0 * np.pi * k * idx / N)
            s = np.sum(h)
            if abs(s) > 1e-10:
                candidate_list.append(h / s)

    # Chirp signals
    for mu in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for amp in [-0.5, -0.25, 0.25, 0.5]:
            phase = 2.0 * np.pi * (idx / N + mu * (idx / N) ** 2)
            h = 1.0 + amp * np.cos(phase)
            s = np.sum(h)
            if abs(s) > 1e-10:
                candidate_list.append(h / s)

    # Two-frequency combinations
    for k1 in [1, 2, 3, 4, 5, 7, 10, 15]:
        for k2 in [2, 4, 6, 8, 12, 15, 20]:
            for a1 in [-0.5, 0.0, 0.5]:
                for a2 in [-0.5, 0.0, 0.5]:
                    if a1 == 0.0 and a2 == 0.0:
                        continue
                    h = 1.0 + a1 * np.cos(2.0 * np.pi * k1 * idx / N) + a2 * np.sin(2.0 * np.pi * k2 * idx / N)
                    s = np.sum(h)
                    if abs(s) > 1e-10:
                        candidate_list.append(h / s)

    # Three-frequency combinations (limited)
    for k1 in [1, 2, 3]:
        for k2 in [4, 6, 8]:
            for k3 in [10, 12, 15]:
                for a1 in [-0.3, 0.3]:
                    for a2 in [-0.3, 0.3]:
                        for a3 in [-0.3, 0.3]:
                            h = 1.0 + a1 * np.cos(2.0 * np.pi * k1 * idx / N) + \
                                  a2 * np.cos(2.0 * np.pi * k2 * idx / N) + \
                                  a3 * np.sin(2.0 * np.pi * k3 * idx / N)
                            s = np.sum(h)
                            if abs(s) > 1e-10:
                                candidate_list.append(h / s)

    # Evaluate all structured candidates
    for h in candidate_list:
        if time.time() > deadline - 80:
            break
        try_candidate(h)

    # Phase 2: Differential evolution for global optimization
    # Represent h as: h[i] = 1 + sum_j a_j * cos(2*pi*f_j*i/N + phi_j)

    def de_objective(params):
        M = len(params) // 3
        h = np.ones(N)
        for j in range(M):
            a = params[3 * j]
            f = params[3 * j + 1]
            phi = params[3 * j + 2]
            h += a * np.cos(2.0 * np.pi * f * idx / N + phi)

        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        h = h / s
        conv = np.convolve(h, h)
        return np.max(conv)

    # Run DE with different numbers of harmonics
    for n_harm in [3, 4, 5, 6]:
        if time.time() > deadline - 40:
            break

        bounds = []
        for j in range(n_harm):
            bounds.append((-1.0, 1.0))
            bounds.append((0.5, 25.0))
            bounds.append((-np.pi, np.pi))

        try:
            res = differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=60,
                popsize=18,
                tol=1e-7,
                seed=42,
                polish=False,
                workers=1
            )

            if time.time() > deadline - 10:
                break

            params = res.x
            M = len(params) // 3
            h = np.ones(N)
            for j in range(M):
                a = params[3 * j]
                f = params[3 * j + 1]
                phi = params[3 * j + 2]
                h += a * np.cos(2.0 * np.pi * f * idx / N + phi)

            s = np.sum(h)
            if abs(s) > 1e-10:
                h = h / s
                try_candidate(h)
        except Exception:
            pass

    # Phase 3: Local refinement using Nelder-Mead with smooth-max
    def soft_max(x, beta=200.0):
        m = np.max(x)
        return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

    def local_refine(h0, max_iter=300, method='Nelder-Mead', beta=200.0):
        if time.time() > deadline - 10:
            return None

        def obj(x):
            conv = np.convolve(x, x)
            sm = soft_max(conv, beta=beta)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty

        try:
            if method == 'Nelder-Mead':
                res = minimize(
                    obj,
                    h0,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            else:
                res = minimize(
                    obj,
                    h0,
                    method=method,
                    options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'gtol': 1e-8}
                )
            return res.x
        except Exception:
            return None

    # Multiple restarts from best solution and perturbations
    if time.time() < deadline - 20:
        h_best = best_h.copy()
        s_best = np.sum(h_best)
        if abs(s_best) > 1e-10:
            h_best = h_best / s_best

        # Refine the best solution
        h_refined = local_refine(h_best, max_iter=400, beta=200.0)
        if h_refined is not None:
            try_candidate(h_refined)

        # Try perturbations
        for trial in range(15):
            if time.time() > deadline - 10:
                break

            # Random perturbation
            h_pert = best_h.copy() + 0.05 * np.random.randn(N)
            s_pert = np.sum(h_pert)
            if abs(s_pert) > 1e-10:
                h_pert = h_pert / s_pert
                h_refined = local_refine(h_pert, max_iter=250, beta=200.0)
                if h_refined is not None:
                    try_candidate(h_refined)

            # Structured perturbation (add a random cosine)
            h_pert = best_h.copy()
            k_rand = np.random.randint(1, 25)
            amp_rand = np.random.uniform(-0.2, 0.2)
            h_pert += amp_rand * np.cos(2.0 * np.pi * k_rand * idx / N)
            s_pert = np.sum(h_pert)
            if abs(s_pert) > 1e-10:
                h_pert = h_pert / s_pert
                h_refined = local_refine(h_pert, max_iter=250, beta=200.0)
                if h_refined is not None:
                    try_candidate(h_refined)

    # Phase 4: Final polish with L-BFGS-B
    if time.time() < deadline - 8:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        for beta in [150.0, 250.0, 350.0, 500.0]:
            if time.time() > deadline - 3:
                break

            def obj_b(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 500.0 * (s - 1.0) ** 2

            try:
                res = minimize(
                    obj_b,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 500, 'gtol': 1e-8}
                )
                if time.time() > deadline - 2:
                    break
                h_new = res.x
                try_candidate(h_new)
                s_new = np.sum(h_new)
                if abs(s_new) > 1e-10:
                    h0 = h_new.copy() / s_new
            except Exception:
                pass

    # Phase 5: Additional refinement with higher beta values
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        for beta in [600.0, 800.0, 1000.0]:
            if time.time() > deadline - 2:
                break

            def obj_high_beta(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 500.0 * (s - 1.0) ** 2

            try:
                res = minimize(
                    obj_high_beta,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 800, 'gtol': 1e-9}
                )
                if time.time() > deadline - 1:
                    break
                h_new = res.x
                try_candidate(h_new)
                s_new = np.sum(h_new)
                if abs(s_new) > 1e-10:
                    h0 = h_new.copy() / s_new
            except Exception:
                pass

    return best_h
