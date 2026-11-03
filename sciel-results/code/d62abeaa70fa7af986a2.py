import numpy as np
from scipy.optimize import fsolve, minimize, differential_evolution
import time

# ---------- Hermite coefficient vectors (probabilists H_n, index = power//2) ----------
H0_arr = np.array([1.0])
H4_arr = np.array([12.0, -48.0, 16.0])
H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                    1032192.0, -98304.0, 4096.0])
H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 665280.0

# ---------- integral moments: int x^{2k} e^{-pi x^2} dx ----------
E_int = np.zeros(13)
E_int[0] = 1.0
for k in range(1, 13):
    df = 1.0
    for j in range(1, 2 * k, 2):
        df *= j
    E_int[k] = df / (2.0 * np.pi) ** k

signs = np.array([(-1.0) ** m for m in range(7)])


def P_of(a0, a4, a8):
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    P = np.zeros(7)
    P[0] += a0 * H0_arr[0]
    for m in range(3):
        P[m] += a4 * H4_arr[m]
    for m in range(5):
        P[m] += a8 * H8_arr[m]
    for m in range(7):
        P[m] += a12 * H12_arr[m]
    return P


def norm_sq(P):
    P2 = np.convolve(P, P)
    return float(np.dot(P2, E_int))


def norm_sq_hat(P):
    Q = P * signs
    Q2 = np.convolve(Q, Q)
    return float(np.dot(Q2, E_int))


def objective(params):
    a0, a4, a8 = params
    P = P_of(a0, a4, a8)
    return norm_sq(P) * norm_sq_hat(P)


def solve_fixed_point(t, p_init):
    """Solve the 3 nonlinear fixed-point equations for direction (a0,a4,a8)
    at homotopy parameter t. Returns solution or None."""
    def equations(p):
        a0, a4, a8 = p
        P = P_of(a0, a4, a8)
        ns = norm_sq(P)
        nsh = norm_sq_hat(P)
        if ns <= 0.0 or nsh <= 0.0:
            return [1e6, 1e6, 1e6]
        # gradient of objective w.r.t. (a0,a4,a8)
        grad = np.array([
            2 * ns * (2 * np.dot(np.convolve(P, P[1:]), E_int)
                      + 2 * np.dot(np.convolve(P, P[:1]), E_int))
            if False else
            2 * ns * (2 * np.dot(np.concatenate([P, P[1:]]), E_int))
        ])
        # recompute gradient cleanly:
        g = np.zeros(3)
        for i, (ai, arr, ncoef) in enumerate([(a0, H0_arr, 1), (a4, H4_arr, 3), (a8, H8_arr, 5)]):
            dP = np.zeros(7)
            for m in range(ncoef):
                dP[m] += arr[m]
            # da12/dai
            dP[0] += -(arr[0] if i == 0 else (H4_arr[0] if i == 1 else H8_arr[0])) * 0  # placeholder
            # compute da12/dai properly
            base = [H0_0, H4_0, H8_0]
            dP = np.zeros(7)
            for m in range(ncoef):
                dP[m] += arr[m]
            dP += (-(base[i]) / H12_0) * H12_arr
            dP2 = np.convolve(dP, P) + np.convolve(P, dP)
            g[i] = 2 * ns * float(np.dot(dP2, E_int)) / 2.0
        # gradient of norm_sq_hat
        h = np.zeros(3)
        for i, (ai, arr, ncoef) in enumerate([(a0, H0_arr, 1), (a4, H4_arr, 3), (a8, H8_arr, 5)]):
            base = [H0_0, H4_0, H8_0]
            dP = np.zeros(7)
            for m in range(ncoef):
                dP[m] += arr[m]
            dP += (-(base[i]) / H12_0) * H12_arr
            dQ = dP * signs
            dQ2 = np.convolve(dQ, Q) + np.convolve(Q, dQ)
            Q = P * signs
            h[i] = 2 * nsh * float(np.dot(dQ2, E_int)) / 2.0
        grad = ns * (g) + nsh * (h)
        # fixed point: p_i - t * grad_i = 0  (homotopy: t=0 -> p=0; t=1 -> grad=0)
        return [p[i] - t * grad[i] for i in range(3)]

    try:
        sol, info, ier, msg = fsolve(equations, p_init, full_output=True, maxfev=20000)
        if ier == 1 and np.all(np.abs(sol) < 1e2):
            # verify it's a real stationary point
            res = equations(sol)
            if np.max(np.abs(res)) < 1e-4 * (1 + np.linalg.norm(sol)):
                return sol
    except Exception:
        pass
    return None


def construct():
    t_start = time.time()
    deadline = 195.0  # safety margin from 210s

    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = x0.copy()
    best_score = objective(x0)

    def time_left():
        return deadline - time.time()

    # ---------- Stage 1: global search on the sphere (2D) via differential evolution ----------
    if time_left() > 60:
        try:
            def sphere_obj(sp):
                theta, phi = sp
                d0 = np.cos(theta)
                d4 = np.sin(theta) * np.cos(phi)
                d8 = np.sin(theta) * np.sin(phi)
                return objective(np.array([d0, d4, d8]))

            de = differential_evolution(
                sphere_obj,
                bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
                seed=12345,
                maxiter=200,
                tol=1e-12,
                popsize=30,
                polish=False,
                init='latinhypercube',
                mutation=(0.4, 0.9),
                recombination=0.7,
            )
            if de.fun < best_score:
                theta, phi = de.x
                d0 = np.cos(theta)
                d4 = np.sin(theta) * np.cos(phi)
                d8 = np.sin(theta) * np.sin(phi)
                cand = np.array([d0, d4, d8])
                best_score = de.fun
                best_result = cand
        except Exception:
            pass

    # ---------- Stage 2: fixed-point solver with homotopy continuation ----------
    # homotopy from t=0 (trivial) to t=1 (real problem)
    if time_left() > 30:
        # build a set of initial guesses: DE result, seed, and sphere samples
        inits = [best_result.copy()]
        # add seed and a coarse sphere grid
        for it in range(0, 12):
            for ip in range(0, 12):
                theta = it * 2 * np.pi / 12
                phi = ip * 2 * np.pi / 12
                inits.append(np.array([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)]))
        inits.append(x0.copy())

        n_steps = 24
        for start_idx, init in enumerate(inits):
            if time_left() < 8:
                break
            # normalize init to unit direction
            nrm = np.linalg.norm(init)
            if nrm < 1e-12:
                continue
            p = init / nrm
            success = True
            for step in range(1, n_steps + 1):
                t = step / n_steps
                if time_left() < 5:
                    success = False
                    break
                sol = solve_fixed_point(t, p)
                if sol is None:
                    success = False
                    break
                p = sol
            if success:
                # evaluate at t=1 solution (a stationary direction)
                cand = p.copy()
                # scale: objective is quartic, so direction matters; pick unit norm
                val = objective(cand)
                if val < best_score:
                    best_score = val
                    best_result = cand

    # ---------- Stage 3: local polish with Nelder-Mead + BFGS + Powell ----------
    def polish(start):
        nonlocal best_score, best_result
        methods = [
            ('Nelder-Mead', {'maxiter': 200000, 'xatol': 1e-20, 'fatol': 1e-22, 'adaptive': True}),
            ('BFGS', {'maxiter': 50000, 'gtol': 1e-24}),
            ('Powell', {'maxiter': 200000, 'xtol': 1e-20, 'ftol': 1e-22}),
        ]
        cur = start.copy()
        for mname, opts in methods:
            if time_left() < 4:
                break
            try:
                res = minimize(objective, cur, method=mname, options=opts)
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
                    cur = res.x.copy()
            except Exception:
                pass

    polish(best_result)
    # also polish the seed in case we drifted
    polish(x0)

    return best_result
