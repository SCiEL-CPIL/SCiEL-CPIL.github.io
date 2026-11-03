import numpy as np
from scipy.optimize import fsolve, minimize, differential_evolution
import time
import itertools

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
        # recompute gradient cleanly:
        g = np.zeros(3)
        for i, (ai, arr, ncoef) in enumerate([(a0, H0_arr, 1), (a4, H4_arr, 3), (a8, H8_arr, 5)]):
            base = [H0_0, H4_0, H8_0]
            dP = np.zeros(7)
            for m in range(ncoef):
                dP[m] += arr[m]
            dP += (-(base[i]) / H12_0) * H12_arr
            dP2 = np.convolve(dP, P) + np.convolve(P, dP)
            g[i] = 2 * ns * float(np.dot(dP2, E_int)) / 2.0
        # gradient of norm_sq_hat
        h = np.zeros(3)
        Q = P * signs
        for i, (ai, arr, ncoef) in enumerate([(a0, H0_arr, 1), (a4, H4_arr, 3), (a8, H8_arr, 5)]):
            base = [H0_0, H4_0, H8_0]
            dP = np.zeros(7)
            for m in range(ncoef):
                dP[m] += arr[m]
            dP += (-(base[i]) / H12_0) * H12_arr
            dQ = dP * signs
            dQ2 = np.convolve(dQ, Q) + np.convolve(Q, dQ)
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

    # ---------- Stage 1: Global search using Nelder-Mead from multiple starts ----------
    # The objective is quartic in the coefficients. We search over a reasonable bounding box.
    # Based on the seed and typical scales, we assume coefficients are O(1).
    # We use a set of diverse starting points on the sphere and inside the unit ball.
    
    # Generate starting points
    starts = [x0.copy()]
    
    # Add points on the sphere (unit norm)
    # Fibonacci sphere for uniform distribution
    N = 50
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden_ratio
        starts.append(np.array([r * np.cos(theta), r * np.sin(theta), z]))
        
    # Add points inside the ball (random uniform in cube, normalized if >1)
    rng = np.random.default_rng(42)
    for i in range(50):
        p = rng.uniform(-2, 2, size=3)
        starts.append(p)

    # Add the seed and its negation
    starts.append(-x0.copy())

    # Sort starts by initial objective value to prioritize good regions? 
    # Or just run them all if time permits.
    # Let's run Nelder-Mead on each start.
    
    nm_opts = {'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
    
    for start in starts:
        if time_left() < 2:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', options=nm_opts)
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

    # ---------- Stage 2: Local polish with higher precision ----------
    def polish(start):
        nonlocal best_score, best_result
        # Use BFGS or L-BFGS-B if we can compute gradients? 
        # The current code doesn't have a clean gradient function exposed.
        # We can use Nelder-Mead again with tighter tolerances.
        methods = [
            ('Nelder-Mead', {'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True}),
            # Try Powell as well
            ('Powell', {'maxiter': 50000, 'xtol': 1e-18, 'ftol': 1e-20}),
        ]
        cur = start.copy()
        for mname, opts in methods:
            if time_left() < 2:
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
    
    # Also try polishing the seed again in case the global search missed it
    polish(x0)

    return best_result
