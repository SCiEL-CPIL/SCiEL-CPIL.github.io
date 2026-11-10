import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    
    Strategy: Exploit the fact that R = ||f||_4/||f||_2 is scale invariant.
    Instead of optimizing over the 21-dimensional coefficient space directly,
    we use a different parameterization: optimize over the direction on the unit sphere
    using a gradient-based method with careful initialization based on the structure
    of spherical harmonics. The key insight is that the optimal solution likely has
    specific symmetry properties, so we focus our search on structured initial conditions
    and use a more robust optimization approach with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 600

    k = 10
    n_m = 2 * k + 1

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))

    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)

    # m = 0
    m = 0
    idx = 10
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P

    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)

    # 3. Objective Function and Gradient
    def objective_and_grad(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2

        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi

        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)

        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2

        # Gradients
        w_f = w_flat * f_vals
        w_f3 = w_flat * f_vals ** 3

        sum_wf_A = A.T @ w_f * d_phi
        sum_wf3_A = A.T @ w_f3 * d_phi

        grad = (N4**3 / N2) * sum_wf3_A - (N4 / N2**3) * sum_wf_A

        return R, grad

    # 4. Generate Initial Candidates
    # Focus on sparse initial guesses and structured patterns.
    candidates = []
    rng = np.random.default_rng(123)

    # 1. Single modes
    for i in range(n_m):
        c = np.zeros(n_m)
        c[i] = 1.0
        candidates.append(c)

    # 2. Pairs of modes (random)
    for _ in range(50):
        i, j = rng.choice(n_m, size=2, replace=False)
        c = np.zeros(n_m)
        c[i] = 1.0
        c[j] = rng.choice([-1.0, 1.0])
        candidates.append(c)

    # 3. Triplets of modes (random)
    for _ in range(30):
        idxs = rng.choice(n_m, size=3, replace=False)
        c = np.zeros(n_m)
        for idx in idxs:
            c[idx] = rng.choice([-1.0, 1.0])
        candidates.append(c)

    # 4. Random sparse (2-5 non-zero)
    for _ in range(50):
        n_nonzero = rng.integers(2, 6)
        idxs = rng.choice(n_m, size=n_nonzero, replace=False)
        c = np.zeros(n_m)
        for idx in idxs:
            c[idx] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
        candidates.append(c)

    # 5. Symmetric patterns (c[m] = c[-m])
    for _ in range(20):
        c = np.zeros(n_m)
        for m in range(1, k + 1):
            val = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            c[m + 10] = val
            c[-m + 10] = val
        c[10] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
        candidates.append(c)

    # 6. Anti-symmetric patterns (c[m] = -c[-m])
    for _ in range(20):
        c = np.zeros(n_m)
        for m in range(1, k + 1):
            val = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            c[m + 10] = val
            c[-m + 10] = -val
        c[10] = 0.0
        candidates.append(c)

    # 7. All ones, all alternating signs
    candidates.append(np.ones(n_m))
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_m)]))

    best_c = candidates[0]
    best_val = np.inf

    # 5. Optimize each candidate
    # Sort candidates by initial R value to prioritize promising starts
    def initial_score(c):
        try:
            val, _ = objective_and_grad(c)
            return val
        except:
            return np.inf

    candidates_with_score = [(initial_score(c), c) for c in candidates]
    candidates_with_score.sort(key=lambda x: x[0])

    for i, (score, c0) in enumerate(candidates_with_score):
        if time.time() > deadline - 20:
            break
        if score >= best_val and i > 10:  # Skip if initial score is worse than best found
            continue

        try:
            # Normalize to unit norm for stability
            norm = np.linalg.norm(c0)
            if norm > 0:
                c0_norm = c0 / norm
            else:
                c0_norm = c0

            res = minimize(
                objective_and_grad,
                c0_norm,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-12}
            )

            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x

        except Exception:
            pass

    # 6. Refine the best candidate
    if time.time() < deadline - 30 and np.all(np.isfinite(best_c)):
        try:
            norm = np.linalg.norm(best_c)
            if norm > 0:
                best_c_norm = best_c / norm
            else:
                best_c_norm = best_c

            res2 = minimize(
                objective_and_grad,
                best_c_norm,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if res2.fun < best_val:
                best_c = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Final check on validity
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_m)

    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)

    return best_c
