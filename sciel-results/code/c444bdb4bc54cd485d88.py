"""Algebraic-structure approach: spherical coordinates + differential evolution + high-precision polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _hermite_poly(n, x):
    """Physicist's Hermite polynomial via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    t0 = time.time()
    deadline = 198.0  # safety margin vs 210s kill

    def elapsed():
        return time.time() - t0

    # ---------- Grid & basis ----------
    N = 8192
    x = np.linspace(-25.0, 25.0, N)
    dx = x[1] - x[0]
    freqs = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-0.5 * x * x)

    # Fold in g(0)=0 constraint:  c12 = -(c0 + 12 c4 + 1680 c8)/40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss
    v = (v0, v4, v8)

    # ---------- Precompute Gram matrices ----------
    def gram(w):
        G = np.empty((3, 3), dtype=float)
        for i in range(3):
            for j in range(i, 3):
                G[i, j] = G[j, i] = float(np.sum(w * v[i] * v[j]) * dx)
        return G

    M = gram(np.ones(N))       # <v_i, v_j>
    Mx = gram(x * x)           # <x^2 v_i, v_j>

    Gf = np.array([np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vi))) * dx for vi in v])
    Mf = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            Mf[i, j] = Mf[j, i] = float(np.sum(freqs * freqs * np.real(Gf[i] * np.conj(Gf[j]))) * dx)

    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]) * dx)

    # Symmetrize
    M = 0.5 * (M + M.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective on normalized c ----------
    def value_of(cn):
        """Upper bound C4 for a normalized coefficient vector cn (||cn||=1)."""
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf
        Cn = float(cn @ C @ cn)
        return Cn * np.sqrt(Mv * Fv)

    def objective_and_grad(c):
        """Return (value, grad) for arbitrary c (internally normalized)."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf, np.zeros(3)
        Cn = float(cn @ C @ cn)
        val = Cn * np.sqrt(Mv * Fv)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        gcn = gC * np.sqrt(Mv * Fv) + Cn * (gM * np.sqrt(Fv) + gF * np.sqrt(Mv)) / (2.0 * np.sqrt(Mv * Fv))
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    # ---------- Spherical coordinates: c = (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)) ----------
    def sphere_to_c(theta, phi):
        sp = np.sin(phi)
        return np.array([sp * np.cos(theta), sp * np.sin(theta), np.cos(phi)])

    def objective_sphere(params):
        """Objective in spherical coordinates (theta, phi). phi in [0, pi]."""
        theta, phi = params
        c = sphere_to_c(theta, phi)
        return value_of(c)

    # ---------- Differential evolution in 2D spherical coordinates ----------
    # theta in [0, 2*pi], phi in [0, pi]
    bounds = [(0.0, 2.0 * np.pi), (0.0, np.pi)]

    best_val = np.inf
    best_c = None

    # Seed solution as a starting reference
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_seed_n = c_seed / np.linalg.norm(c_seed)
    seed_val = value_of(c_seed_n)
    best_val = seed_val
    best_c = c_seed_n.copy()

    # Convert best_c to spherical for DE initial population seeding
    def c_to_sphere(c):
        c = c / np.linalg.norm(c)
        phi = np.arccos(np.clip(c[2], -1.0, 1.0))
        theta = np.arctan2(c[1], c[0]) % (2.0 * np.pi)
        return [theta, phi]

    # Run differential evolution with time budget
    de_budget = deadline - elapsed() - 20.0  # reserve 20s for polish
    if de_budget > 5.0:
        try:
            # Use a moderate population; DE with 2D is fast
            # maxiter tuned to fit budget: each iteration evaluates ~pop_size*2 points
            pop_size = 30
            maxiter = int(de_budget / 0.5)  # rough estimate: ~0.5s per DE iteration
            maxiter = max(20, min(maxiter, 500))

            # Provide a callback to track best and enforce deadline
            de_best = {'val': best_val}

            def de_callback(xk, convergence):
                nonlocal best_val, best_c
                val = objective_sphere(xk)
                if val < de_best['val']:
                    de_best['val'] = val
                    c = sphere_to_c(xk[0], xk[1])
                    best_c = c.copy()
                    best_val = val
                if elapsed() > deadline - 25.0:
                    raise RuntimeError("timeout")

            res_de = differential_evolution(
                objective_sphere,
                bounds=bounds,
                maxiter=maxiter,
                popsize=pop_size,
                tol=1e-12,
                mutation=(0.3, 1.2),
                recombination=0.9,
                seed=42,
                callback=de_callback,
                polish=False,  # we do our own polish
                init='sobol'
            )
            # Check the final result too
            if res_de.fun < best_val:
                c = sphere_to_c(res_de.x[0], res_de.x[1])
                best_val = res_de.fun
                best_c = c.copy()
        except (RuntimeError, Exception):
            pass  # timeout or error; keep best so far

    # ---------- High-precision polish: Riemannian gradient + L-BFGS ----------
    if best_c is not None and elapsed() < deadline - 5.0:
        c = best_c.copy()
        val = value_of(c)

        # Phase 1: Riemannian gradient descent with backtracking
        lr = 0.3
        for it in range(500):
            if elapsed() > deadline - 5.0:
                break
            _, grad = objective_and_grad(c)
            gproj = grad - float(np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gproj)
            if gnorm < 1e-14:
                break
            dstep = -gproj / gnorm
            alpha = lr
            improved = False
            for _ in range(25):
                cand = c + alpha * dstep
                cnorm = np.linalg.norm(cand)
                if cnorm < 1e-12:
                    break
                cand = cand / cnorm
                cand_val = value_of(cand)
                if np.isfinite(cand_val) and cand_val < val:
                    c = cand
                    val = cand_val
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.7
                if lr < 1e-10:
                    break

        # Phase 2: L-BFGS-B polish in Cartesian with tight tolerances
        if elapsed() < deadline - 3.0:
            try:
                res = minimize(
                    lambda cc: value_of(cc),
                    x0=c,
                    jac=lambda cc: objective_and_grad(cc)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 60}
                )
                if np.isfinite(res.fun) and res.fun < val:
                    c = res.x / np.linalg.norm(res.x)
                    val = value_of(c)
            except Exception:
                pass

        # Phase 3: Final Riemannian polish with even tighter tolerance
        if elapsed() < deadline - 1.0:
            lr = 0.1
            for it in range(200):
                if elapsed() > deadline - 1.0:
                    break
                _, grad = objective_and_grad(c)
                gproj = grad - float(np.dot(grad, c)) * c
                gnorm = np.linalg.norm(gproj)
                if gnorm < 1e-15:
                    break
                dstep = -gproj / gnorm
                alpha = lr
                improved = False
                for _ in range(30):
                    cand = c + alpha * dstep
                    cnorm = np.linalg.norm(cand)
                    if cnorm < 1e-12:
                        break
                    cand = cand / cnorm
                    cand_val = value_of(cand)
                    if np.isfinite(cand_val) and cand_val < val:
                        c = cand
                        val = cand_val
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    lr *= 0.5
                    if lr < 1e-12:
                        break

        best_c = c / np.linalg.norm(c)
        best_val = value_of(best_c)

    if best_c is None:
        best_c = c_seed_n

    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
