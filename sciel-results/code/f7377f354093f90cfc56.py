"""Model-and-optimize approach: analytic gradient + Riemannian projection on the unit sphere."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


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
    deadline = 195.0  # safety margin vs 210s kill

    def elapsed():
        return time.time() - t0

    # ---------- Grid & basis (higher resolution than seed) ----------
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

    # ---------- Precompute Gram matrices ----------
    def gram(w):
        G = np.empty((3, 3), dtype=float)
        for i in range(3):
            for j in range(i, 3):
                G[i, j] = G[j, i] = float(np.sum(w * v[i] * v[j]) * dx)
        return G

    v = (v0, v4, v8)
    M = gram(np.ones(N))
    Mx = gram(x * x)

    Gf = np.array([np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vi))) * dx for vi in v])
    Mf = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            Mf[i, j] = Mf[j, i] = float(np.sum(freqs * freqs * np.real(Gf[i] * np.conj(Gf[j]))) * dx)

    # Build C correctly (nested)
    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]) * dx)

    # Symmetrize for numerical safety
    M = 0.5 * (M + M.T)
    Mx = 0.5 * (Mx + Mx.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective & analytic gradient ----------
    def objective(c):
        """Return (value, grad) for normalized c. Minimize value * sqrt(M) * sqrt(F)."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        if Mv <= 1e-30 or Fv <= 1e-30:
            return np.inf, np.zeros(3)
        # objective = C*cn  (C* = sum_i c_i^2 C_i), scaled
        Cn = float(cn @ C @ cn)
        val = Cn * np.sqrt(Mv * Fv)
        # gradient of val w.r.t. cn (then embed to c)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        gcn = gC * np.sqrt(Mv * Fv) + Cn * (gM * np.sqrt(Fv) + gF * np.sqrt(Mv)) / (2.0 * np.sqrt(Mv * Fv))
        # chain rule through normalization: grad_c = (grad_cn - (grad_cn.c c) c)/|c|^3
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    def obj_only(c):
        val, _ = objective(c)
        return val

    # ---------- Deterministic multi-start ----------
    rng = np.random.RandomState(12345)

    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-12:
            return c
        return c / n

    # Starting points: seed solution + random directions
    starts = []
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    starts.append(c_seed / np.linalg.norm(c_seed))
    # coordinate axis directions
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        starts.append(e)
    # random directions
    for _ in range(14):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))

    best_c = None
    best_val = np.inf

    def try_refine(c0):
        nonlocal best_c, best_val
        if elapsed() > deadline:
            return
        c0 = project_to_sphere(c0)
        # Riemannian gradient descent with backtracking
        c = c0.copy()
        val, grad = objective(c)
        if not np.isfinite(val):
            return
        lr = 0.5
        for it in range(400):
            if elapsed() > deadline:
                break
            # project gradient to tangent space
            gproj = grad - float(np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gproj)
            if gnorm < 1e-13:
                break
            dstep = -gproj / gnorm
            # backtracking line search
            alpha = lr
            improved = False
            for _ in range(20):
                cand = project_to_sphere(c + alpha * dstep)
                cand_val, _ = objective(cand)
                if np.isfinite(cand_val) and cand_val < val:
                    c = cand
                    val = cand_val
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.7
                if lr < 1e-8:
                    break
        # polish with L-BFGS-B in Cartesian (constrained by renormalizing in wrapper)
        if elapsed() < deadline - 3.0:
            try:
                def cart(c):
                    return obj_only(c)
                res = minimize(
                    cart, x0=c,
                    jac=lambda c: objective(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40}
                )
                if np.isfinite(res.fun) and res.fun < val:
                    c = project_to_sphere(res.x)
                    val, _ = objective(c)
            except Exception:
                pass
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_c = c.copy()

    for s in starts:
        try_refine(s)
        if elapsed() > deadline - 5.0:
            break

    if best_c is None:
        # Fallback to seed
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = c_seed / np.linalg.norm(c_seed)
    else:
        best_c = project_to_sphere(best_c)

    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
