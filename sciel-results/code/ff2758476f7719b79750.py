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

    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]) * dx)

    # Symmetrize for numerical safety
    M = 0.5 * (M + M.T)
    Mx = 0.5 * (Mx + Mx.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective & analytic gradient (exact) ----------
    def objective(c):
        """Return (value, grad) for normalized c. Minimize Cn * sqrt(Mv * Fv).
        Returns large value if Mv or Fv <= 0."""
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        cn = c / np.sqrt(nrm)
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        Cn = float(cn @ C @ cn)
        
        # If Mv or Fv is non-positive, the objective is undefined or infinite in the context of the inequality bound
        # We return a large penalty and zero gradient to steer away
        if Mv <= 1e-15 or Fv <= 1e-15:
            return 1e10, np.zeros(3)
            
        val = Cn * np.sqrt(Mv * Fv)
        
        # Gradient of Cn * sqrt(Mv * Fv) w.r.t. cn
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        gC = 2.0 * (C @ cn)
        
        # d/dcn [Cn * sqrt(Mv*Fv)] = gC * sqrt(Mv*Fv) + Cn * (gM*Fv + gF*Mv) / (2*sqrt(Mv*Fv))
        sqrtMF = np.sqrt(Mv * Fv)
        gcn = gC * sqrtMF + Cn * (gM * Fv + gF * Mv) / (2.0 * sqrtMF)
        
        # Project to tangent space and normalize by nrm^1.5
        # Note: The gradient w.r.t c is (gcn - <gcn, cn>cn) / nrm^1.5
        # But since we optimize on the sphere, we often just use the projected gradient
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        return val, grad

    def obj_only(c):
        val, _ = objective(c)
        return val
        
    def jac_only(c):
        _, grad = objective(c)
        return grad

    # ---------- Generalized-eigenvalue sweep with adaptive refinement ----------
    def eig_solve():
        """Generalized-eigenvalue sweep: for fixed Cn, minimize Mv*Fv via eigh."""
        if elapsed() > deadline - 10.0:
            return None, np.inf
        try:
            c_eig = np.linalg.eigvalsh(C)
            c_lo, c_hi = float(c_eig[0]), float(c_eig[-1])
        except Exception:
            c_lo, c_hi = float(np.min(np.diag(C))), float(np.max(np.diag(C)))
        span = max(c_hi - c_lo, 1e-6)
        c_lo -= 0.3 * span
        c_hi += 0.3 * span
        
        # Coarse sweep
        n_sweep = 100
        candidates = []
        for k in range(n_sweep):
            if elapsed() > deadline - 5.0:
                break
            lam1 = c_lo + (c_hi - c_lo) * k / max(n_sweep - 1, 1)
            A = M - lam1 * C
            try:
                mu, vecs = np.linalg.eigh(A, Mf)
            except Exception:
                continue
            for idx in range(vecs.shape[1]):
                v = vecs[:, idx]
                nrm_v = float(np.dot(v, v))
                if nrm_v <= 1e-30:
                    continue
                vn = v / np.sqrt(nrm_v)
                Mv_v = float(vn @ M @ vn)
                Fv_v = float(vn @ Mf @ vn)
                Cn_v = float(vn @ C @ vn)
                if Mv_v <= 1e-30 or Fv_v <= 1e-30 or Cn_v <= 1e-30:
                    continue
                val_v = Cn_v * np.sqrt(Mv_v * Fv_v)
                if np.isfinite(val_v):
                    candidates.append((val_v, lam1, vn.copy()))
        
        if not candidates:
            return None, np.inf
        
        # Sort by value and refine top 10
        candidates.sort(key=lambda x: x[0])
        best_c_local = candidates[0][2]
        best_val_local = candidates[0][0]
        
        # Adaptive refinement around best lam1 values
        top_lams = [c[1] for c in candidates[:10]]
        for lam_center in top_lams:
            if elapsed() > deadline - 3.0:
                break
            # Fine sweep around this lam1
            delta = span / n_sweep
            for k in range(-5, 6):
                lam1 = lam_center + k * delta * 0.1
                A = M - lam1 * C
                try:
                    mu, vecs = np.linalg.eigh(A, Mf)
                except Exception:
                    continue
                for idx in range(vecs.shape[1]):
                    v = vecs[:, idx]
                    nrm_v = float(np.dot(v, v))
                    if nrm_v <= 1e-30:
                        continue
                    vn = v / np.sqrt(nrm_v)
                    Mv_v = float(vn @ M @ vn)
                    Fv_v = float(vn @ Mf @ vn)
                    Cn_v = float(vn @ C @ vn)
                    if Mv_v <= 1e-30 or Fv_v <= 1e-30 or Cn_v <= 1e-30:
                        continue
                    val_v = Cn_v * np.sqrt(Mv_v * Fv_v)
                    if np.isfinite(val_v) and val_v < best_val_local:
                        best_val_local = val_v
                        best_c_local = vn.copy()
        
        return best_c_local, best_val_local

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
    for _ in range(20):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))

    best_c = None
    best_val = np.inf

    # --- Try generalized eigenvalue solve first ---
    eig_c, eig_val = eig_solve()
    if eig_c is not None and np.isfinite(eig_val):
        best_c = eig_c.copy()
        best_val = eig_val
        # Also add as a starting point for gradient refinement
        starts.append(project_to_sphere(eig_c))

    def try_refine(c0):
        nonlocal best_c, best_val
        if elapsed() > deadline:
            return
        c0 = project_to_sphere(c0)
        
        # Direct optimization in spherical coordinates using L-BFGS-B
        # This is more robust than manual Riemannian GD for 2D manifolds
        try:
            theta = np.arccos(np.clip(c0[2], -1, 1))
            phi = np.arctan2(c0[1], c0[0])
            
            def spherical_obj(params):
                th, ph = params
                # Handle periodicity/singularities in spherical coords
                c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                return obj_only(c_sph)
            
            def spherical_jac(params):
                th, ph = params
                c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                _, grad_c = objective(c_sph)
                # Jacobian of c w.r.t. (th, ph)
                dcdth = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
                dcdph = np.array([-np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), 0.0])
                return np.array([np.dot(grad_c, dcdth), np.dot(grad_c, dcdph)])
            
            # Use multiple methods to ensure convergence
            methods = ['L-BFGS-B', 'SLSQP']
            for method in methods:
                if elapsed() > deadline - 2.0:
                    break
                try:
                    res = minimize(
                        spherical_obj,
                        x0=np.array([theta, phi]),
                        jac=spherical_jac,
                        method=method,
                        bounds=[(0, np.pi), (-np.pi, np.pi)],
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
                    )
                    if np.isfinite(res.fun):
                        th, ph = res.x
                        c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                        c_opt = project_to_sphere(c_sph)
                        val_opt, _ = objective(c_opt)
                        if np.isfinite(val_opt) and val_opt < best_val:
                            best_val = val_opt
                            best_c = c_opt.copy()
                            # Update theta/phi for next method if it improved
                            theta = np.arccos(np.clip(best_c[2], -1, 1))
                            phi = np.arctan2(best_c[1], best_c[0])
                except Exception:
                    pass
        except Exception:
            pass

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

    # --- Final high-precision polish pass ---
    if elapsed() < deadline - 2.0 and best_c is not None:
        try:
            theta = np.arccos(np.clip(best_c[2], -1, 1))
            phi = np.arctan2(best_c[1], best_c[0])
            
            def spherical_obj(params):
                th, ph = params
                c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                return obj_only(c_sph)
            
            def spherical_jac(params):
                th, ph = params
                c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                _, grad_c = objective(c_sph)
                dcdth = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
                dcdph = np.array([-np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), 0.0])
                return np.array([np.dot(grad_c, dcdth), np.dot(grad_c, dcdph)])
            
            # Very tight tolerance polish
            res = minimize(
                spherical_obj,
                x0=np.array([theta, phi]),
                jac=spherical_jac,
                method='SLSQP',
                bounds=[(0, np.pi), (-np.pi, np.pi)],
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if np.isfinite(res.fun):
                th, ph = res.x
                c_sph = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                polished = project_to_sphere(c_sph)
                pval, _ = objective(polished)
                if np.isfinite(pval) and pval < best_val:
                    best_c = polished
                    best_val = pval
        except Exception:
            pass

    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
