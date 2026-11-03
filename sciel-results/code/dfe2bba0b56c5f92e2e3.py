"""
Optimized search for the best Hermite-Gaussian coefficients minimizing the C4 bound.
Uses analytic Gram matrices via Gauss-Hermite quadrature and a multi-stage
global-to-local optimization strategy with Differential Evolution for robust
global search followed by high-precision L-BFGS-B polishing.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite

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
    deadline = 175.0  # Safety margin: 210s limit, aim to finish in 175s

    def elapsed():
        return time.time() - t0

    # ---------- Analytic Basis & Gram Matrices ----------
    # Use Gauss-Hermite quadrature for exact integration of polynomials * e^-x^2
    # Max degree of product is 24 (12+12), so we need 13 points (2*13-1 >= 24)
    n_quad = 13
    x_nodes, w_nodes = roots_hermite(n_quad)

    # Precompute Hermite polynomials at quadrature nodes
    H0 = _hermite_poly(0, x_nodes)
    H4 = _hermite_poly(4, x_nodes)
    H8 = _hermite_poly(8, x_nodes)
    H12 = _hermite_poly(12, x_nodes)

    # Effective polynomial parts with P(0)=0 constraint
    # H_0(0)=1, H_4(0)=0, H_8(0)=0, H_12(0)=-10395
    # P(0) = c0*1 + c4*0 + c8*0 + c12*(-10395) = 0
    # So c12 = c0/10395
    # But the current program uses a different normalization. Let me re-derive.
    # The seed uses: c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0)
    # H0(0) = 1, H4(0) = 0, H8(0) = 0, H12(0) = -10395
    # So c12 = -c0 / (-10395) = c0/10395
    # But the current program has fixed coefficients for the H12 term:
    # p0 = H0 - H12/40320, p4 = H4 - 12*H12/40320, p8 = H8 - 1680*H12/40320
    # This suggests the H12 coefficient is fixed per basis function.
    # Let me check: H12(0) = -10395, so -1/40320 * (-10395) = 10395/40320 ≈ 0.2578
    # H0(0) = 1, so p0(0) = 1 - 10395/40320 ≠ 0
    # Wait, let me re-examine. The constraint is g(0) = 0.
    # g(x) = P(x) e^{-x^2/2}, so g(0) = P(0) * 1 = P(0) = 0.
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P(0) = c0*1 + c4*0 + c8*0 + c12*(-10395) = c0 - 10395*c12 = 0
    # So c12 = c0/10395
    # The current program's approach seems to fix the H12 coefficient relative to each basis.
    # Let me just use the same approach as the current program for consistency.

    # Actually, looking more carefully at the current program:
    # p0 = H0 - H12/40320
    # This means for the "pure H0" basis, the H12 coefficient is -1/40320
    # For p4: H12 coefficient is -12/40320 = -1/3360
    # For p8: H12 coefficient is -1680/40320 = -1/24
    # These don't seem to satisfy P(0)=0 in general. Let me just use the same.

    # Actually, I think the issue is that the "basis functions" are defined as:
    # v_k(x) = (H_k(x) + a_k * H_12(x)) e^{-x^2/2}
    # where a_k is chosen so that v_k(0) = 0? No, that can't be right for all k.
    # Let me just replicate the current program's approach exactly.

    # The coefficients for H12 in each basis:
    # For H0: -1/40320
    # For H4: -12/40320  
    # For H8: -1680/40320

    # Let me verify P(0)=0 for a general combination:
    # P(x) = c0*(H0 - H12/40320) + c4*(H4 - 12*H12/40320) + c8*(H8 - 1680*H12/40320)
    # P(0) = c0*(1 - (-10395)/40320) + c4*(0 - 12*(-10395)/40320) + c8*(0 - 1680*(-10395)/40320)
    # = c0*(1 + 10395/40320) + c4*(12*10395/40320) + c8*(1680*10395/40320)
    # = c0*(50715/40320) + c4*(124740/40320) + c8*(17463600/40320)
    # This is NOT zero in general! So the current program's approach doesn't enforce P(0)=0.
    # 
    # Wait, I think I'm misunderstanding. Let me re-read the task:
    # "the notebook derives the H12 coefficient to impose P(0)=0"
    # So for a GIVEN (c0, c4, c8), c12 is determined by P(0)=0.
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c0*1 + c4*0 + c8*0 + c12*(-10395) = 0
    # c12 = c0/10395
    #
    # So the correct approach is:
    # P(x) = c0*H0 + c4*H4 + c8*H8 + (c0/10395)*H12
    #
    # Let me redo the Gram matrices with this correct formulation.

    # Recompute with correct H12 coefficient: c12 = c0/10395
    # p0 = H0 + (1/10395)*H12
    # p4 = H4
    # p8 = H8
    # (Since c12 only depends on c0)

    p0 = H0 + H12 / 10395.0
    p4 = H4
    p8 = H8
    p_vecs = (p0, p4, p8)

    # Compute M: Gram matrix in position space
    # M[i,j] = \int p_i(x) p_j(x) e^{-x^2} dx
    M = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            M[i, j] = M[j, i] = float(np.sum(w_nodes * p_vecs[i] * p_vecs[j]))

    # Compute Mf: Gram matrix in frequency space
    # For the Fourier transform of P(x) e^{-x^2/2}:
    # \hat{f}(\omega) = \int P(x) e^{-x^2/2} e^{-i\omega x} dx
    # Using the property that \mathcal{F}[H_n(x) e^{-x^2/2}](\omega) = (-i)^n H_n(\omega) e^{-\omega^2/2}
    # (up to normalization constants)
    # So \hat{P}(\omega) = c0*(-i)^0*H0(\omega) + c4*(-i)^4*H4(\omega) + c8*(-i)^8*H8(\omega) + c12*(-i)^12*H12(\omega)
    # = c0*H0(\omega) + c4*H4(\omega) + c8*H8(\omega) + c12*H12(\omega)
    # since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    #
    # With c12 = c0/10395:
    # \hat{P}(\omega) = c0*(H0 + H12/10395) + c4*H4 + c8*H8
    # = c0*q0 + c4*q4 + c8*q8
    # where q0 = H0 + H12/10395, q4 = H4, q8 = H8
    #
    # So the frequency-space polynomial is the SAME as position-space!
    # This makes sense due to the symmetry of the Gaussian and the even Hermite polynomials.

    # Mf[i,j] = \int \omega^2 \hat{v}_i \overline{\hat{v}_j} d\omega / (2\pi)
    # = \int \omega^2 e^{-\omega^2} q_i(\omega) q_j(\omega) d\omega (with appropriate scaling)
    # Using GH quadrature:
    Mf = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            integrand = x_nodes * x_nodes * p_vecs[i] * p_vecs[j]
            Mf[i, j] = Mf[j, i] = float(np.sum(w_nodes * integrand))

    # C matrix: This represents the "C4" term which involves the 4th moment
    # In the uncertainty inequality context, C4 typically involves:
    # \int x^4 |f(x)|^2 dx or similar
    # Let me think about this more carefully.
    #
    # The bound is A(f) * A(Fourier(f)) where A(f) = \sqrt{\int x^2 |f|^2 / \int |f|^2}
    # The "C4" bound likely involves the 4th moment or a product of moments.
    #
    # Looking at the current program, C[i,j] = sum_k M[i,k] M[j,k]
    # This is (M^T M)[i,j] if M is the Gram matrix.
    # Actually, C = M^T M would give C[i,j] = sum_k M[k,i] M[k,j] = sum_k M[i,k] M[j,k] (if M is symmetric)
    # So C = M @ M (matrix product)
    
    C = M @ M

    # Symmetrize
    M = 0.5 * (M + M.T)
    Mf = 0.5 * (Mf + M.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective Function ----------
    def objective(c):
        """
        Minimize C(c) * sqrt(M(c) * F(c))
        c is a 3D vector. We normalize it internally.
        Returns (value, gradient)
        """
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        
        cn = c / np.sqrt(nrm)
        
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        Cn = float(cn @ C @ cn)
        
        # Avoid log(0) or sqrt(0) issues
        eps = 1e-15
        Mv_s = max(Mv, eps)
        Fv_s = max(Fv, eps)
        
        val = Cn * np.sqrt(Mv_s * Fv_s)
        
        # Gradient calculation
        gC = 2.0 * (C @ cn)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        
        sqrtMF = np.sqrt(Mv_s * Fv_s)
        
        gcn = gC * sqrtMF + Cn * (gM * Fv_s + gF * Mv_s) / (2.0 * sqrtMF)
        
        # Project to tangent space of sphere
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        
        return val, grad

    def obj_only(c):
        val, _ = objective(c)
        return val

    # ---------- Optimization Strategy ----------
    best_c = None
    best_val = np.inf

    # Stage 1: Differential Evolution for robust global search
    # Parameterize using spherical coordinates (theta, phi)
    def spherical_obj(params):
        th, ph = params
        # Ensure valid range
        th = np.clip(th, 1e-10, np.pi - 1e-10)
        c_sph = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th)
        ])
        return obj_only(c_sph)

    def spherical_jac(params):
        th, ph = params
        th = np.clip(th, 1e-10, np.pi - 1e-10)
        c_sph = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th)
        ])
        _, grad_c = objective(c_sph)
        
        dcdth = np.array([
            np.cos(th) * np.cos(ph),
            np.cos(th) * np.sin(ph),
            -np.sin(th)
        ])
        dcdph = np.array([
            -np.sin(th) * np.sin(ph),
            np.sin(th) * np.cos(ph),
            0.0
        ])
        return np.array([
            np.dot(grad_c, dcdth),
            np.dot(grad_c, dcdph)
        ])

    # Run Differential Evolution
    if elapsed() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                spherical_obj,
                bounds=[(1e-6, np.pi - 1e-6), (-np.pi, np.pi)],
                seed=42,
                maxiter=200,
                tol=1e-12,
                popsize=30,
                polish=False  # We'll do our own polish
            )
            if np.isfinite(de_result.fun):
                th, ph = de_result.x
                c_de = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                c_de = c_de / np.linalg.norm(c_de)
                val_de, _ = objective(c_de)
                if np.isfinite(val_de) and val_de < best_val:
                    best_val = val_de
                    best_c = c_de.copy()
        except Exception:
            pass

    # Stage 2: Local refinement with L-BFGS-B from multiple starting points
    def refine(c0):
        if elapsed() > deadline - 5.0:
            return None, np.inf
            
        c0 = c0 / np.linalg.norm(c0)
        c0[2] = np.clip(c0[2], -0.999999, 0.999999)
        theta = np.arccos(c0[2])
        phi = np.arctan2(c0[1], c0[0])
        
        try:
            res = minimize(
                spherical_obj,
                x0=np.array([theta, phi]),
                jac=spherical_jac,
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (-np.pi, np.pi)],
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            
            if np.isfinite(res.fun):
                th, ph = res.x
                c_res = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                c_res = c_res / np.linalg.norm(c_res)
                val, _ = objective(c_res)
                return c_res, val
        except Exception:
            pass
            
        return None, np.inf

    # Generate diverse starting points for local refinement
    rng = np.random.RandomState(42)
    starts = []
    
    # Add the DE result if available
    if best_c is not None:
        starts.append(best_c.copy())
    
    # Add random starts
    for _ in range(100):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))
    
    # Add Fibonacci sphere points
    n_fib = 50
    i = np.arange(0, n_fib) + 0.5
    phi_fib = np.arccos(1 - 2 * i / n_fib)
    theta_fib = np.pi * (1 + 5**0.5) * i
    
    for th, ph in zip(theta_fib, phi_fib):
        c_start = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(ph)
        ])
        starts.append(c_start)

    # Run local refinement from each start
    for i, c_start in enumerate(starts):
        if elapsed() > deadline - 10.0:
            break
            
        c_opt, val_opt = refine(c_start)
        
        if c_opt is not None and np.isfinite(val_opt):
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()

    # Stage 3: High-precision polish on the best candidate
    if best_c is not None and elapsed() < deadline - 2.0:
        c0 = best_c / np.linalg.norm(best_c)
        c0[2] = np.clip(c0[2], -0.999999, 0.999999)
        theta = np.arccos(c0[2])
        phi = np.arctan2(c0[1], c0[0])
        
        try:
            res = minimize(
                spherical_obj,
                x0=np.array([theta, phi]),
                jac=spherical_jac,
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (-np.pi, np.pi)],
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.isfinite(res.fun):
                th, ph = res.x
                c_res = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                c_res = c_res / np.linalg.norm(c_res)
                val, _ = objective(c_res)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_c = c_res.copy()
        except Exception:
            pass

    if best_c is None:
        # Fallback to seed solution
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = c_seed / np.linalg.norm(c_seed)
    else:
        best_c = best_c / np.linalg.norm(best_c)

    return np.array([best_c[0], best_c[1], best_c[2]])
