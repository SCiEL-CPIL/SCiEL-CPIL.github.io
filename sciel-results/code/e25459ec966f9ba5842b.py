"""Second autocorrelation inequality.
Structural design: DFT parameterization h_k = |P(w^k)|^2 with sparse Sidon polynomial.
This ensures non-negativity and leverages spectral structure of autoconvolution.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 140.0  # safety margin before the 150 s wall-clock kill


def _score(h):
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    l2sq = float(np.dot(h, h))
    sup = float(np.max(np.convolve(h, h)))
    if sup <= 0.0:
        return 0.0
    return l2sq / (l1 * sup)


def _h_from_poly_params(coeffs_real, coeffs_imag, exponents):
    """Compute h_k = |sum_j c_j * w^{k*exp_j}|^2 for k=0..N-1."""
    w = np.exp(2j * np.pi / N)
    # Build frequency grid
    k = np.arange(N)
    # Compute P(w^k) for each k
    # coeffs shape: (K, 2) where [:,0]=real, [:,1]=imag
    c = coeffs_real + 1j * coeffs_imag  # shape (K,)
    # exponents shape: (K,)
    # P_k = sum_j c_j * w^{k * exp_j}
    # Use matrix form: exp_matrix[i,j] = w^{k_i * exp_j}
    exp_matrix = np.exp(2j * np.pi * np.outer(k, exponents) / N)  # (N, K)
    P_vals = exp_matrix @ c  # (N,)
    h = np.abs(P_vals) ** 2
    return h


def _autoconv_peak(h):
    """Compute supremum of autoconvolution h*h."""
    conv = np.convolve(h, h)
    return float(np.max(conv))


def _smooth_max(conv, beta=50.0):
    """Soft max via log-sum-exp."""
    cmax = np.max(conv)
    shifted = beta * (conv - cmax)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    return cmax + np.log(sum_exp) / beta


def _objective_complex(x, exponents, use_soft_max=True, beta=50.0):
    """Objective in terms of real/imag parts of polynomial coefficients."""
    K = len(exponents)
    coeffs_real = x[:K]
    coeffs_imag = x[K:2*K]
    h = _h_from_poly_params(coeffs_real, coeffs_imag, exponents)
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    l2sq = float(np.dot(h, h))
    conv = np.convolve(h, h)
    if use_soft_max:
        sup = _smooth_max(conv, beta)
    else:
        sup = float(np.max(conv))
    if sup <= 0.0:
        return 0.0
    return -(l2sq / (l1 * sup))


def _sidon_exponents(K, seed=0):
    """Generate K Sidon exponents in [0, N-1] for the polynomial P(z)."""
    rng = np.random.default_rng(seed)
    # Start with 0, then greedily add elements ensuring distinct pairwise sums
    pts = [0]
    used_sums = {0}
    for _ in range(K - 1):
        found = False
        for x in range(1, N):
            if x in pts:
                continue
            ok = True
            for a in pts:
                s = a + x
                if s in used_sums:
                    ok = False
                    break
            if ok:
                pts.append(x)
                for a in pts:
                    used_sums.add(a + x)
                found = True
                break
        if not found:
            # Fallback: pick random unused
            used_idx = set(pts)
            candidates = [x for x in range(N) if x not in used_idx]
            if candidates:
                pts.append(rng.choice(candidates))
    return sorted(pts[:K])


def _soft_max_and_grad(conv, beta):
    """Compute soft-max of conv and its gradient w.r.t. conv.
    Returns (sm, d_sm) where sm = soft_max(conv, beta), d_sm = d(sm)/d(conv).
    """
    cmax = np.max(conv)
    shifted = beta * (conv - cmax)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = cmax + np.log(sum_exp) / beta
    d_sm = exp_shifted / sum_exp / beta
    return sm, d_sm


def _jac_complex(x, exponents, use_soft_max=True, beta=50.0):
    """Gradient of _objective_complex w.r.t. x = [real_coeffs, imag_coeffs]."""
    K = len(exponents)
    coeffs_real = x[:K]
    coeffs_imag = x[K:2*K]
    c = coeffs_real + 1j * coeffs_imag
    
    h = _h_from_poly_params(coeffs_real, coeffs_imag, exponents)
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return np.zeros(2*K)
    l2sq = float(np.dot(h, h))
    conv = np.convolve(h, h)
    
    if use_soft_max:
        sup, d_sup_conv = _soft_max_and_grad(conv, beta)
    else:
        sup = float(np.max(conv))
        if sup <= 0.0:
            return np.zeros(2*K)
        am = (conv == sup)
        d_sup_conv = np.zeros_like(conv)
        d_sup_conv[am] = 1.0 / np.sum(am)
    
    if sup <= 0.0:
        return np.zeros(2*K)
    
    # d_sup/dh_i for each i
    d_sup_h = np.zeros(N)
    for i in range(N):
        k_lo = max(0, i)
        k_hi = min(2*N-2, i+N-1)
        if k_lo > k_hi:
            continue
        jj = np.arange(k_lo, k_hi+1) - i
        d_sup_h[i] = 2.0 * float(np.sum(d_sup_conv[k_lo:k_hi+1] * h[jj]))
    
    # val = l2sq / (l1 * sup)
    # d(val)/d(h_i) = (2*h_i*l1*sup - l2sq*(sup + l1*d_sup_h[i])) / (l1*sup)^2
    dh = (2.0 * h * l1 * sup - l2sq * (sup + l1 * d_sup_h)) / (l1 * sup) ** 2
    
    # Now we need d(h)/d(coeffs)
    # h_k = |P_k|^2 where P_k = sum_j c_j * w^{k*exp_j}
    # d(h_k)/d(c_j) = 2 * Re(conj(P_k) * w^{k*exp_j})
    # d(h_k)/d(conj(c_j)) = 2 * Re(P_k * w^{k*exp_j})
    
    k = np.arange(N)
    exp_matrix = np.exp(2j * np.pi * np.outer(k, exponents) / N)  # (N, K)
    P_vals = exp_matrix @ c  # (N,)
    
    # d(h)/d(real_c_j) = 2 * Re(conj(P_k) * w^{k*exp_j})
    # d(h)/d(imag_c_j) = 2 * Im(conj(P_k) * w^{k*exp_j})  [since c = re + i*im, dc/dim = i]
    
    # Actually: h_k = |P_k|^2 = P_k * conj(P_k)
    # d(h_k)/d(c_j) = conj(P_k) * w^{k*exp_j}
    # d(h_k)/d(conj(c_j)) = P_k * w^{k*exp_j}
    
    # c_j = re_j + i*im_j
    # d(c_j)/d(re_j) = 1, d(c_j)/d(im_j) = i
    # d(conj(c_j))/d(re_j) = 1, d(conj(c_j))/d(im_j) = -i
    
    # d(h_k)/d(re_j) = conj(P_k)*w^{k*exp_j} + P_k*w^{k*exp_j} = 2*Re(conj(P_k)*w^{k*exp_j})
    # d(h_k)/d(im_j) = i*conj(P_k)*w^{k*exp_j} - i*P_k*w^{k*exp_j} = 2*Im(conj(P_k)*w^{k*exp_j})
    
    grad_re = np.zeros(K)
    grad_im = np.zeros(K)
    
    for j in range(K):
        wj = exp_matrix[:, j]  # w^{k*exp_j} for all k
        # d(h_k)/d(re_j) = 2*Re(conj(P_k)*w_j)
        d_h_d_re = 2.0 * np.real(np.conj(P_vals) * wj)
        grad_re[j] = float(np.sum(dh * d_h_d_re))
        
        # d(h_k)/d(im_j) = 2*Im(conj(P_k)*w_j)
        d_h_d_im = 2.0 * np.imag(np.conj(P_vals) * wj)
        grad_im[j] = float(np.sum(dh * d_h_d_im))
    
    g = np.zeros(2*K)
    g[:K] = grad_re
    g[K:] = grad_im
    return g


def _run_opt_poly(exponents, x0, tol, max_iter, use_soft_max=True, beta=50.0):
    """Optimize complex polynomial coefficients given fixed Sidon exponents."""
    K = len(exponents)
    
    def objective(x):
        return _objective_complex(x, exponents, use_soft_max=use_soft_max, beta=beta)
    
    def jac(x):
        return _jac_complex(x, exponents, use_soft_max=use_soft_max, beta=beta)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        if res.fun is not None:
            return res.x, -res.fun
    except Exception:
        pass
    return x0, -objective(x0)


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)

    def remaining():
        return _DEADLINE - (time.time() - t0)

    # Try different numbers of Sidon terms in the polynomial
    for K in (3, 4, 5, 6, 7):
        if remaining() < 8.0:
            break
        
        # Generate a few different Sidon exponent sets
        for seed in range(3):
            if remaining() < 4.0:
                break
            
            exponents = _sidon_exponents(K, seed=seed * 1000 + K)
            
            # Initial conditions: try all-real equal, then random complex
            inits = []
            # All real, equal magnitude
            inits.append(np.concatenate([np.ones(K) * 1.0, np.zeros(K)]))
            
            rng = np.random.default_rng(seed * 10000 + K)
            for _ in range(3):
                re = rng.uniform(-1, 1, size=K)
                im = rng.uniform(-1, 1, size=K)
                inits.append(np.concatenate([re, im]))
            
            for x0 in inits:
                if remaining() < 2.0:
                    break
                
                # First pass with soft max
                x_opt, sc = _run_opt_poly(exponents, x0, tol=1e-10, max_iter=200,
                                          use_soft_max=True, beta=30.0)
                if sc > best_score:
                    best_score = sc
                    h = _h_from_poly_params(x_opt[:K], x_opt[K:], exponents)
                    best_h = h.copy()
                
                # Second pass with exact max (polish)
                if remaining() > 2.0:
                    x_opt2, sc2 = _run_opt_poly(exponents, x_opt, tol=1e-12, max_iter=150,
                                                use_soft_max=False, beta=50.0)
                    if sc2 > best_score:
                        best_score = sc2
                        h = _h_from_poly_params(x_opt2[:K], x_opt2[K:], exponents)
                        best_h = h.copy()

    # Final safety: ensure feasibility and return
    if np.any(best_h < 0):
        best_h = np.clip(best_h, 0.0, None)
    return [float(v) for v in best_h]
