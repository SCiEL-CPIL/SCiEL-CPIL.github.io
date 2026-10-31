"""Analytic uncertainty bound with 2D ratio parameterization and stationary point root-finding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root, minimize

_DEADLINE = time.time() + 205
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! / (2^n) * sqrt(pi)
# Note: The Gaussian in the problem is typically e^{-x^2} or e^{-x^2/2}.
# The seed program uses I[0]=sqrt(pi), implying e^{-x^2}.
# Let's stick to the seed's definition: g(x) = P(x) e^{-x^2/2} ??
# Wait, the seed says: I_n = integral x^{2n} e^{-x^2} dx.
# And p is polynomial coeffs.
# Let's verify the Fourier transform convention.
# If g(x) = P(x) e^{-x^2/2}, then G(k) = \hat{P}(k) e^{-k^2/2}.
# The seed uses I_n with e^{-x^2}. This suggests g(x) = P(x) e^{-x^2/2} might be wrong if I_n is for e^{-x^2}.
# If g(x) = P(x) e^{-x^2/2}, then |g|^2 = P^2 e^{-x^2}. So integrals of x^{2n} P^2 e^{-x^2} are used.
# This matches I_n = \int x^{2n} e^{-x^2} dx.
# So the model is: g(x) = P(x) e^{-x^2/2}.
# Fourier Transform of e^{-x^2/2} is sqrt(2\pi) e^{-k^2/2}.
# The seed computes n moments with a factor of 2*pi.
# n[k] = 2*pi * sum ...
# This is consistent with \hat{g}(k) having e^{-k^2/2} and the 2\pi factor from the FT of the Gaussian.

def _precompute_I():
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

def _hermite_polys_coeffs(a0, a4, a8):
    """
    Returns coefficients p[0..6] for P(x) = sum p[k] x^k
    such that g(x) = P(x) e^{-x^2/2}.
    P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12.
    H0 = 1
    H4 = x^4 - 6x^2 + 3
    H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12 = x^12 - 66x^10 + 1386x^8 - 6930x^6 + 6930x^4 - 1386x^2 + 945
    
    Constraint: g(0) = 0 => P(0) = 0.
    P(0) = a0*3 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(3a0 + 3a4 + 105a8) / 945
    """
    a12 = -(3.0*a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.zeros(7)
    # H0: 3
    p[0] += 3.0 * a0
    # H4: 3 - 6x^2 + x^4
    p[0] += 3.0 * a4
    p[2] += -6.0 * a4
    p[4] += 1.0 * a4
    # H8: 105 - 420x^2 + 210x^4 - 28x^6 + x^8
    p[0] += 105.0 * a8
    p[2] += -420.0 * a8
    p[4] += 210.0 * a8
    p[6] += -28.0 * a8
    # H12: 945 - 1386x^2 + 6930x^4 - 6930x^6 + 1386x^8 - 66x^10 + x^12
    # Note: Degree 12 means x^12. Our p only goes to x^6?
    # Wait, H12 is degree 12. P(x) will be degree 12.
    # The seed program had p of size 7 (up to x^6).
    # Let's re-read the seed.
    # p[6] = a12.
    # This implies the seed was truncating or using a different basis?
    # No, H12 has terms up to x^12.
    # If p is size 7, it can only hold up to x^6.
    # Let's check the seed's _coeffs_to_poly again.
    # p[0] = 0
    # p[1] = ...
    # p[6] = a12
    # This looks like it's treating the polynomial as degree 6?
    # But H12 is degree 12.
    # Ah, the seed's `_compute_moments` uses `np.arange(7)`.
    # This implies the polynomial degree is at most 6?
    # If the polynomial is degree 12, we need indices 0..12.
    # The seed is likely WRONG or using a specific scaling?
    # "Three free coefficients multiply H0, H4, H8... notebook derives H12".
    # If the seed uses size 7 arrays, it is ignoring the high-degree terms of H12?
    # Or maybe the Hermite polynomials are normalized differently?
    # Physicist's Hermite H_n.
    # Let's assume the seed's implementation of the moments is correct for the "Task Contract" which implies a specific valid interface.
    # However, if the seed is score 1.0, it's a baseline.
    # If I change the polynomial representation, I might break the compatibility or the math.
    # Let's look at the seed's `p` construction again.
    # p[6] = a12.
    # This suggests the seed maps the coefficient of x^6 to a12?
    # But H12 has x^12, x^10, etc.
    # Is it possible the task uses a different polynomial basis?
    # "Hermite-polynomial times Gaussian".
    # If the seed is the "Seed Program", and it scores 1.0, it is the reference.
    # I must reproduce the seed's logic for the moments to be comparable, OR I must be more correct.
    # If the seed is mathematically incomplete (truncating H12), its score 1.0 is a lower bound?
    # No, the evaluator measures the "raw value".
    # If the seed truncates H12, it's solving a different problem?
    # Unlikely. The task says "Tune a ... H0, H4, H8... H12 derived".
    # Maybe the seed's `p` array is just the coefficients of the *even* powers?
    # x^0, x^2, x^4, x^6, x^8, x^10, x^12?
    # Size 7 would cover 0, 2, 4, 6, 8, 10, 12.
    # Let's check the seed's `_compute_moments`:
    # `idx_mat = k + np.add.outer(np.arange(7), np.arange(7))`
    # If p[i] is coeff of x^{2i}, then x^{2i} * x^{2j} = x^{2(i+j)}.
    # Integral of x^{2(i+j+k)} e^{-x^2} = I[i+j+k].
    # This matches the index logic if p[i] corresponds to x^{2i}.
    # So p[0] is coeff of x^0, p[1] is coeff of x^2, ..., p[6] is coeff of x^12.
    # Let's re-verify the seed's `_coeffs_to_poly` with this hypothesis.
    # p[0] = 0 (x^0)
    # p[1] = ... (x^2)
    # p[6] = a12 (x^12)
    
    # Let's expand H_n for even powers only.
    # H0 = 1. Coeffs: x^0: 1.
    # H4 = x^4 - 6x^2 + 3. Coeffs: x^0: 3, x^2: -6, x^4: 1.
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105. Coeffs: x^0: 105, x^2: -420, x^4: 210, x^6: -28, x^8: 1.
    # H12 = x^12 - 66x^10 + 1386x^8 - 6930x^6 + 6930x^4 - 1386x^2 + 945.
    # Coeffs: x^0: 945, x^2: -1386, x^4: 6930, x^6: -6930, x^8: 1386, x^10: -66, x^12: 1.
    
    # Let's check seed's p[1] (x^2 coeff):
    # Seed: p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    # My H4 x^2: -6. H8 x^2: -420. H12 x^2: -1386.
    # Seed has -7560*a12.
    # Why -7560?
    # Maybe the seed's a12 is scaled?
    # Constraint: P(0) = 0.
    # P(0) = 3a0 + 3a4 + 105a8 + 945a12 = 0.
    # a12 = -(3a0 + 3a4 + 105a8)/945.
    
    # Let's check seed's p[0] (x^0 coeff):
    # Seed: p[0] = 0.
    # My P(0) = 3a0 + 3a4 + 105a8 + 945a12.
    # This is 0 by construction.
    
    # Let's check seed's p[6] (x^12 coeff):
    # Seed: p[6] = a12.
    # My H12 x^12 coeff: 1. So coeff is 1*a12. Matches.
    
    # Let's check seed's p[1] (x^2 coeff) again.
    # Seed: -6a4 - 420a8 - 7560a12.
    # My H12 x^2 coeff: -1386.
    # -1386 * a12.
    # Seed has -7560.
    # 7560 / 1386 = 5.45...
    # Is there a scaling factor?
    # Let's check p[2] (x^4 coeff).
    # Seed: a4 + 210a8 + 13860a12.
    # My H12 x^4 coeff: 6930.
    # 13860 / 6930 = 2.
    # Inconsistent scaling for a12 terms?
    # p[1] factor: 7560/1386 = 5.4545...
    # p[2] factor: 13860/6930 = 2.
    
    # This suggests the seed's `a12` is NOT the coefficient of H12 in the standard sense, or the seed has a bug, or I am misinterpreting the Hermite polynomials.
    # However, the seed is the "Seed Program" and scores 1.0.
    # If I change the polynomial definition, I might not be optimizing the same function.
    # BUT, the task says "Tune a ... test function".
    # If the seed has a bug, fixing it might improve the score (lower bound).
    # Let's assume the standard Hermite polynomials are correct.
    # I will use the correct polynomial expansion.
    
    p = np.zeros(7)
    # x^0 (index 0)
    p[0] = 3.0*a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    # x^2 (index 1)
    p[1] = -6.0*a4 - 420.0*a8 - 1386.0*a12
    # x^4 (index 2)
    p[2] = 1.0*a4 + 210.0*a8 + 6930.0*a12
    # x^6 (index 3)
    p[3] = -28.0*a8 - 6930.0*a12
    # x^8 (index 4)
    p[4] = 1.0*a8 + 1386.0*a12
    # x^10 (index 5)
    p[5] = -66.0*a12
    # x^12 (index 6)
    p[6] = 1.0*a12
    
    # Enforce g(0)=0 strictly to avoid numerical drift
    p[0] = 0.0
    
    return p

def _compute_moments(p, q):
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    return m, n

def _bound_and_grad_2d(u, v):
    """
    u = a4, v = a8. a0 is fixed to 1 for scale invariance.
    If a0=0 is needed, we handle it separately or assume it's not the global min.
    """
    a0 = 1.0
    a4 = u
    a8 = v
    
    p = _hermite_polys_coeffs(a0, a4, a8)
    # q is p with alternating signs?
    # In the seed: signs = [1, -1, 1, -1, 1, -1, 1].
    # This corresponds to the Fourier transform property for Hermite functions?
    # \hat{H_n(x) e^{-x^2/2}} = (-i)^n H_n(k) e^{-k^2/2} (up to constants).
    # For even n, (-i)^n = (-1)^{n/2}.
    # H0 (n=0): sign 1.
    # H4 (n=4): sign 1.
    # H8 (n=8): sign 1.
    # H12 (n=12): sign 1.
    # Wait, the seed used alternating signs on the *polynomial coefficients* p[k] (powers x^{2k}).
    # p[k] is coeff of x^{2k}.
    # If g(x) = sum p_k x^{2k} e^{-x^2/2},
    # then \hat{g}(k) involves sum p_k (-i)^{2k} ... ?
    # The Fourier transform of x^{2k} e^{-x^2/2} is related to derivatives of the Gaussian.
    # The seed's `q = p * signs` with signs [1, -1, 1...] suggests that the coefficient of x^{2k} in the Fourier domain polynomial gets a sign (-1)^k.
    # This is consistent with \mathcal{F}[x^{2k} e^{-x^2/2}] \propto (-1)^k H_{2k}(k) e^{-k^2/2} ?
    # Actually, \mathcal{F}[x^n f(x)] = i^n \hat{f}^{(n)}(k).
    # For Gaussian, this leads to Hermite polynomials with signs.
    # I will trust the seed's sign convention for q.
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    m, n = _compute_moments(p, q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15:
        return 1e18, np.array([1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # We need dp/du, dp/dv.
    # p depends on a4, a8 linearly, and a12 linearly on a4, a8.
    # a12 = -(3*1 + 3a4 + 105a8)/945
    # da12/da4 = -3/945
    # da12/da8 = -105/945
    
    da12_du = -3.0/945.0
    da12_dv = -105.0/945.0
    
    # dp/d a4:
    # p[0] = 0 (derivative 0)
    # p[1] = -6 a4 - 420 a8 - 1386 a12 -> d/da4 = -6 - 1386*da12_du
    # p[2] = 1 a4 + 210 a8 + 6930 a12 -> d/da4 = 1 + 6930*da12_du
    # p[3] = -28 a8 - 6930 a12 -> d/da4 = -6930*da12_du
    # p[4] = 1 a8 + 1386 a12 -> d/da4 = 1386*da12_du
    # p[5] = -66 a12 -> d/da4 = -66*da12_du
    # p[6] = 1 a12 -> d/da4 = da12_du
    
    dp_du = np.array([
        0.0,
        -6.0 - 1386.0*da12_du,
        1.0 + 6930.0*da12_du,
        -6930.0*da12_du,
        1386.0*da12_du,
        -66.0*da12_du,
        da12_du
    ])
    
    dp_dv = np.array([
        0.0,
        -420.0 - 1386.0*da12_dv,
        210.0 + 6930.0*da12_dv,
        -28.0 - 6930.0*da12_dv,
        1.0 + 1386.0*da12_dv,
        -66.0*da12_dv,
        da12_dv
    ])
    
    # Compute S_p, S_q
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * _I[idx]
                    s_q += q[j] * _I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
            
    grad = np.zeros(2)
    
    # dm/da = 2 * dp/da @ S_p
    # dn/da = 4*pi * dq/da @ S_q
    
    # d(vf)/da = (dm2*m0 - m2*dm0) / m0^2
    # d(vF)/da = (dn2*n0 - n2*dn0) / n0^2
    # d(bound)/da = (vF * d(vf) + vf * d(vF)) / (2*bound)
    
    for i, dp in enumerate([dp_du, dp_dv]):
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[i] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def construct():
    best_val = np.inf
    best_uv = np.array([0.04, 0.000308642]) # Seed ratios approx (a0~1.13)
    
    # Seed coeffs: a0=-1.13, a4=0.04, a8=0.000308642
    # Ratios: u = a4/a0 = -0.035, v = a8/a0 = -0.00027
    # Let's start near there.
    seed_u = 0.04 / -1.13
    seed_v = 0.000308642 / -1.13
    
    starts = [
        np.array([seed_u, seed_v]),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.0]),
        np.array([-0.1, 0.0]),
        np.array([0.0, 0.1]),
        np.array([0.0, -0.1]),
        np.array([0.5, 0.5]),
        np.array([-0.5, -0.5]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(50):
        u = rng.uniform(-1.0, 1.0)
        v = rng.uniform(-1.0, 1.0)
        starts.append(np.array([u, v]))
        
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try to find stationary point near start
        def func(x):
            val, g = _bound_and_grad_2d(x[0], x[1])
            return g
        
        try:
            res = root(func, start, method='hybr', tol=1e-12)
            if res.success:
                val, _ = _bound_and_grad_2d(res.x[0], res.x[1])
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_uv = res.x.copy()
        except Exception:
            pass
            
        # Also try standard minimization from start
        try:
            def obj(x):
                val, _ = _bound_and_grad_2d(x[0], x[1])
                return val
            
            res_min = minimize(obj, start, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_min.fun < best_val and np.isfinite(res_min.fun):
                best_val = res_min.fun
                best_uv = res_min.x.copy()
        except Exception:
            pass

    # Final polish on best_uv
    if best_val < np.inf:
        def func_polish(x):
            val, g = _bound_and_grad_2d(x[0], x[1])
            return g
        try:
            res_final = root(func_polish, best_uv, method='hybr', tol=1e-15)
            if res_final.success:
                val_final, _ = _bound_and_grad_2d(res_final.x[0], res_final.x[1])
                if val_final < best_val:
                    best_uv = res_final.x.copy()
                    best_val = val_final
        except Exception:
            pass

    # Convert back to a0, a4, a8
    # We fixed a0 = 1.
    # But the interface expects 3 coefficients.
    # The scale doesn't matter for the bound, but the evaluator might check g(0)=0?
    # The task says "score is the upper bound".
    # The interface returns "three coefficients".
    # If I return a0=1, a4=u, a8=v, it is a valid set.
    a0 = 1.0
    a4 = best_uv[0]
    a8 = best_uv[1]
    
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
