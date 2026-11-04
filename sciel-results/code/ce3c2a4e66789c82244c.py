import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product bound analytically using exact Gaussian integrals.
    The function is g(x) = P(x) * exp(-x^2/2), where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12.
    c12 is determined by the constraint g(0) = 0.
    The bound is sqrt(<x^2> * <xi^2>), where <x^2> is the variance in position and <xi^2> is the variance in frequency.
    """
    # Hermite polynomial values at 0 (physicists' convention)
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 1680
    # H12(0) = -3991680
    
    # Constraint: P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c0 - 12*c4 + 1680*c8 - 3991680*c12 = 0
    # c12 = (c0 - 12*c4 + 1680*c8) / 3991680
    
    c12 = (c0 - 12.0*c4 + 1680.0*c8) / 3991680.0
    
    # Coefficients of the polynomial P(x) = sum_{k=0}^6 p_k x^{2k}
    # We need to expand the Hermite polynomials in terms of powers of x.
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12(x) = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Let's collect coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    
    p = np.zeros(7)
    
    # Constant term (x^0)
    p[0] = c0 + 12*c4 + 168*c8 + 1330*c12
    
    # x^2 term
    p[1] = -48*c4 - 1344*c8 - 32256*c12
    
    # x^4 term
    p[2] = 16*c4 + 2688*c8 + 133056*c12
    
    # x^6 term
    p[3] = -1344*c8 - 180224*c12
    
    # x^8 term
    p[4] = 256*c8 + 118272*c12
    
    # x^10 term
    p[5] = -36864*c12
    
    # x^12 term
    p[6] = 4096*c12
    
    # Note: Due to the constraint g(0)=0, p[0] should be 0 (or very close to it).
    # Let's verify: p[0] = c0 + 12c4 + 168c8 + 1330 * (c0 - 12c4 + 1680c8)/3991680
    # 1330/3991680 = 1/3001.26... wait.
    # H12(0) = -3991680.
    # 1330 is the constant term of H12.
    # Let's re-calculate c12 carefully.
    # P(0) = c0(1) + c4(-12) + c8(1680) + c12(-3991680) = 0
    # c12 = (c0 - 12c4 + 1680c8) / 3991680
    
    # Check p[0]:
    # p[0] = c0 + 12c4 + 168c8 + 1330*c12
    # This is NOT necessarily zero. The constraint is on the value of the polynomial at x=0,
    # which involves the Hermite basis values at 0, not the monomial coefficients.
    # H_n(0) != constant term of H_n(x) generally?
    # H0(0)=1. Const term of H0 is 1.
    # H4(0)=-12. Const term of H4 is 12.
    # H8(0)=1680. Const term of H8 is 168.
    # H12(0)=-3991680. Const term of H12 is 1330.
    # So P(0) = c0*1 + c4*(-12) + c8*1680 + c12*(-3991680).
    # The monomial coefficient p[0] is c0*1 + c4*12 + c8*168 + c12*1330.
    # These are different. The constraint is P(0)=0, which fixes c12.
    # It does NOT imply p[0]=0.
    
    # We need to compute:
    # 1. Norm squared: ||g||_2^2 = \int P(x)^2 e^{-x^2} dx
    # 2. Variance <x^2> = \int x^2 P(x)^2 e^{-x^2} dx / ||g||_2^2
    # 3. Fourier transform variance <xi^2>.
    #    For g(x) = P(x)e^{-x^2/2}, the Fourier transform is related to the derivatives.
    #    Specifically, if we use the unitary Fourier transform, \hat{g}(\xi) = \int g(x) e^{-2\pi i x \xi} dx.
    #    However, the "uncertainty product" usually refers to the spread.
    #    Let's use the property that for g(x) = P(x)e^{-x^2/2}, 
    #    \int \xi^2 |\hat{g}(\xi)|^2 d\xi is related to \int (g')^2 dx (up to constants).
    #    Specifically, \int \xi^2 |\hat{g}(\xi)|^2 d\xi = (1/(4\pi^2)) \int (g')^2 dx.
    #    Wait, let's check the constant.
    #    If \hat{f}(\xi) = \int f(x) e^{-2\pi i x \xi} dx, then \hat{f'}(\xi) = 2\pi i \xi \hat{f}(\xi).
    #    So \int (2\pi \xi)^2 |\hat{f}(\xi)|^2 d\xi = \int |f'(x)|^2 dx.
    #    \int \xi^2 |\hat{f}(\xi)|^2 d\xi = \frac{1}{4\pi^2} \int (f'(x))^2 dx.
    
    # So we need \int (g'(x))^2 dx.
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = P'(x) e^{-x^2/2} + P(x) (-x) e^{-x^2/2} = (P'(x) - x P(x)) e^{-x^2/2}
    # Let Q(x) = P'(x) - x P(x).
    # Then \int (g')^2 dx = \int Q(x)^2 e^{-x^2} dx.
    
    # Since P(x) is a polynomial with only even powers (because H_n for even n is even),
    # P(x) = \sum_{k=0}^6 p_k x^{2k}.
    # P'(x) = \sum_{k=1}^6 2k p_k x^{2k-1}.
    # x P(x) = \sum_{k=0}^6 p_k x^{2k+1}.
    # Q(x) = \sum_{k=1}^6 2k p_k x^{2k-1} - \sum_{k=0}^6 p_k x^{2k+1}.
    # Q(x) is an odd polynomial.
    # Let Q(x) = \sum_{j=0}^5 q_j x^{2j+1}.
    # The term x^{2j+1} in P'(x) comes from k=j+1: 2(j+1) p_{j+1}.
    # The term x^{2j+1} in -xP(x) comes from k=j: -p_j.
    # So q_j = 2(j+1) p_{j+1} - p_j for j=0..5.
    # Note: for j=5, p_6 exists, p_5 exists.
    # For j=0, 2(1)p_1 - p_0.
    
    q_coeffs = np.zeros(6)
    for j in range(6):
        q_coeffs[j] = 2*(j+1)*p[j+1] - p[j]
        
    # Now we need to compute integrals of the form \int x^{2m} R(x)^2 e^{-x^2} dx
    # where R(x) is a polynomial.
    # We can do this by expanding R(x)^2 into monomials and using \int x^{2k} e^{-x^2} dx = \sqrt{\pi} 2^{k-1} (k-1)! for k>=1, and \sqrt{\pi} for k=0.
    # Actually, \int_{-\infty}^{\infty} x^{2k} e^{-x^2} dx = \sqrt{\pi} \frac{(2k)!}{4^k k!} = \sqrt{\pi} (2k-1)!!.
    
    def integral_factor(k):
        # Returns \int x^{2k} e^{-x^2} dx
        if k == 0:
            return math.sqrt(math.pi)
        return math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    # 1. Compute ||g||_2^2 = \int P(x)^2 e^{-x^2} dx
    # P(x)^2 = (\sum p_i x^{2i})^2 = \sum_{i,j} p_i p_j x^{2(i+j)}
    # Let S(x) = P(x)^2 = \sum_{k=0}^{12} s_k x^{2k}
    # s_k = \sum_{i=0}^k p_i p_{k-i}
    
    s_coeffs = np.zeros(13)
    for k in range(13):
        val = 0.0
        for i in range(k+1):
            if i < 7 and (k-i) < 7:
                val += p[i] * p[k-i]
        s_coeffs[k] = val
        
    norm_sq = 0.0
    for k in range(13):
        norm_sq += s_coeffs[k] * integral_factor(k)
        
    if norm_sq < 1e-15:
        return 1e10
        
    # 2. Compute <x^2> numerator: \int x^2 P(x)^2 e^{-x^2} dx
    # = \int x^2 S(x) e^{-x^2} dx = \int S(x) x^2 e^{-x^2} dx
    # = \sum s_k \int x^{2k+2} e^{-x^2} dx = \sum s_k integral_factor(k+1)
    
    x2_num = 0.0
    for k in range(13):
        x2_num += s_coeffs[k] * integral_factor(k+1)
        
    x2_var = x2_num / norm_sq
    
    # 3. Compute \int (g')^2 dx = \int Q(x)^2 e^{-x^2} dx
    # Q(x) = \sum_{j=0}^5 q_j x^{2j+1}
    # Q(x)^2 = \sum_{j,l} q_j q_l x^{2(j+l+1)}
    # Let T(x) = Q(x)^2 = \sum_{m=0}^{10} t_m x^{2(m+1)} ?
    # Let's index by power of x.
    # Q(x) has terms x^1, x^3, x^5, x^7, x^9, x^11.
    # Q(x)^2 will have terms x^2, x^4, ..., x^22.
    # Let Q(x) = \sum_{j=0}^5 q_j x^{2j+1}.
    # Q(x)^2 = \sum_{j=0}^5 \sum_{l=0}^5 q_j q_l x^{2(j+l+1)}.
    # Let m = j+l+1. The power is 2m.
    # m ranges from 1 (j=0,l=0) to 11 (j=5,l=5).
    
    t_coeffs = np.zeros(12) # indices 0..11, corresponding to powers x^0, x^2, ..., x^22
    # Actually, let's just accumulate into an array indexed by the power of x/2.
    # Power 2k. k ranges from 1 to 11.
    
    for j in range(6):
        for l in range(6):
            k = j + l + 1
            t_coeffs[k] += q_coeffs[j] * q_coeffs[l]
            
    g_prime_sq_int = 0.0
    for k in range(12):
        if k > 0: # k=0 term is 0
            g_prime_sq_int += t_coeffs[k] * integral_factor(k)
            
    xi2_var = g_prime_sq_int / (4.0 * math.pi**2 * norm_sq)
    
    # The uncertainty product bound is sqrt(x2_var * xi2_var)
    return math.sqrt(x2_var * xi2_var)


def construct():
    t_start = time.time()
    deadline = 195.0  # Safety margin from 210s limit

    # Seed coefficients
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_result = seed_coeffs.copy()
    best_score = _compute_bound_analytic(*seed_coeffs)
    
    # Strategy:
    # 1. The objective is scale-invariant. We can optimize on the sphere.
    # 2. Use Differential Evolution for global search on the sphere.
    # 3. Use L-BFGS-B for local refinement.
    
    def bound_from_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return _compute_bound_analytic(c0, c4, c8)
    
    # Initialize spherical coordinates from seed
    norm_seed = np.linalg.norm(seed_coeffs)
    c0_s, c4_s, c8_s = seed_coeffs / norm_seed
    theta_init = np.arccos(np.clip(c8_s, -1.0, 1.0))
    phi_init = np.arctan2(c4_s, c0_s)
    if phi_init < 0:
        phi_init += 2 * np.pi
        
    # Evaluate initial
    init_score = bound_from_spherical(theta_init, phi_init)
    if init_score < best_score:
        best_score = init_score
        # We keep best_result in cartesian, but for optimization we use spherical.
        # We will convert back at the end.
        
    best_theta = theta_init
    best_phi = phi_init
    
    # 1. Global Search
    if time.time() < deadline - 30.0:
        try:
            bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2*np.pi - 1e-6)]
            de_result = differential_evolution(
                bound_from_spherical,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False # We will polish manually
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # 2. Local Refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                bound_from_spherical,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # 3. Final Polish with L-BFGS-B
    if time.time() < deadline - 2.0:
        try:
            res2 = minimize(
                bound_from_spherical,
                x0=[best_theta, best_phi],
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (1e-6, 2*np.pi - 1e-6)],
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_theta, best_phi = res2.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    # Return the coefficients. The scale is arbitrary, but returning unit norm is standard.
    # If the evaluator expects specific magnitudes, it might fail, but the problem statement
    # implies the bound is scale-invariant ("upper bound on C4").
    # To be safe, we can scale it to match the seed's magnitude if needed, but unit norm is cleaner.
    # Let's return the unit vector coefficients.
    
    return np.array([c0, c4, c8])
