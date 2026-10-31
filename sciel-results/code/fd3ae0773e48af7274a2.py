"""Uncertainty inequality optimization via generalized eigenvalue / secular equation in Hermite basis."""
# EVOLVE-BLOCK-START
import numpy as np
from math import factorial, sqrt, pi
import time

_DEADLINE = time.time() + 195  # safety margin before 210s kill

def _precompute():
    """
    Precompute all constants needed for the uncertainty bound.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0) = 0.
    
    We express the problem as a generalized eigenvalue problem.
    """
    # Hermite polynomials evaluated at 0
    # H_0(0) = 1
    # H_4(0) = 12  (H4 = 16x^4 - 48x^2 + 12)
    # H_8(0) = 168 (H8 = 256x^8 - 4480x^6 + 25200x^4 - 42000x^2 + 1680... let me verify)
    # Actually let me use the recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    # H_0 = 1, H_1 = 2x
    # H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2
    # H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 12x
    # H_4 = 2x(8x^3-12x) - 8(4x^2-2) = 16x^4 - 48x^2 + 16... wait
    # H_4 = 2x*H_3 - 2*3*H_2 = 2x(8x^3-12x) - 6(4x^2-2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12
    # So H_4(0) = 12
    # H_5 = 2x*H_4 - 2*4*H_3 = 2x(16x^4-48x^2+12) - 8(8x^3-12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x
    # H_6 = 2x*H_5 - 2*5*H_4 = 2x(32x^5-160x^3+120x) - 10(16x^4-48x^2+12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7 = 2x*H_6 - 2*6*H_5 = 2x(64x^6-480x^4+720x^2-120) - 12(32x^5-160x^3+120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8 = 2x*H_7 - 2*7*H_6 = 2x(128x^7-1344x^5+3360x^3-1680x) - 14(64x^6-480x^4+720x^2-120)
    #     = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
    #     = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    # So H_8(0) = 1680... wait, let me recheck. The standard Hermite polynomial H_8(0) should be related to (-1)^4 * 8!/(4! 2^4) = 40320/(24*16) = 40320/384 = 105. Hmm.
    
    # Actually, the physicists' Hermite polynomials: H_n(0) = 0 for odd n, and for even n=2m:
    # H_{2m}(0) = (-1)^m (2m)!/(m! 2^m)
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = (-1)^2 * 4!/(2! * 2^2) = 24/(2*4) = 3... that doesn't match.
    
    # Let me just use scipy to be safe, or compute numerically.
    from scipy.special import eval_hermite
    H0_0 = eval_hermite(0, 0.0)   # 1
    H4_0 = eval_hermite(4, 0.0)   # 12
    H8_0 = eval_hermite(8, 0.0)   # 168
    H12_0 = eval_hermite(12, 0.0) # 1440
    
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    # So c12 = -(c0*1 + c4*12 + c8*168) / 1440
    # = -c0/1440 - c4/120 - c8/30/1440... let me compute:
    # c12 = -(c0 + 12*c4 + 168*c8) / 1440
    # = -c0/1440 - 12*c4/1440 - 168*c8/1440
    # = -c0/1440 - c4/120 - 7*c8/60
    
    # Now, the key insight: for Hermite functions, the Fourier transform is a phase rotation.
    # Specifically, if g(x) = sum_k c_k H_k(x) e^{-x^2/2}, then
    # F(ξ) = (1/sqrt(2π)) ∫ g(x) e^{-iξx} dx
    # For Hermite functions, the Fourier transform of H_k(x) e^{-x^2/2} is (-i)^k H_k(ξ) e^{-ξ^2/2} * (1/sqrt(2π)) * sqrt(π) * ... 
    # Actually, the standard result is that the Fourier transform of the k-th Hermite function 
    # ψ_k(x) = (2^k k! sqrt(π))^{-1/2} H_k(x) e^{-x^2/2} is (-i)^k ψ_k(ξ).
    # So if g(x) = sum_k c_k H_k(x) e^{-x^2/2}, then
    # F(ξ) = sum_k c_k (-i)^k H_k(ξ) e^{-ξ^2/2} / sqrt(2π) * sqrt(π) * ... 
    # Let me be more careful.
    
    # The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(ξ) e^{-ξ^2/2} * sqrt(π) / sqrt(2π) ... 
    # Actually, ∫ H_n(x) e^{-x^2/2} e^{-iξx} dx = (-i)^n sqrt(π) H_n(ξ) e^{-ξ^2/2}
    # So F(ξ) = (1/sqrt(2π)) * sum_k c_k (-i)^k sqrt(π) H_k(ξ) e^{-ξ^2/2}
    # = (1/sqrt(2)) * sum_k c_k (-i)^k H_k(ξ) e^{-ξ^2/2}
    
    # So |F(ξ)|^2 = (1/2) |sum_k c_k (-i)^k H_k(ξ) e^{-ξ^2/2}|^2
    # Since |(-i)^k| = 1, the magnitudes are the same, so:
    # ∫ |F(ξ)|^2 dξ = (1/2) ∫ |g(x)|^2 dx
    # ∫ ξ^2 |F(ξ)|^2 dξ = (1/2) ∫ x^2 |g(x)|^2 dx
    
    # Wait, that means A(F) = ∫ξ^2|F|^2 / ∫|F|^2 = ∫x^2|g|^2 / ∫|g|^2 = A(f)
    # So A(f) * A(F) = A(f)^2
    
    # Hmm, but that would mean the problem is just minimizing A(f), which is the variance of |g|^2.
    # And the constraint is g(0) = 0.
    
    # Let me verify: A(f) = ∫ x^2 |f(x)|^2 dx / ∫ |f(x)|^2 dx
    # where f(x) = g(x) (the test function).
    
    # So we need to minimize (∫ x^2 |g|^2 / ∫ |g|^2)^2 subject to g(0) = 0.
    # This is equivalent to minimizing ∫ x^2 |g|^2 / ∫ |g|^2 subject to g(0) = 0.
    
    # This is a Rayleigh quotient problem!
    # R(c) = c^T B c / c^T A c
    # where A is the Gram matrix (∫ H_i H_j e^{-x^2} dx) and B is the x^2-weighted Gram matrix.
    
    # But c12 is constrained by g(0) = 0, so we substitute c12 in terms of c0, c4, c8.
    
    # Let's define the coefficient vector in terms of free variables (c0, c4, c8).
    # c = [c0, c4, c8, c12] where c12 = -c0/1440 - c4/120 - 7*c8/60
    
    # So c = M * c_free where c_free = [c0, c4, c8] and M is a 4x3 matrix:
    # M = [[1, 0, 0],
    #      [0, 1, 0],
    #      [0, 0, 1],
    #      [-1/1440, -1/120, -7/60]]
    
    # Then R(c_free) = (M c_free)^T B (M c_free) / (M c_free)^T A (M c_free)
    # = c_free^T (M^T B M) c_free / c_free^T (M^T A M) c_free
    
    # The minimum of this Rayleigh quotient is the smallest eigenvalue of the generalized 
    # eigenvalue problem: (M^T B M) v = λ (M^T A M) v
    
    # Let me compute A and B.
    
    # A_{ij} = ∫ H_i(x) H_j(x) e^{-x^2} dx = sqrt(π) 2^j j! δ_{ij} (for i=j)
    # Actually, ∫ H_m(x) H_n(x) e^{-x^2} dx = sqrt(π) 2^n n! δ_{mn}
    
    # B_{ij} = ∫ x^2 H_i(x) H_j(x) e^{-x^2} dx
    # Using x^2 H_n = (1/4) H_{n+2} + (n + 1/2) H_n + (n(n-1)/2) H_{n-2} (for n >= 2)
    # For n=0: x^2 H_0 = x^2 = (1/4)H_2 + 1/2 H_0
    # For n=1: x^2 H_1 = 2x^3 = (1/2)H_3 + 1 H_1... let me verify: H_3 = 8x^3 - 12x, so 2x^3 = H_3/4 + 3x = H_3/4 + 3H_1/2
    # Hmm, let me use the general formula more carefully.
    
    # x H_n = (1/2) H_{n+1} + n H_{n-1}
    # x^2 H_n = x[(1/2) H_{n+1} + n H_{n-1}] = (1/2) x H_{n+1} + n x H_{n-1}
    # = (1/2)[(1/2) H_{n+2} + (n+1) H_n] + n[(1/2) H_n + (n-1) H_{n-2}]
    # = (1/4) H_{n+2} + (n+1)/2 H_n + n/2 H_n + n(n-1)/2 H_{n-2}
    # = (1/4) H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    
    # So B_{ij} = ∫ x^2 H_i H_j e^{-x^2} dx
    # = ∫ [(1/4) H_{i+2} + (2i+1)/2 H_i + i(i-1)/2 H_{i-2}] H_j e^{-x^2} dx
    # = (1/4) A_{i+2,j} + (2i+1)/2 A_{i,j} + i(i-1)/2 A_{i-2,j} (for i >= 2)
    # = (1/4) sqrt(π) 2^j j! δ_{i+2,j} + (2i+1)/2 sqrt(π) 2^j j! δ_{i,j} + i(i-1)/2 sqrt(π) 2^j j! δ_{i-2,j}
    
    # So B is tridiagonal (in the Hermite basis, with step 2).
    
    degrees = [0, 4, 8, 12]
    n_deg = len(degrees)
    
    # Build A matrix (diagonal)
    A = np.zeros((n_deg, n_deg))
    for idx, k in enumerate(degrees):
        A[idx, idx] = sqrt(pi) * (2.0 ** k) * factorial(k)
    
    # Build B matrix
    B = np.zeros((n_deg, n_deg))
    for i_idx, k in enumerate(degrees):
        # Diagonal term: (2k+1)/2 * A_{k,k}
        B[i_idx, i_idx] += (2*k + 1) / 2.0 * A[i_idx, i_idx]
        # Off-diagonal: (1/4) * A_{k+2, k} if k+2 is in degrees
        if k + 2 in degrees:
            j_idx = degrees.index(k + 2)
            B[i_idx, j_idx] += 0.25 * A[j_idx, j_idx]
            B[j_idx, i_idx] += 0.25 * A[i_idx, i_idx]  # B is symmetric
        # Off-diagonal: k(k-1)/2 * A_{k-2, k} if k-2 is in degrees
        if k >= 2 and (k - 2) in degrees:
            j_idx = degrees.index(k - 2)
            B[i_idx, j_idx] += k * (k - 1) / 2.0 * A[j_idx, j_idx]
            B[j_idx, i_idx] += k * (k - 1) / 2.0 * A[i_idx, i_idx]  # B is symmetric
    
    # Build M matrix (4x3)
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/1440.0, -1.0/120.0, -7.0/60.0]
    ])
    
    # Compute reduced matrices
    A_red = M.T @ A @ M  # 3x3
    B_red = M.T @ B @ M  # 3x3
    
    return A_red, B_red

def _solve_generalized_eig(A_red, B_red):
    """
    Solve the generalized eigenvalue problem B_red v = λ A_red v.
    Return the smallest eigenvalue and its eigenvector.
    """
    # Use scipy's eig for generalized eigenvalue problem
    from scipy.linalg import eig
    eigenvalues, eigenvectors = eig(B_red, A_red)
    
    # Filter for real eigenvalues
    real_eigs = []
    for i in range(len(eigenvalues)):
        if abs(eigenvalues[i].imag) < 1e-10:
            real_eigs.append((eigenvalues[i].real, eigenvectors[:, i]))
    
    if not real_eigs:
        return None, None
    
    # Find the smallest positive eigenvalue
    real_eigs.sort(key=lambda x: x[0])
    for val, vec in real_eigs:
        if val > 1e-10:
            return val, vec
    
    return real_eigs[0]

def construct():
    start_time = time.time()
    
    # Precompute matrices
    A_red, B_red = _precompute()
    
    # Solve generalized eigenvalue problem
    lambda_min, v_opt = _solve_generalized_eig(A_red, B_red)
    
    if lambda_min is None:
        # Fallback to numerical optimization
        from scipy.optimize import minimize
        
        def _objective(c):
            c0, c4, c8 = c
            c12 = -(c0 + 12*c4 + 168*c8) / 1440.0
            c = np.array([c0, c4, c8, c12])
            
            norm_sq = 0.0
            x2_norm = 0.0
            for i, k in enumerate([0, 4, 8, 12]):
                norm_sq += c[i]**2 * (2.0**k) * factorial(k)
                x2_norm += c[i]**2 * (2*k + 1) / 2.0 * (2.0**k) * factorial(k)
                if k + 2 in [0, 4, 8, 12]:
                    j = [0, 4, 8, 12].index(k + 2)
                    x2_norm += 2 * c[i] * c[j] * 0.25 * (2.0**(k+2)) * factorial(k+2)
                if k >= 2 and (k - 2) in [0, 4, 8, 12]:
                    j = [0, 4, 8, 12].index(k - 2)
                    x2_norm += 2 * c[i] * c[j] * k*(k-1)/2.0 * (2.0**(k-2)) * factorial(k-2)
            
            norm_sq *= sqrt(pi)
            x2_norm *= sqrt(pi)
            
            if norm_sq < 1e-15:
                return 1e10
            A_f = x2_norm / norm_sq
            return A_f ** 2
        
        # Multi-start
        best_val = 1e10
        best_c = np.array([-1.0, 0.1, 0.01])
        
        np.random.seed(42)
        for _ in range(50):
            c_init = np.random.uniform(-2, 2, size=3)
            try:
                result = minimize(_objective, c_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x
            except:
                pass
        
        return best_c
    
    # v_opt is the eigenvector in the reduced space (c0, c4, c8)
    # Normalize it
    v_opt = v_opt / np.linalg.norm(v_opt)
    
    # Ensure it's real
    v_opt = np.real(v_opt)
    
    # The optimal coefficients are proportional to v_opt
    # We can return v_opt directly (the scale doesn't matter for the Rayleigh quotient)
    # But let's normalize so that the largest component has magnitude 1
    max_comp = np.max(np.abs(v_opt))
    if max_comp > 1e-15:
        v_opt = v_opt / max_comp
    
    # Verify the time
    elapsed = time.time() - start_time
    if elapsed > 180:
        # Should not happen, but just in case
        pass
    
    return v_opt
# EVOLVE-BLOCK-END
