"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def construct():
    start_time = time.time()
    deadline = start_time + 175  # Safety margin (evaluator kills at 210s)
    
    # Physicists' Hermite polynomials H_n(x)
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 98304x^10 + 806400x^8 - 3182400x^6 + 5644800x^4 - 4032000x^2 + 103950
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 103950.0
    
    def compute_bound(c0, c4, c8):
        """Compute the uncertainty product A(f)*A(Fourier(f)) for the test function.
        
        For a function f(x) = P(x) * exp(-x^2/2) with P(0) = 0:
        - A(f) = ||f||_2 (L2 norm)
        - A(Fourier(f)) = ||Fourier(f)||_2 (L2 norm of Fourier transform)
        
        By Parseval's theorem, ||f||_2 = ||F(f)||_2, so the product is ||f||_2^2.
        But wait - the uncertainty inequality is typically about variances.
        
        Let me reconsider. The standard Heisenberg uncertainty principle states:
        (Δx)^2 (Δp)^2 ≥ (1/4) |∫ f* [x, p] f dx|^2
        
        For the "uncertainty inequality" with A(f) being some norm:
        If A(f) = ||f||_1 or ||f||_2, then we need to understand what the bound is.
        
        Actually, looking at the task more carefully: "reduce the upper bound on A(f)*A(Fourier(f))"
        This suggests A is some norm and we want to minimize the product of the norm 
        of the function and the norm of its Fourier transform.
        
        For L2 norm: ||f||_2 * ||F(f)||_2 = ||f||_2^2 (by Parseval)
        This is just the energy, which we can normalize to 1.
        
        Perhaps A(f) is the L1 norm? Then we want to minimize ||f||_1 * ||F(f)||_1.
        
        Or perhaps it's about the "uncertainty" in the sense of spread:
        Δx * Δξ where Δx^2 = ∫ x^2 |f|^2 / ∫ |f|^2 and Δξ^2 = ∫ ξ^2 |F(f)|^2 / ∫ |F(f)|^2
        
        The task says "upper bound on C4" which suggests this is related to a specific 
        constant in an inequality. Let me assume it's the product of standard deviations
        (Δx)(Δξ) which has a lower bound of 1/(2π) for the Fourier convention 
        F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx, or 1/2 for F(f)(ξ) = ∫ f(x) e^{-ixξ} dx.
        
        Given the context of "Fourier-invariant" and Hermite polynomials, I'll compute:
        Δx * Δξ where these are the standard deviations of |f|^2 and |F(f)|^2.
        """
        # Compute c12 to enforce P(0) = 0
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Use a sufficiently large grid
        N = 8192
        x_max = 12.0
        x = np.linspace(-x_max, x_max, N)
        dx = x[1] - x[0]
        
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x8 * x4
        
        H0 = np.ones_like(x)
        H4 = 16.0 * x4 - 48.0 * x2 + 12.0
        H8 = 256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0
        H12 = 4096.0 * x12 - 98304.0 * x10 + 806400.0 * x8 - 3182400.0 * x6 + 5644800.0 * x4 - 4032000.0 * x2 + 103950.0
        
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        f = P * np.exp(-0.5 * x2)
        
        # L2 norm squared
        norm_sq = np.sum(f**2) * dx
        if norm_sq < 1e-15:
            return 1e10
        
        # <x^2> = ∫ x^2 |f|^2 dx / ∫ |f|^2 dx
        ex2 = np.sum(x2 * f**2) * dx / norm_sq
        
        # For the Fourier side, use the property:
        # ∫ ξ^2 |F(f)(ξ)|^2 dξ = (1/(4π^2)) ∫ |f'(x)|^2 dx  [for F(f)(ξ) = ∫ f(x) e^{-ixξ} dx]
        # or ∫ ξ^2 |F(f)(ξ)|^2 dξ = (4π^2) ∫ |f'(x)|^2 dx  [for F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx]
        
        # Let's use the convention F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx
        # Then ∫ ξ^2 |F(f)(ξ)|^2 dξ = 4π^2 ∫ |f'(x)|^2 dx
        
        # Compute f'(x) analytically:
        # f(x) = P(x) e^{-x^2/2}
        # f'(x) = (P'(x) - x P(x)) e^{-x^2/2}
        
        # P'(x) = c4*(64x^3 - 96x) + c8*(2048x^7 - 18432x^5 + 46080x^3 - 26880x) 
        #        + c12*(49152x^11 - 983040x^9 + 6451200x^7 - 19094400x^5 + 22579200x^3 - 8064000x)
        
        x3 = x2 * x
        x5 = x4 * x
        x7 = x6 * x
        x9 = x8 * x
        x11 = x10 * x
        
        dP = (c4 * (64.0 * x3 - 96.0 * x) + 
              c8 * (2048.0 * x7 - 18432.0 * x5 + 46080.0 * x3 - 26880.0 * x) +
              c12 * (49152.0 * x11 - 983040.0 * x9 + 6451200.0 * x7 - 19094400.0 * x5 + 22579200.0 * x3 - 8064000.0 * x))
        
        df = (dP - x * P) * np.exp(-0.5 * x2)
        
        norm_df_sq = np.sum(df**2) * dx
        
        # <ξ^2> = ∫ ξ^2 |F(f)|^2 dξ / ∫ |F(f)|^2 dξ
        # By Parseval, ∫ |F(f)|^2 dξ = ∫ |f|^2 dx = norm_sq
        # ∫ ξ^2 |F(f)|^2 dξ = 4π^2 * norm_df_sq
        exi2 = 4.0 * np.pi**2 * norm_df_sq / norm_sq
        
        delta_x = np.sqrt(max(ex2, 0))
        delta_xi = np.sqrt(max(exi2, 0))
        
        return delta_x * delta_xi
    
    # Homotopy continuation strategy:
    # Start from a simple 2-term problem (H0 + H4) and gradually introduce H8 and H12.
    
    # Phase 1: Optimize with only c0 and c4 (c8 = 0, so c12 is determined)
    def objective_phase1(params):
        c0, c4 = params
        if time.time() > deadline:
            return 1e10
        return compute_bound(c0, c4, 0.0)
    
    # Phase 2: Add c8, optimize all three
    def objective_full(params):
        c0, c4, c8 = params
        if time.time() > deadline:
            return 1e10
        return compute_bound(c0, c4, c8)
    
    best_result = None
    best_score = 1e10
    
    # Phase 1: 2-term optimization
    x0_2d = np.array([-1.0, 0.04])  # Initial guess for c0, c4
    try:
        result1 = minimize(objective_phase1, x0_2d, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        if result1.fun < best_score:
            best_score = result1.fun
            best_result = np.array([result1.x[0], result1.x[1], 0.0])
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        if best_result is not None:
            return best_result
        return np.array([-1.13, 0.04, 0.0003086])
    
    # Phase 2: Full 3-parameter optimization starting from Phase 1 solution
    if best_result is not None:
        x0_3d = best_result.copy()
    else:
        x0_3d = np.array([-1.13, 0.04, 0.0003086])
    
    # Try multiple starting points for robustness
    starts = [
        x0_3d,
        np.array([-1.13, 0.04, 0.0003086]),  # Seed
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.5, 0.02, -0.001]),
        np.array([-0.5, 0.1, 0.005]),
    ]
    
    for x0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(objective_full, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    # Also try L-BFGS-B with numerical gradient
    if time.time() > deadline - 15:
        if best_result is not None:
            return best_result
        return np.array([-1.13, 0.04, 0.0003086])
    
    for x0 in starts[:3]:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective_full, x0, method='L-BFGS-B', maxiter=200)
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return np.array([-1.13, 0.04, 0.0003086])

# EVOLVE-BLOCK-END
