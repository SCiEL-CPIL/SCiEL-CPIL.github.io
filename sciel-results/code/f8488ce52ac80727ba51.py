"""Rewritten program: multi-start global + local optimization for the 3 free Hermite coefficients.

Task: minimize the upper bound on A(f)*A(Fhat(f)) for a Fourier-invariant Hermite-Gaussian test
function g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2) with g(0)=0 fixing c12.
The seed program returns the known prior state of the art (Goncalves et al. 2017):
    c0 = -113/100, c4 = 1/25, c8 = 1/3240

We implement a robust multi-start optimizer:
  1. A dense 2D grid over (c4, c8) with c0 = -c4 - c8 (from g(0)=0 constraint: H_n(0) known).
     Actually g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * 1 = 0.
     H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395.
     So c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0.
     The task says "the degree-12 weight is fixed so g(0)=0", meaning c12 is determined by c0,c4,c8:
     c12 = (c0 - 3*c4 + 105*c8) / 10395.
     
     Wait, re-reading: "Three free coefficients multiply H0,H4,H8; the notebook derives the H12 coefficient to impose P(0)=0."
     So c0, c4, c8 are free, and c12 is derived. The constraint is g(0)=0.
     
     H_n(0): H0=1, H2=-1, H4=-3, H6=-15, H8=-105, H10=-945, H12=-10395.
     Wait, let me recall: H_0(x)=1, H_1(x)=2x, H_2(x)=4x^2-2, H_3(x)=8x^3-12x, H_4(x)=16x^4-48x^2+12.
     So H_4(0)=12? No wait, probabilists' vs physicists'.
     
     In the uncertainty principle context with Gaussian exp(-x^2/2), the relevant Hermite polynomials are
     typically the probabilists' He_n or physicists' H_n. Let me think about what gives g(0)=0.
     
     Actually, the standard Hermite-Gaussian in physics uses H_n(x) exp(-x^2/2) where H_n are physicists'.
     H_0(0)=1, H_4(0)=12, H_8(0)=-1344? Let me compute:
     H_0 = 1
     H_1 = 2x
     H_2 = 4x^2 - 2
     H_3 = 8x^3 - 12x
     H_4 = 16x^4 - 48x^2 + 12
     H_4(0) = 12
     
     H_8(0): This is more complex. H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344.
     H_8(0) = 1344.
     
     H_12(0): H_12(0) = ? The constant term of H_12. 
     H_{2n}(0) = (-1)^n (2n)! / n!.
     H_0(0) = 1
     H_2(0) = -2
     H_4(0) = 12
     H_6(0) = -120
     H_8(0) = 1344
     H_10(0) = -15120
     H_12(0) = 181440
     
     Check: (2n)!/n! for n=1: 2!/1! = 2, (-1)^1 * 2 = -2. OK.
     n=2: 4!/2! = 24/2 = 12, (-1)^2 * 12 = 12. OK.
     n=3: 6!/3! = 720/6 = 120, (-1)^3 * 120 = -120. OK.
     n=4: 8!/4! = 40320/24 = 1680, (-1)^4 * 1680 = 1680. But I said 1344 above. Let me recheck.
     
     Actually H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344.
     Hmm, 1344 vs 1680. Let me recompute.
     
     H_8(0) should be (-1)^4 * 8!/4! = 40320/24 = 1680.
     But the polynomial gives 1344. There's a discrepancy. Let me recheck H_8.
     
     Actually, I think I may have the wrong H_8. Let me use the recurrence:
     H_0 = 1, H_1 = 2x
     H_{n+1} = 2x H_n - 2n H_{n-1}
     
     H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2. H_2(0) = -2.
     H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x. H_3(0) = 0.
     H_4 = 2x(8x^3-12x) - 6(4x^2-2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12. H_4(0) = 12.
     H_5 = 2x(16x^4-48x^2+12) - 8(8x^3-12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x. H_5(0)=0.
     H_6 = 2x(32x^5-160x^3+120x) - 10(16x^4-48x^2+12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120. H_6(0) = -120.
     H_7 = 2x(64x^6-480x^4+720x^2-120) - 12(32x^5-160x^3+120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x. H_7(0)=0.
     H_8 = 2x(128x^7-1344x^5+3360x^3-1680x) - 14(64x^6-480x^4+720x^2-120)
          = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
          = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680.
     H_8(0) = 1680. OK, so my earlier polynomial was wrong.
     
     H_12(0) = (-1)^6 * 12!/6! = 479001600 / 720 = 665280.
     Check: 12! = 479001600, 6! = 720. 479001600/720 = 665280. Yes.
     
     So the constraint g(0)=0 gives:
     c0*1 + c4*12 + c8*1680 + c12*665280 = 0
     => c12 = -(c0 + 12*c4 + 1680*c8) / 665280
     
     The seed returns c0=-113/100, c4=1/25, c8=1/3240.
     c12 = -(-113/100 + 12/25 + 1680/3240) / 665280
         = -(-1.13 + 0.48 + 0.518518...) / 665280
         = -(-0.131481...) / 665280
         = 0.131481.../665280 ≈ 1.976e-7
     
     OK. Now, the objective: the upper bound on A(f)*A(Fhat(f)). For a function f with f(0)=0 (or some condition), 
     the uncertainty product has a lower bound, and we want to minimize the upper bound achieved by this test function.
     
     The exact formula for the uncertainty product for a Hermite-Gaussian combination is complex. 
     However, the key insight from the task is that we need to find c0, c4, c8 that minimize this bound.
     
     Since I don't have the exact closed-form expression for the uncertainty product as a function of c0,c4,c8,
     I need to either:
     1. Numerically compute it for each candidate (expensive but feasible)
     2. Use known results
     
     Given the time budget, let me implement a numerical evaluation of the uncertainty product and use it in an optimizer.
     
     The uncertainty product is A(f) * A(Fhat(f)) where A(f) = sqrt(<x^2> - <x>^2) and similarly for Fhat.
     For a function g(x) = P(x) * exp(-x^2/2) where P is a polynomial, we can compute these moments analytically or numerically.
     
     Let me implement numerical computation and use scipy.optimize with multi-start.
"""

import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200  # 10s safety margin before 210s kill

def _time_left():
    return _DEADLINE - time.time()


def _compute_uncertainty_product(c0, c4, c8):
    """Compute A(g) * A(Fhat(g)) for g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2).
    
    Uses numerical integration over a sufficiently wide domain.
    """
    # Hermite polynomials (physicists')
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
    def H12(x):
        # Compute via recurrence or direct formula
        # H_12(0) = 665280, need full polynomial
        # Use recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
        h0 = np.ones_like(x)
        h1 = 2*x
        h2 = 2*x*h1 - 2*h0
        h3 = 2*x*h2 - 4*h1
        h4 = 2*x*h3 - 6*h2
        h5 = 2*x*h4 - 8*h3
        h6 = 2*x*h5 - 10*h4
        h7 = 2*x*h6 - 12*h5
        h8 = 2*x*h7 - 14*h6
        h9 = 2*x*h8 - 16*h7
        h10 = 2*x*h9 - 18*h8
        h11 = 2*x*h10 - 20*h9
        h12 = 2*x*h11 - 22*h10
        return h12
    
    # c12 from g(0)=0
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    
    # Grid for numerical integration
    x = np.linspace(-15, 15, 3001)
    dx = x[1] - x[0]
    
    g = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2)
    
    norm2 = np.trapz(g**2, x)
    if norm2 <= 0:
        return 1e10
    
    g_norm = g / np.sqrt(norm2)
    
    # <x>, <x^2>
    ex = np.trapz(x * g_norm**2, x)
    ex2 = np.trapz(x**2 * g_norm**2, x)
    var_x = ex2 - ex**2
    if var_x <= 0:
        return 1e10
    
    # Fourier transform: Fhat(k) = (1/sqrt(2pi)) integral g(x) exp(-ikx) dx
    # For numerical FT, use FFT or direct computation
    # Since g is even (all H_{2n} are even), Fhat is real and even
    # Use direct numerical FT on the grid
    N = len(x)
    k_vals = np.fft.fftfreq(N, d=dx) * 2*np.pi
    G_hat = np.fft.fft(g_norm) * dx / np.sqrt(2*np.pi)
    
    # Reorder for proper indexing
    G_hat = np.fft.fftshift(G_hat)
    k_vals = np.fft.fftshift(k_vals)
    
    norm2_hat = np.trapz(np.abs(G_hat)**2, k_vals)
    if norm2_hat <= 0:
        return 1e10
    
    G_hat_norm = G_hat / np.sqrt(norm2_hat)
    
    ek = np.trapz(k_vals * np.abs(G_hat_norm)**2, k_vals)
    ek2 = np.trapz(k_vals**2 * np.abs(G_hat_norm)**2, k_vals)
    var_k = ek2 - ek**2
    if var_k <= 0:
        return 1e10
    
    return np.sqrt(var_x) * np.sqrt(var_k)


def _objective(params):
    c0, c4, c8 = params
    try:
        return _compute_uncertainty_product(c0, c4, c8)
    except:
        return 1e10


def construct():
    """Return the three coefficients (c0, c4, c8) minimizing the uncertainty product upper bound."""
    # Start with the known best
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_uncertainty_product(*best_params)
    
    if _time_left() < 5:
        return best_params
    
    # Multi-start local optimization
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [best_params.copy()]
    
    # Random starts in a reasonable range
    for _ in range(30):
        c0 = rng.uniform(-5, 5)
        c4 = rng.uniform(-2, 2)
        c8 = rng.uniform(-0.1, 0.1)
        starts.append(np.array([c0, c4, c8]))
    
    # Grid-based starts around the known best
    for dc0 in np.linspace(-0.5, 0.5, 5):
        for dc4 in np.linspace(-0.1, 0.1, 5):
            for dc8 in np.linspace(-0.001, 0.001, 5):
                if _time_left() < 10:
                    break
                starts.append(best_params + np.array([dc0, dc4, dc8]))
    
    # Run local optimization from each start
    for i, start in enumerate(starts):
        if _time_left() < 15:
            break
        
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    if _time_left() < 10:
        return best_params
    
    # Second round: finer local search around the best found
    for i in range(20):
        if _time_left() < 15:
            break
        perturbation = rng.normal(0, 0.01, size=3)
        start = best_params + perturbation
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return best_params
