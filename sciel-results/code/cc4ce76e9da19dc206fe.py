import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_bound(c):
    """
    c: array [c0, c4, c8, c12]
    Returns the uncertainty product Δx * Δp for f(x) = sum c_n H_n(x) exp(-x^2/2)
    """
    degrees = np.array([0, 4, 8, 12])
    coeffs = c
    
    # Normalization: <f|f> = sum c_n^2 * <H_n|H_n> = sum c_n^2 (since H_n are orthonormal with weight exp(-x^2))
    # Actually with the convention f = sum c_n H_n(x) e^{-x^2/2}, the inner product is:
    # <f|f> = integral |f|^2 dx = sum_{n,m} c_n c_m integral H_n H_m e^{-x^2} dx = sum c_n^2 sqrt(pi) 2^n n!
    # Let's use normalized basis: phi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2}
    # Then f = sum a_n phi_n, and <f|f> = sum a_n^2
    # But the task says "coefficients multiply H0,H4,H8" so let's work with the raw coefficients.
    # 
    # For f(x) = sum_n c_n H_n(x) e^{-x^2/2}:
    # <f|f> = sum_{n,m} c_n c_m sqrt(pi) 2^n n! delta_nm = sqrt(pi) sum_n c_n^2 2^n n!
    #
    # <x f | f> = sum_{n,m} c_n c_m <H_n x H_m> integral e^{-x^2} dx
    # x H_m = (1/2) H_{m+1} + m H_{m-1}  (physicist's convention)
    # So <H_n | x | H_m> = (1/2) sqrt(2^{m+1}(m+1)!) / (sqrt(2^m m!)) delta_{n,m+1} + m sqrt(2^{m-1}(m-1)!)/sqrt(2^m m!) delta_{n,m-1}
    # = (1/2) sqrt(2(m+1)) delta_{n,m+1} + m / sqrt(2) delta_{n,m-1}
    # Wait, let me be more careful.
    #
    # The orthonormal Hermite functions: psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    # Then <psi_n | x | psi_m> = (1/sqrt(2)) (sqrt(m+1) delta_{n,m+1} + sqrt(m) delta_{n,m-1})
    #
    # If f = sum a_n psi_n, then:
    # <f|f> = sum a_n^2
    # <f|x|f> = sum_{n,m} a_n a_m <psi_n|x|psi_m> = (1/sqrt(2)) sum_n a_n a_{n+1} sqrt(n+1) + (1/sqrt(2)) sum_n a_n a_{n-1} sqrt(n)
    # = (1/sqrt(2)) sum_n a_n a_{n+1} (sqrt(n+1) + sqrt(n))  ... no wait
    # = (1/sqrt(2)) [sum_n a_{n+1} a_n sqrt(n+1) + sum_n a_n a_{n-1} sqrt(n)]
    # Let k = n+1 in first sum: sum_k a_k a_{k-1} sqrt(k)
    # Let k = n-1 in second sum: sum_k a_{k+1} a_k sqrt(k+1)
    # So <f|x|f> = (1/sqrt(2)) sum_k a_k a_{k-1} [sqrt(k) + sqrt(k+1)]  ... this doesn't seem right either.
    #
    # Actually: <psi_n|x|psi_m> = (1/sqrt(2))(sqrt(m+1) delta_{n,m+1} + sqrt(m) delta_{n,m-1})
    # <f|x|f> = sum_{n,m} a_n a_m <psi_n|x|psi_m>
    # = (1/sqrt(2)) sum_{n,m} a_n a_m [sqrt(m+1) delta_{n,m+1} + sqrt(m) delta_{n,m-1}]
    # = (1/sqrt(2)) [sum_m a_{m+1} a_m sqrt(m+1) + sum_m a_m a_{m-1} sqrt(m)]
    # = (1/sqrt(2)) [sum_k a_{k} a_{k-1} sqrt(k) + sum_k a_{k+1} a_k sqrt(k+1)]
    # = (1/sqrt(2)) sum_k a_k a_{k-1} [sqrt(k) + sqrt(k+1)]  ... no, the indices are off.
    # First term: sum_m a_{m+1} a_m sqrt(m+1). Let k=m+1: sum_k a_k a_{k-1} sqrt(k)
    # Second term: sum_m a_m a_{m-1} sqrt(m). Let k=m: sum_k a_k a_{k-1} sqrt(k)
    # So <f|x|f> = (1/sqrt(2)) sum_k a_k a_{k-1} [sqrt(k) + sqrt(k)] = (1/sqrt(2)) sum_k a_k a_{k-1} 2 sqrt(k) = sqrt(2) sum_k a_k a_{k-1} sqrt(k)
    #
    # Similarly <f|x^2|f> = <f|x|x|f>
    # x^2 in Hermite basis: x^2 psi_m = (1/2)(sqrt(m+1) psi_{m+2} + 2m+1) psi_m + (1/2) sqrt(m(m-1)) psi_{m-2} ... 
    # Actually: x^2 = (1/2)(a+a†)^2 = (1/2)(a^2 + a†2 + aa† + a†a) = (1/2)(a^2 + a†2 + 2a†a + 1)
    # So <psi_n|x^2|psi_m> = (1/2)sqrt((m+1)(m+2)) delta_{n,m+2} + (m+1/2) delta_{n,m} + (1/2)sqrt(m(m-1)) delta_{n,m-2}
    #
    # For momentum: p = -i d/dx, and in the Hermite function basis:
    # <psi_n|p|psi_m> = i/sqrt(2) (sqrt(m) delta_{n,m-1} - sqrt(m+1) delta_{n,m+1})
    # <psi_n|p^2|psi_m> = (m+1/2) delta_{n,m}  (since p^2 = -d^2/dx^2, and psi_n are eigenfunctions of the harmonic oscillator Hamiltonian H = p^2 + x^2/2 with eigenvalue n+1/2, so p^2 = H - x^2/2... this is getting complex)
    #
    # Actually, for the harmonic oscillator: H = p^2/2 + x^2/2 (with m=1, omega=1)
    # Eigenvalues: E_n = n + 1/2
    # <psi_n|p^2|psi_n> = 2 E_n - <psi_n|x^2|psi_n> = 2(n+1/2) - (n+1/2) = n+1/2
    # Wait, that's not right either. Let me just use: <psi_n|p^2|psi_m> = (n+1/2) delta_{nm} is NOT correct in general.
    #
    # The cleanest approach: use the fact that for the orthonormal Hermite functions psi_n:
    # <psi_n|x|psi_m> = (1/sqrt(2))(sqrt(m+1) delta_{n,m+1} + sqrt(m) delta_{n,m-1})
    # <psi_n|p|psi_m> = i/sqrt(2)(sqrt(m) delta_{n,m-1} - sqrt(m+1) delta_{n,m+1})
    #
    # So for f = sum a_n psi_n:
    # <x> = sum_{n,m} a_n a_m <psi_n|x|psi_m> = (1/sqrt(2)) sum_m a_{m+1} a_m sqrt(m+1) + (1/sqrt(2)) sum_m a_m a_{m-1} sqrt(m)
    # <x^2> = sum_{n,m} a_n a_m <psi_n|x^2|psi_m>
    # <p> = sum_{n,m} a_n a_m <psi_n|p|psi_m> = i/sqrt(2) sum_m a_m a_{m-1} sqrt(m) - i/sqrt(2) sum_m a_{m+1} a_m sqrt(m+1)
    # <p^2> = sum_{n,m} a_n a_m <psi_n|p^2|psi_m>
    #
    # For <x^2> and <p^2>, we can use the number operator approach or direct computation.
    #
    # Given the complexity, let me just use a numerical approach on a grid.
    
    # Use a numerical grid to compute the uncertainty product
    # This is slower but more reliable
    
    # Define the function on a grid
    x = np.linspace(-8, 8, 4096)
    dx = x[1] - x[0]
    
    # Compute H_n(x) for n = 0, 4, 8, 12
    # Use recurrence: H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
    H = np.zeros((13, len(x)))
    H[0] = 1.0
    H[1] = 2.0 * x
    for n in range(2, 13):
        H[n] = 2.0 * x * H[n-1] - 2.0 * n * H[n-2]
    
    # f(x) = sum c_n H_n(x) e^{-x^2/2}
    gauss = np.exp(-x**2 / 2.0)
    f = np.zeros_like(x)
    for i, deg in enumerate([0, 4, 8, 12]):
        f += coeffs[i] * H[deg] * gauss
    
    # Compute |f|^2
    abs_f_sq = f**2  # f is real
    
    # Normalization
    norm = np.trapz(abs_f_sq, x)
    if norm < 1e-15:
        return np.inf
    
    abs_f_sq_norm = abs_f_sq / norm
    
    # <x>
    x_mean = np.trapz(x * abs_f_sq_norm, x)
    x_sq_mean = np.trapz(x**2 * abs_f_sq_norm, x)
    sigma_x_sq = x_sq_mean - x_mean**2
    
    # Fourier transform
    # F(k) = integral f(x) e^{-ikx} dx
    # Use FFT
    # Need to handle the grid properly
    N_pts = len(x)
    x_range = x[-1] - x[0]
    k = 2*np.pi*np.fft.fftfreq(N_pts, d=dx)
    
    # FFT of f
    F_k = np.fft.fft(f) * dx
    # Shift so that k=0 is in the center
    F_k = np.fft.fftshift(F_k)
    k = np.fft.fftshift(k)
    
    # |F(k)|^2
    abs_F_sq = np.abs(F_k)**2
    norm_F = np.trapz(abs_F_sq, k)
    if norm_F < 1e-15:
        return np.inf
    abs_F_sq_norm = abs_F_sq / norm_F
    
    # <k> and <k^2>
    k_mean = np.trapz(k * abs_F_sq_norm, k)
    k_sq_mean = np.trapz(k**2 * abs_F_sq_norm, k)
    sigma_k_sq = k_sq_mean - k_mean**2
    
    if sigma_x_sq < 0:
        sigma_x_sq = 0
    if sigma_k_sq < 0:
        sigma_k_sq = 0
    
    return np.sqrt(sigma_x_sq) * np.sqrt(sigma_k_sq)

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s kill
    
    # Precompute Hermite polynomials and Gaussian on a grid for fast evaluation
    x = np.linspace(-8, 8, 4096)
    dx = x[1] - x[0]
    H = np.zeros((13, len(x)))
    H[0] = 1.0
    H[1] = 2.0 * x
    for n in range(2, 13):
        H[n] = 2.0 * x * H[n-1] - 2.0 * n * H[n-2]
    gauss = np.exp(-x**2 / 2.0)
    
    # Precompute H_n * gauss for the relevant degrees
    Hg = np.zeros((13, len(x)))
    for n in range(13):
        Hg[n] = H[n] * gauss
    
    # Precompute x, x^2, and FFT-related quantities
    N_pts = len(x)
    k = 2*np.pi*np.fft.fftfreq(N_pts, d=dx)
    k_shifted = np.fft.fftshift(k)
    
    def fast_bound(c):
        """Fast computation of uncertainty product using precomputed grid."""
        # f(x) = sum c_n H_n(x) e^{-x^2/2}
        f = c[0]*Hg[0] + c[1]*Hg[4] + c[2]*Hg[8] + c[3]*Hg[12]
        
        abs_f_sq = f**2
        norm = np.trapz(abs_f_sq, x)
        if norm < 1e-15:
            return np.inf
        abs_f_sq_norm = abs_f_sq / norm
        
        x_mean = np.trapz(x * abs_f_sq_norm, x)
        x_sq_mean = np.trapz(x**2 * abs_f_sq_norm, x)
        sigma_x_sq = max(0.0, x_sq_mean - x_mean**2)
        
        F_k = np.fft.fft(f) * dx
        F_k = np.fft.fftshift(F_k)
        abs_F_sq = np.abs(F_k)**2
        norm_F = np.trapz(abs_F_sq, k_shifted)
        if norm_F < 1e-15:
            return np.inf
        abs_F_sq_norm = abs_F_sq / norm_F
        
        k_mean = np.trapz(k_shifted * abs_F_sq_norm, k_shifted)
        k_sq_mean = np.trapz(k_shifted**2 * abs_F_sq_norm, k_shifted)
        sigma_k_sq = max(0.0, k_sq_mean - k_mean**2)
        
        return np.sqrt(sigma_x_sq) * np.sqrt(sigma_k_sq)
    
    # Constraint: f(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=3991680
    # c0 + 12*c4 + 1680*c8 + 3991680*c12 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 3991680
    
    def objective(params):
        c4, c8 = params
        c0 = 1.0
        c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 3991680.0
        c = np.array([c0, c4, c8, c12])
        return fast_bound(c)
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    # Use differential evolution for global search
    if time.time() - start_time < deadline - 30:
        try:
            bounds = [(-1.0, 1.0), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=100, 
                popsize=15, tol=1e-8, polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Local refinement from multiple starting points
    start_points = [
        np.array([0.0, 0.0]),
        np.array([0.01, 0.0]),
        np.array([-0.01, 0.0]),
        np.array([0.0, 0.01]),
        np.array([0.0, -0.01]),
        np.array([0.01, 0.01]),
        np.array([-0.01, -0.01]),
        np.array([0.01, -0.01]),
        np.array([-0.01, 0.01]),
        np.array([0.001, 0.001]),
        np.array([-0.001, -0.001]),
    ]
    
    # Add the DE result as a starting point
    if np.isfinite(best_score):
        start_points.append(best_params.copy())
    
    for x0 in start_points:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final local polish from the best point
    if time.time() - start_time < deadline - 5 and np.isfinite(best_score):
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array([1.0, best_params[0], best_params[1]])
