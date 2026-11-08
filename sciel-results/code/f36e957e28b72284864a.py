import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import fft, ifft
import time

def _hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x)"""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2 * x
    for k in range(2, n + 1):
        H_n = 2 * x * H_nm1 - 2 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_nm1

def _compute_c12(c0, c4, c8):
    """Compute c12 from constraint g(0) = 0"""
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=103950
    return -(c0 + 12*c4 + 1680*c8) / 103950

def _compute_bound(c0, c4, c8):
    """Compute ||g||_1 * ||ĝ||_1 for the test function"""
    c12 = _compute_c12(c0, c4, c8)
    
    # Use a fine grid
    N = 200000
    x_max = 15
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # Compute H_n(x)
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 73728*x**10 + 451584*x**8 - 1221120*x**6 + 1451520*x**4 - 665280*x**2 + 103950
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = P * np.exp(-x**2/2)
    
    # ||g||_1
    norm_g_1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform using FFT
    G = np.fft.fft(g) * dx
    G_shifted = np.fft.fftshift(G)
    
    # ||ĝ||_1 = ∫ |ĝ(ξ)| dξ
    # The frequency spacing is dξ = 1 / (N * dx)
    dxi = 1.0 / (N * dx)
    norm_G_1 = np.sum(np.abs(G_shifted)) * dxi
    
    return norm_g_1 * norm_G_1

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds max
    
    # Fix c0 = 1.0 to handle scale invariance
    c0_fixed = 1.0
    
    # Initial guess for c4, c8
    c4_init = 0.0
    c8_init = 0.0
    
    def objective_2d(params):
        c4, c8 = params
        try:
            return _compute_bound(c0_fixed, c4, c8)
        except:
            return 1e10
    
    # Check time
    if time.time() > deadline - 10:
        return np.array([c0_fixed, c4_init, c8_init])
        
    best_c4, best_c8 = c4_init, c8_init
    best_val = objective_2d([c4_init, c8_init])
    
    # Use Differential Evolution for global search
    try:
        bounds = [(-5, 5), (-5, 5)]
        res = differential_evolution(objective_2d, bounds, maxiter=50, popsize=10, tol=1e-6, seed=42, polish=False)
        
        if time.time() < deadline - 10:
            if res.fun < best_val:
                best_val = res.fun
                best_c4, best_c8 = res.x
    except:
        pass
    
    # Local polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective_2d, [best_c4, best_c8], method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c4, best_c8 = res2.x
        except:
            pass
            
    return np.array([c0_fixed, best_c4, best_c8])
