import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    # Precompute Hermite polynomials (probabilists' He_n) on a fine grid
    x_grid = np.linspace(-15, 15, 5000)
    exp_factor = np.exp(-x_grid**2 / 2)
    
    He0 = np.ones_like(x_grid)
    He4 = x_grid**4 - 6*x_grid**2 + 3
    He8 = x_grid**8 - 28*x_grid**6 + 210*x_grid**4 - 420*x_grid**2 + 105
    He12 = (x_grid**12 - 66*x_grid**10 + 1155*x_grid**8 - 6930*x_grid**6 + 
            13860*x_grid**4 - 7560*x_grid**2 + 10395)
    
    # Precompute weighted Hermite functions
    h0 = exp_factor * He0
    h4 = exp_factor * He4
    h8 = exp_factor * He8
    h12 = exp_factor * He12
    
    def compute_c12(c0, c4, c8):
        # He_n(0): He0(0)=1, He4(0)=3, He8(0)=105, He12(0)=10395
        return -(c0 + 3*c4 + 105*c8) / 10395.0
    
    def get_g_and_fft(c0, c4, c8):
        c12 = compute_c12(c0, c4, c8)
        g = c0*h0 + c4*h4 + c8*h8 + c12*h12
        
        # Fourier transform of He_n(x)*exp(-x^2/2)
        # For probabilists' Hermite: F[He_n(x) e^{-x^2/2}](xi) = (-i)^n He_n(xi) e^{-xi^2/2} * sqrt(2*pi)
        # But we need to be careful with conventions. Let's use the fact that
        # the Fourier transform maps the Hermite function space to itself with phase factors.
        # For our purposes, |F[g]| has the same magnitude structure as g but with
        # coefficients multiplied by (-i)^n.
        # |(-i)^0| = 1, |(-i)^4| = 1, |(-i)^8| = 1, |(-i)^12| = 1
        # So |F[g]| has the same form as |g| but we need to compute it properly.
        
        # Actually, let's compute the FFT numerically for accuracy
        dx = x_grid[1] - x_grid[0]
        # Zero-pad for better FFT resolution
        n_fft = 16384
        g_padded = np.zeros(n_fft)
        # Place g in the center
        mid = len(x_grid) // 2
        g_padded[mid:mid+len(x_grid)] = g
        # Shift so that x=0 is at the center
        g_padded = np.roll(g_padded, -mid)
        
        G = np.fft.fft(g_padded) * dx
        G = np.fft.fftshift(G)
        
        return np.max(np.abs(g)), np.max(np.abs(G))
    
    def objective(c):
        c0, c4, c8 = c
        max_g, max_G = get_g_and_fft(c0, c4, c8)
        return max_g * max_G
    
    # Multi-start optimization
    best_val = np.inf
    best_c = None
    
    # Start from seed
    c_seed = np.array([-113/100, 1/25, 1/3240])
    
    # Generate multiple starting points
    np.random.seed(42)
    starts = [c_seed]
    for _ in range(20):
        starts.append(c_seed + 0.5 * np.random.randn(3))
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except:
            pass
    
    # Also try differential evolution for global search
    try:
        bounds = [(-5, 5), (-5, 5), (-1, 1)]
        de_res = differential_evolution(objective, bounds, maxiter=100, 
                                        popsize=15, tol=1e-8, seed=42)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_c = de_res.x
    except:
        pass
    
    if best_c is not None:
        return best_c
    else:
        return c_seed
