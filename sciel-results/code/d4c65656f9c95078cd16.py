import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 180.0

    # Hermite polynomials H_0, H_4, H_8, H_12
    # We need integrals of the form I_{n,m} = \int_{-\infty}^{\infty} e^{-x^2} H_n(x) H_m(x) dx
    # and J_{n,m} = \int_{-\infty}^{\infty} x^2 e^{-x^2} H_n(x) H_m(x) dx
    
    # Using the property that H_n are orthogonal with weight e^{-x^2/2}, 
    # but here the weight is e^{-x^2}.
    # Let's use the explicit values or compute them.
    # Actually, we can compute the matrix elements numerically or analytically.
    # Given the small number of terms (4x4 matrix), numerical integration is safe and fast.
    
    def hermite(n, x):
        # Physicists' Hermite polynomials
        if n == 0: return np.ones_like(x)
        if n == 1: return 2*x
        # Use recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
        h0 = np.ones_like(x)
        h1 = 2*x
        for k in range(1, n):
            h2 = 2*x*h1 - 2*k*h0
            h0 = h1
            h1 = h2
        return h1

    # Precompute H_0, H_4, H_8, H_12 on a grid? 
    # No, we can compute the integrals analytically or using scipy.integrate.quad.
    # Quad is robust.
    
    def compute_integrals():
        # We need M_0[n,m] = \int e^{-x^2} H_n(x) H_m(x) dx
        # and M_1[n,m] = \int x^2 e^{-x^2} H_n(x) H_m(x) dx
        # for n,m in [0, 4, 8, 12]
        
        indices = [0, 4, 8, 12]
        M0 = np.zeros((4,4))
        M1 = np.zeros((4,4))
        
        # To save time, we can compute these once.
        # But doing it inside the objective is too slow if called many times.
        # Let's compute them once outside the objective function.
        
        from scipy.integrate import quad
        
        for i, ni in enumerate(indices):
            hi = hermite(ni, 0) # dummy, we need function
            # Define function for quad
            def integrand0(x):
                return np.exp(-x*x) * hermite(ni, x) * hermite(indices[0], x) # placeholder
            
            # Better: loop
            for j, nj in enumerate(indices):
                def f0(x):
                    return np.exp(-x*x) * hermite(ni, x) * hermite(nj, x)
                val0, _ = quad(f0, -np.inf, np.inf, limit=100)
                M0[i,j] = val0
                
                def f1(x):
                    return x*x * np.exp(-x*x) * hermite(ni, x) * hermite(nj, x)
                val1, _ = quad(f1, -np.inf, np.inf, limit=100)
                M1[i,j] = val1
        return M0, M1

    # Compute matrices once
    M0, M1 = compute_integrals()
    
    # Constraint: P(0) = 0
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 560
    # H_12(0) = 425280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 560.0
    H12_0 = 425280.0
    
    def get_a12(a0, a4, a8):
        return -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0

    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        a = np.array([a0, a4, a8, a12])
        
        # Norm squared: a^T M0 a
        norm_sq = np.dot(a, np.dot(M0, a))
        if norm_sq < 1e-10:
            return 1e10 # Avoid division by zero
            
        # Moment squared: a^T M1 a
        moment_sq = np.dot(a, np.dot(M1, a))
        
        # Ratio to minimize: (moment_sq / norm_sq)^2
        # Actually the uncertainty product is proportional to moment_sq^2 / norm_sq^2
        # We just need to minimize moment_sq / norm_sq
        return moment_sq / norm_sq

    # Initial guess from seed
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    best_val = objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Multi-start optimization
    # Try the seed and some random points
    rng = np.random.default_rng(42)
    
    # Add some structured starts
    starts = [seed_coeffs]
    # Random starts in a reasonable range
    for _ in range(20):
        # Coefficients are likely small
        starts.append(rng.normal(0, 1, 3))
        
    for s in starts:
        try:
            res = minimize(objective, s, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            continue
            
        if time.time() > deadline - 10:
            break
            
    return best_coeffs
