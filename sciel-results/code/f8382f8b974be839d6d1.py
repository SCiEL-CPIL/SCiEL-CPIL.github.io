import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin
    
    def hermite_p(n, x):
        """Compute nth probabilists' Hermite polynomial? Or physicists'?"""
        # Physicists' Hermite polynomials
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2*x
        elif n == 2:
            return 4*x**2 - 2
        elif n == 3:
            return 8*x**3 - 12*x
        elif n == 4:
            return 16*x**4 - 48*x**2 + 12
        elif n == 5:
            return 32*x**5 - 160*x**3 + 120*x
        elif n == 6:
            return 64*x**6 - 480*x**4 + 720*x**2 - 120
        elif n == 7:
            return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
        elif n == 8:
            return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
        elif n == 12:
            return 4096*x**12 - 98304*x**10 + 1245184*x**8 - 6635520*x**6 + 13305600*x**4 - 9031680*x**2 + 1039500
        else:
            raise ValueError("n not supported")
    
    # Wait, let me use the standard physicists' Hermite polynomials
    # H_0(x) = 1
    # H_1(x) = 2x
    # H_2(x) = 4x^2 - 2
    # H_3(x) = 8x^3 - 12x
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_5(x) = 32x^5 - 160x^3 + 120x
    # H_6(x) = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7(x) = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8(x) = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    # H_9(x) = 512x^9 - 10752x^7 + 60480x^5 - 100800x^3 + 30240x
    # H_10(x) = 1024x^10 - 28672x^8 + 241920x^6 - 604800x^4 + 403200x^2 - 30240
    # H_11(x) = 2048x^11 - 73728x^9 + 704640x^7 - 2217600x^5 + 2217600x^3 - 302400x
    # H_12(x) = 4096x^12 - 184320x^10 + 2654208x^8 - 16515072x^6 + 49545600x^4 - 44089920x^2 + 1039500
    
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
    def H12(x): return 4096*x**12 - 184320*x**10 + 2654208*x**8 - 16515072*x**6 + 49545600*x**4 - 44089920*x**2 + 1039500
    
    # Check values at 0
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 1039500
    
    # g(0) = c0*1 + c4*12 + c8*1680 + c12*1039500 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 1039500
    
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2)
    
    h0 = H0(x)
    h4 = H4(x)
    h8 = H8(x)
    h12 = H12(x)
    
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 + 12*c4 + 1680*c8) / 1039500.0
        
        g = (c0*h0 + c4*h4 + c8*h8 + c12*h12) * gauss
        
        # L1 norm
        l1 = np.sum(np.abs(g)) * dx
        # L2 norm squared
        l2_sq = np.sum(g**2) * dx
        
        if l2_sq < 1e-30:
            return 1e30
        
        # Scale-invariant quantity: (L1)^2 / L2^2
        # This is minimized when the function is as "concentrated" as possible
        # For a Gaussian, this ratio is fixed. Adding Hermite terms changes it.
        return (l1**2) / l2_sq
    
    # Start from seed
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = x0
    best_val = objective(x0)
    
    # Try multiple random starts
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 30:
            break
        x_rand = np.random.randn(3) * 0.5
        res = minimize(objective, x_rand, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    
    # Polish from best
    if time.time() < deadline - 10:
        res = minimize(objective, best_result, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    
    return best_result
