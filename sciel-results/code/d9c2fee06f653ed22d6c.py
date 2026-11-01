"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 180.0
    
    # Hermite polynomial values at 0
    # H_n(0) for even n
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -10395
    
    # Factorials
    f0 = 1.0
    f4 = 24.0
    f8 = 40320.0
    f12 = 479001600.0
    
    # h_k = H_k(0) / sqrt(k!)
    h0 = 1.0 / np.sqrt(f0)
    h4 = -3.0 / np.sqrt(f4)
    h8 = 105.0 / np.sqrt(f8)
    h12 = -10395.0 / np.sqrt(f12)
    
    # We want to minimize Q = (u0^2 * 1 + u4^2 * 9 + u8^2 * 17 + u12^2 * 25) / (u0^2 + u4^2 + u8^2 + u12^2)
    # subject to u0 * h0 + u4 * h4 + u8 * h8 + u12 * h12 = 0
    
    # u0 = -(h4*u4 + h8*u8 + h12*u12) / h0
    # Since h0 = 1, u0 = -(h4*u4 + h8*u8 + h12*u12)
    
    def objective(vars):
        if time.time() > deadline - 5.0:
            return np.inf
        u4, u8, u12 = vars
        u0 = -(h4 * u4 + h8 * u8 + h12 * u12)
        
        num = 1.0 * u0**2 + 9.0 * u4**2 + 17.0 * u8**2 + 25.0 * u12**2
        den = 1.0 * u0**2 + 1.0 * u4**2 + 1.0 * u8**2 + 1.0 * u12**2
        
        if den == 0:
            return np.inf
        return num / den

    # Initial guess
    # The minimum should be close to 1, so u0 should be large relative to others.
    # This implies h4*u4 + h8*u8 + h12*u12 is small.
    # Let's try a random start or a simple start.
    x0 = np.array([0.1, 0.1, 0.1])
    
    # Use Nelder-Mead or BFGS
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if time.time() > deadline - 2.0:
             # Fallback to seed if too close to deadline
             return np.array([-113 / 100, 1 / 25, 1 / 3240])
        
        if res.success:
            u4_opt, u8_opt, u12_opt = res.x
            u0_opt = -(h4 * u4_opt + h8 * u8_opt + h12 * u12_opt)
            
            # Calculate coefficients a_k = u_k / sqrt(k!)
            a0 = u0_opt / np.sqrt(f0)
            a4 = u4_opt / np.sqrt(f4)
            a8 = u8_opt / np.sqrt(f8)
            
            # The scale doesn't matter for the ratio, but let's normalize to make numbers nice or just return them.
            # The problem asks for coefficients. Any scalar multiple is valid.
            # Let's return the raw computed values.
            return np.array([a0, a4, a8])
        else:
            # If optimization fails, return seed
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
            
    except Exception:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

# EVOLVE-BLOCK-END
