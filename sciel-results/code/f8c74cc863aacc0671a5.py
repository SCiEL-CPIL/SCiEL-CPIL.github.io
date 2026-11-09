import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Construct A using n calls
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # 2. Verify time-invariance and linearity
    f_t1_y0 = f(t1, y0)
    pred_t1_y0 = A @ y0
    
    err = np.linalg.norm(f_t1_y0 - pred_t1_y0)
    scale = max(1.0, np.linalg.norm(f_t1_y0))
    
    if err / scale < 1e-10:
        dt = t1 - t0
        B = A * dt
        
        # Handle trivial case
        if np.allclose(B, 0):
            return y0
            
        # Scaling and Squaring with (5,5) Padé approximation
        # Coefficients for (5,5) Padé approximant of exp(x)
        # N(x) = 1 + 5/2 x + 3/2 x^2 + 1/2 x^3 + 1/10 x^4 + 1/120 x^5
        # D(x) = 1 - 5/2 x + 3/2 x^2 - 1/2 x^3 + 1/10 x^4 - 1/120 x^5
        c = np.array([1.0, 2.5, 1.5, 0.5, 0.1, 1.0/120.0])
        
        # Determine scaling factor s
        # Threshold for (5,5) is typically around 0.95 to 1.0
        norm_B = np.linalg.norm(B, 1)
        s = 0
        while norm_B / (2 ** s) > 0.95:
            s += 1
            
        Q = B / (2 ** s)
        
        # Precompute N and D matrices
        N = np.eye(n)
        D = np.eye(n)
        Qk = np.eye(n)
        for k in range(1, 6):
            Qk = Qk @ Q
            N += c[k] * Qk
            if k % 2 == 1:
                D -= c[k] * Qk
            else:
                D += c[k] * Qk
                
        # Apply the approximation s times
        v = y0.copy()
        for _ in range(s):
            v = solve(D, N @ v)
            
        return v
    else:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        return sol.y[:, -1]
