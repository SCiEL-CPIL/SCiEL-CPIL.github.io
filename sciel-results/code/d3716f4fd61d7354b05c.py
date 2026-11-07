import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    def crra(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def crra_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Fine grid for internal computation
    # 2000 points should be very accurate and fast
    N_fine = 2000
    fine_grid = np.geomspace(0.001, 40.0, N_fine)
    
    # V[t, z, m_idx]
    V = np.zeros((horizon, n_z, N_fine))
    C = np.zeros((horizon, n_z, N_fine))
    
    # Last period: consume all
    for z in range(n_z):
        V[horizon-1, z, :] = crra(fine_grid)
        C[horizon-1, z, :] = fine_grid
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Prepare splines for next period
        splines = []
        spline_derivs = []
        for z in range(n_z):
            s = CubicSpline(fine_grid, V[t+1, z, :])
            splines.append(s)
            spline_derivs.append(s.derivative())
            
        for z in range(n_z):
            for m_idx, m in enumerate(fine_grid):
                # Define the function to find the root of: J'(c) = 0
                # J'(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * V'_{t+1}(z', R(m-c) + y_{z'})
                
                def euler_residual(c):
                    up = crra_prime(c)
                    cont_deriv = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        # Ensure m_next is within spline bounds or handle extrapolation
                        # CubicSpline extrapolates by default.
                        vp = spline_derivs[zp](m_next)
                        cont_deriv += transition[z, zp] * vp
                    return up - beta * R * cont_deriv
                
                # Bisection bounds
                c_low = 1e-12
                c_high = m
                
                # Check boundaries
                f_low = euler_residual(c_low)
                f_high = euler_residual(c_high)
                
                if f_low <= 0:
                    c_opt = c_low
                elif f_high >= 0:
                    c_opt = c_high
                else:
                    # Bisection
                    # f is decreasing, so f_low > 0 and f_high < 0
                    # We want f(c) = 0
                    # If f_low > 0 and f_high < 0, root is in between
                    # Wait, f is decreasing.
                    # If f_low > 0, the root is to the right.
                    # If f_high < 0, the root is to the left.
                    # So if f_low > 0 and f_high < 0, the root is in (c_low, c_high).
                    
                    # Correct logic:
                    # f(c) is strictly decreasing.
                    # If f(c_low) <= 0, then f(c) < 0 for all c >= c_low. Max at c_low.
                    # If f(c_high) >= 0, then f(c) > 0 for all c <= c_high. Max at c_high.
                    # Otherwise, f(c_low) > 0 and f(c_high) < 0. Root exists.
                    
                    # My previous check:
                    # if f_low <= 0: c_opt = c_low (Correct)
                    # elif f_high >= 0: c_opt = c_high (Correct)
                    # else: f_low > 0 and f_high < 0. Bisect.
                    
                    for _ in range(50):
                        c_mid = (c_low + c_high) / 2.0
                        f_mid = euler_residual(c_mid)
                        if f_mid > 0:
                            c_low = c_mid
                        else:
                            c_high = c_mid
                        if c_high - c_low < 1e-10:
                            break
                    c_opt = (c_low + c_high) / 2.0
                
                C[t, z, m_idx] = c_opt
                
                # Compute Value
                val = crra(c_opt)
                for zp in range(n_z):
                    m_next = R * (m - c_opt) + income[zp]
                    v_next = splines[zp](m_next)
                    val += beta * transition[z, zp] * v_next
                V[t, z, m_idx] = val
                
        # Check time
        if time.time() > deadline:
            # If we run out of time, we should break and interpolate what we have
            # But we need all t for the output.
            # If we break early, we don't have V for earlier t.
            # However, the loop is from t=4 down to 0.
            # If we break at t=2, we have V[4], V[3], V[2]. We are missing V[1], V[0].
            # We can't just return.
            # Given the speed, this is unlikely.
            pass

    # Interpolate C from fine_grid to resource_grid
    C_out = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.interp(resource_grid, fine_grid, C[t, z, :])
            
    # Ensure c[5] is exactly the grid
    C_out[horizon-1, :, :] = resource_grid
    
    # Final validity check and clipping
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.maximum(C_out[t, z, :], 1e-12)
            C_out[t, z, :] = np.minimum(C_out[t, z, :], resource_grid + 1e-10)
            
    return C_out
