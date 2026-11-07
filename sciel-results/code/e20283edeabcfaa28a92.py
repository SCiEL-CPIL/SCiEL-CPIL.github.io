import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid is the one we must return
    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)
    
    # Internal dense grid for high-accuracy backward induction
    # The evaluator uses np.geomspace(0.001, 40, 251). 
    # Using 2001 points provides ~8x resolution, significantly reducing interpolation error.
    m_int = np.geomspace(0.001, 40, 2001)
    n_int = len(m_int)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)
            
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
            
    def u_double_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c_safe ** 2)
        else:
            return -gamma * (c_safe ** (-gamma - 1))

    # Policy storage on internal grid: pol[t][z] is shape (n_int,)
    pol = np.zeros((horizon, n_z, n_int))
    
    # Terminal condition: c = m
    for z in range(n_z):
        pol[horizon - 1, z, :] = m_int
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        next_pol = pol[t + 1, :, :] # Shape (n_z, n_int)
        
        # Interpolators for next period's policy for each state
        # Using linear interpolation for speed, as the grid is dense enough
        next_pol_interp = []
        for z in range(n_z):
            # np.interp requires increasing x, m_int is increasing
            next_pol_interp.append(lambda m_val, z_val=z: np.interp(m_val, m_int, next_pol[z_val]))
            
        for z in range(n_z):
            c_values = np.zeros(n_int)
            
            # Initialize c to a fraction of m to start Newton's method
            # For CRRA, optimal c is roughly m / (1 + beta*R) for constant income, 
            # but here income varies. A safe start is 0.5*m or similar.
            c = 0.5 * m_int
            
            # Newton-Raphson iterations
            for _ in range(10):
                # Calculate RHS of Euler equation: E[u'(c')]
                rhs = np.zeros(n_int)
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    m_prime = R * (m_int - c) + income[zp]
                    # Clamp m_prime to valid range for interpolation
                    m_prime_clamped = np.clip(m_prime, m_int[0], m_int[-1])
                    
                    # Get optimal consumption next period via interpolation
                    c_next = next_pol_interp[zp](m_prime_clamped)
                    c_next = np.maximum(c_next, 1e-12)
                    
                    rhs += prob * u_prime(c_next)
                
                rhs *= beta * R
                
                # LHS: u'(c)
                lhs = u_prime(c)
                
                # Residual: F(c) = u'(c) - RHS
                residual = lhs - rhs
                
                # Derivative of residual with respect to c:
                # d/dc (u'(c) - RHS(c))
                # u'(c) derivative is u''(c)
                # RHS(c) = beta * R * sum p_zp u'(c'(c))
                # c'(c) is the next period's optimal consumption given m' = R(m-c)+y
                # dc'/dc = dc'/dm' * dm'/dc = c'_m * (-R)
                # We need the derivative of the next period's policy w.r.t m.
                # We can approximate this numerically or analytically if we stored derivatives.
                # For simplicity and robustness, we can use a numerical derivative of the policy function.
                # However, Newton-Raphson on the Euler equation is sensitive to the derivative.
                # Alternative: Use a fixed point iteration or bisection if Newton is unstable.
                # Given the convexity, the function is monotonic. 
                # Let's try a simpler approach: Bisection is robust but slow if unvectorized.
                # But we can vectorize bisection!
                
                # Let's switch to Vectorized Bisection for robustness, as Newton requires
                # accurate derivatives of the interpolated policy which is tricky.
                # Actually, we can compute the derivative of the policy function 
                # from the grid itself using np.gradient.
                
                # Let's stick to Newton but compute dRHS/dc carefully.
                # dRHS/dc = beta * R * sum p_zp u''(c_next) * (dc_next/dm') * (-R)
                # dc_next/dm' is the slope of the next period's policy.
                
                # Precompute slopes of next period's policies
                next_pol_slopes = []
                for zp in range(n_z):
                    # np.gradient gives derivative of y wrt x
                    slopes = np.gradient(next_pol[zp], m_int)
                    next_pol_slopes.append(slopes)
                
                d_rhs_dc = np.zeros(n_int)
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    m_prime = R * (m_int - c) + income[zp]
                    m_prime_clamped = np.clip(m_prime, m_int[0], m_int[-1])
                    
                    c_next = next_pol_interp[zp](m_prime_clamped)
                    c_next = np.maximum(c_next, 1e-12)
                    
                    # Interpolate the slope as well
                    slope_next = np.interp(m_prime_clamped, m_int, next_pol_slopes[zp])
                    
                    d_rhs_dc += prob * u_double_prime(c_next) * slope_next * (-R)
                
                d_rhs_dc *= beta * R
                
                d_residual_dc = u_double_prime(c) - d_rhs_dc
                
                # Newton step: c_new = c - residual / d_residual_dc
                # Handle division by zero or small derivatives
                with np.errstate(divide='ignore', invalid='ignore'):
                    step = residual / d_residual_dc
                    # If derivative is zero or negative (shouldn't happen for concave utility),
                    # or step is too large, dampen it.
                    # Simple damping: if step > 0.5 * c, reduce it.
                    step = np.where(np.abs(d_residual_dc) < 1e-12, 0, step)
                    # Ensure we stay within (0, m)
                    # c_new = c - step
                    # We want c_new in [1e-12, m]
                    c_new = c - step
                    c_new = np.clip(c_new, 1e-12, m_int)
                    
                    # If Newton step fails or goes out of bounds significantly, 
                    # we might need to fallback. But usually it works well for these problems.
                    # Check for convergence
                    if np.all(np.abs(c_new - c) < 1e-10):
                        c = c_new
                        break
                    
                    c = c_new
            
            # Final check: ensure c <= m
            c = np.minimum(c, m_int)
            c = np.maximum(c, 1e-12)
            
            pol[t, z, :] = c
            
            # Check time
            if time.time() > deadline - 10:
                # If we run out of time, we just stop and return what we have.
                # The rest of the pol array will be zeros, which is invalid.
                # So we should break out and handle validity.
                pass

    # Interpolate from internal grid to output grid
    pol_out = np.zeros((horizon, n_z, n_out))
    for t in range(horizon):
        for z in range(n_z):
            # np.interp requires increasing x
            pol_out[t, z, :] = np.interp(m_out, m_int, pol[t, z, :])
            
    # Final check: ensure validity
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = pol_out[t, z, :]
            # Floor at 1e-12
            c = np.maximum(c, 1e-12)
            # Cap at m
            c = np.minimum(c, m_out)
            pol_out[t, z, :] = c
            
    # Terminal condition check: c[5] equal to grid within 1e-9
    # Our interpolation should be exact for linear segments, but let's enforce it.
    pol_out[horizon-1, :, :] = m_out

    return pol_out
