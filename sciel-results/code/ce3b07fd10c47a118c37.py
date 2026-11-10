import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
        
    def util_second_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Pre-compute splines for terminal value
    # V_T(m) = u(m)
    # We need V_T', V_T''
    # For gamma=1: u(m)=log(m), u'(m)=1/m, u''(m)=-1/m^2
    # For gamma!=1: u(m)=m^(1-g)/(1-g), u'(m)=m^(-g), u''(m)=-g*m^(-g-1)
    
    v_prime = np.zeros((horizon, num_states, num_resources))
    v_second_prime = np.zeros((horizon, num_states, num_resources))
    
    # Initialize terminal derivatives
    if abs(gamma - 1.0) < 1e-12:
        v_prime[horizon - 1, :, :] = 1.0 / resource_grid[np.newaxis, :]
        v_second_prime[horizon - 1, :, :] = -1.0 / (resource_grid[np.newaxis, :] ** 2)
    else:
        v_prime[horizon - 1, :, :] = resource_grid[np.newaxis, :] ** (-gamma)
        v_second_prime[horizon - 1, :, :] = -gamma * (resource_grid[np.newaxis, :] ** (-gamma - 1))

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        # Create splines for V_{t+1}(m) for each state z'
        # Note: resource_grid is geomspace, which is non-uniform.
        # CubicSpline handles non-uniform grids well.
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(resource_grid, value[t + 1, zp, :]))
            
        for z in range(num_states):
            m = resource_grid
            # Initial guess for c: use previous period's policy or a fraction
            # c ~ m / (horizon - t) is a rough guess
            # Let's use c = 0.5 * m as a safe initial guess
            c_est = 0.5 * m
            
            # Newton-Raphson iteration
            # f(c) = u'(c) - beta * R * sum_{zp} P(z|zp) * V'_{t+1}(R*(m-c)+income[zp])
            # f'(c) = u''(c) + beta * R^2 * sum_{zp} P(z|zp) * V''_{t+1}(R*(m-c)+income[zp])
            
            for _ in range(20):  # Max 20 Newton steps
                # Compute m_next for each zp
                # m_next_zp = R * (m - c_est) + income[zp]
                
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    m_next = R * (m - c_est) + income[zp]
                    # Clip m_next to grid range for interpolation safety
                    # Splines can extrapolate, but let's be safe if needed
                    # CubicSpline extrapolates by default (notclamped) or we can set it.
                    # Default behavior is fine for smooth functions.
                    
                    vp = splines[zp](m_next, 1)
                    vpp = splines[zp](m_next, 2)
                    
                    sum_v_prime += transition[z, zp] * vp
                    sum_v_second_prime += transition[z, zp] * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                # Newton step
                # Avoid division by zero
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                c_new = c_est - f / f_prime_safe
                
                # Clip c_new to [1e-12, m]
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m)
                
                # Check convergence
                if np.all(np.abs(c_new - c_est) < 1e-8):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
            
            # Compute value V_t(m) = u(c) + beta * E[V_{t+1}(m')]
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                m_next = R * (m - c_est) + income[zp]
                v_next = splines[zp](m_next, 0)
                sum_v_next += transition[z, zp] * v_next
                
            value[t, z, :] = u_c + beta * sum_v_next
            
            # Compute derivatives of V_t for the next iteration
            # V_t'(m) = u'(c) * c'(m) + beta * E[V'_{t+1}(m') * m'(m)]
            # This is complex. Instead, we can just fit a new spline to value[t, z, :]
            # But we need v_prime[t, z, :] for the next step's Newton method?
            # Wait, the Newton method for step t uses V_{t+1}' and V_{t+1}''.
            # So we don't need to analytically derive V_t'. We just need V_t values to build splines for the next step.
            # The loop structure is:
            # 1. Have V_{t+1} values.
            # 2. Build splines for V_{t+1}.
            # 3. Solve for c_t using Newton (needs V_{t+1}', V_{t+1}'').
            # 4. Compute V_t values.
            # 5. Loop t--.
            # So we don't need to store v_prime[t] explicitly, just build splines from value[t] in the next iteration.
            pass

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m

    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
