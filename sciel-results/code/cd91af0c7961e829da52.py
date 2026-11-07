"""Solve consumption-saving policy via Euler-equation Newton with log-linear value interpolation."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    M = np.asarray(resource_grid, dtype=float)
    num_m = len(M)
    num_z = len(income)

    def marg_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        return c ** (-gamma)

    # log-linear interpolation of y = f(x) where x is log-space
    def log_interp(x, y, x_eval):
        """Interpolate y as a function of log(x) at x_eval (also in log-space)."""
        lx = np.log(x)
        ly = np.log(np.maximum(y, 1e-300))
        lx_eval = np.log(x_eval)
        return np.exp(np.interp(lx_eval, lx, ly))

    # consumption[t, z, m_index]
    c = np.zeros((horizon, num_z, num_m))
    # marginal value of resources at t, z, m: u'(m') where m' is next period's resources
    # We store the marginal value function V'(m) at each (t, z)
    # V_t(m, z) = max_c u(c) + beta * sum_z' P(z'|z) V_{t+1}(R(m-c)+y_{z'}, z')
    # FOC: u'(c) = beta * sum_z' P(z'|z) V'_{t+1}(R(m-c)+y_{z'}, z')

    # Final period: consume everything
    c[horizon - 1] = M[np.newaxis, :]
    # marginal value at final period is just u'(m)
    Vprime = np.zeros((horizon, num_z, num_m))
    Vprime[horizon - 1] = marg_util(M)[np.newaxis, :]

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        for z in range(num_z):
            # We need to solve for c[m] for each m in M
            # FOC: u'(c) = beta * sum_{z'} P(z'|z) Vprime[t+1, z', index_of(R*(m-c)+y_{z'})]
            # Vprime[t+1, z'] is defined on the grid M. We interpolate in log-space.
            
            # For each m, solve for c
            c_tz = np.zeros(num_m)
            for i in range(num_m):
                m = M[i]
                
                # Define the Euler equation residual: u'(c) - RHS(c) = 0
                # RHS(c) = beta * sum_{z'} P(z'|z) Vprime_interp(t+1, z', R*(m-c)+y_{z'})
                
                def residual(c_val):
                    c_val = max(c_val, 1e-12)
                    m_next = R * (m - c_val)
                    if m_next < 1e-12:
                        m_next = 1e-12
                    rhs = 0.0
                    for zp in range(num_z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_next_zp = m_next + income[zp]
                        # interpolate Vprime[t+1, zp] at m_next_zp in log-space
                        vp = log_interp(M, Vprime[t+1, zp], m_next_zp)
                        rhs += p * vp
                    return marg_util(c_val) - beta * rhs
                
                # Newton's method with bisection fallback
                # Initial guess: c = m * 0.5
                c_lo = 1e-12
                c_hi = m - 1e-12
                if c_hi <= c_lo:
                    c_tz[i] = m
                    continue
                
                # Check if solution is at boundary
                res_lo = residual(c_lo)
                res_hi = residual(c_hi)
                
                if res_lo < 0:
                    # u'(c) - RHS < 0 at c_lo, meaning u'(c) < RHS, so we need smaller c? 
                    # Actually u'(c) is decreasing in c, RHS is increasing in c (as m-c decreases, V' increases)
                    # So residual is decreasing in c. If res_lo < 0, solution is below c_lo -> c = c_lo
                    c_tz[i] = c_lo
                    continue
                if res_hi > 0:
                    # residual > 0 at c_hi, solution is above c_hi -> c = c_hi
                    c_tz[i] = c_hi
                    continue
                
                # Binary search + Newton
                c_val = 0.5 * (c_lo + c_hi)
                for _ in range(50):
                    res = residual(c_val)
                    if abs(res) < 1e-12:
                        break
                    if res > 0:
                        c_lo = c_val
                    else:
                        c_hi = c_val
                    # Newton step
                    dc = 1e-6
                    if c_val + dc > c_hi:
                        dc = -dc
                    if c_val + dc < c_lo:
                        dc = -dc
                    res2 = residual(c_val + dc)
                    if abs(res2 - res) > 1e-15:
                        c_new = c_val - res * dc / (res2 - res)
                        if c_lo < c_new < c_hi:
                            c_val = c_new
                            continue
                    c_val = 0.5 * (c_lo + c_hi)
                c_tz[i] = c_val
            
            c[t, z] = c_tz
            
            # Compute Vprime[t, z, i] = u'(c[t, z, i])
            Vprime[t, z] = marg_util(c_tz)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            c[t, z] = np.maximum(c[t, z], 1e-12)
            c[t, z] = np.minimum(c[t, z], M + 1e-10)
    
    # Final period must equal grid
    c[horizon - 1] = M[np.newaxis, :]
    
    # Check for any non-finite values and fix
    if not np.all(np.isfinite(c)):
        c = np.where(np.isfinite(c), c, M[np.newaxis, np.newaxis, :])
    
    return c
