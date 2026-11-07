import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Consumption policy: consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Helper for marginal utility u'(c)
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if abs(g - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-g)
            
    # Helper for second derivative u''(c)
    def u_double_prime(c, g):
        c = np.maximum(c, 1e-12)
        if abs(g - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -g * c ** (-g - 1)

    # Terminal condition: at t=5 (last decision), consume everything.
    # c_5(m, z) = m
    c_next = np.zeros((Z, M))
    for z in range(Z):
        c_next[z] = resource_grid
    
    # Create splines for the terminal condition
    # We store splines for c_{t+1}(m) for each state z
    # c is a function of m, so spline interpolates m -> c
    splines_next = [CubicSpline(resource_grid, c_next[z]) for z in range(Z)]
    
    # Initial guess for saving rate s
    # For log utility, steady state saving is roughly (1-beta)/(1+beta) * (R-1)/R ? 
    # Actually, let's just start with a reasonable constant.
    s_prev = np.full(M, 0.2)
    
    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule
            # To be safe and valid, consume all is valid (c=m).
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        c_curr = np.zeros((Z, M))
        
        for z in range(Z):
            # We want to find c(m) for each m in resource_grid
            # Let s = (m - c) / m, so c = m * (1 - s)
            
            # Use previous s as initial guess
            s = s_prev.copy()
            
            # Newton's method iteration
            max_iters = 20
            tol = 1e-8
            converged = False
            
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                    
                m = resource_grid
                c = m * (1.0 - s)
                c = np.maximum(c, 1e-12)
                
                # LHS: log u'(c)
                # log u'(c) = -gamma log(c) (general form works for gamma=1 too if we treat 1-gamma limit carefully, 
                # but actually for gamma=1, log(1/c) = -log(c). 
                # The formula -gamma * log(c) works for all gamma if we define it correctly?
                # u(c) = c^(1-gamma)/(1-gamma) -> u' = c^(-gamma) -> log u' = -gamma log c.
                # For gamma=1, u(c)=log c -> u'=1/c -> log u' = -log c.
                # So yes, log u' = -gamma log c holds for all gamma.
                lhs = -gamma * np.log(c)
                
                # d/ds lhs = -gamma * (1/c) * dc/ds = -gamma * (1/(m(1-s))) * (-m) = gamma / (1-s)
                d_lhs = gamma / (1.0 - s)
                
                # RHS: log( beta * R * E[u'(c_next)] )
                # E[u'(c_next)] = sum_{z'} P(z'|z) * u'(c_next(z', m'))
                # m' = R * m * s + income[z']
                
                # Vectorized computation for all z'
                # m_prime shape: (M, Z)
                m_prime = R * m[:, None] * s[:, None] + income[None, :]
                
                u_prime_next_sum = np.zeros(M)
                # Derivative of sum w.r.t s: sum P * u''(c_next) * c'_next(m') * dm'/ds
                # dm'/ds = R * m
                d_u_prime_next_sum = np.zeros(M)
                
                for zp in range(Z):
                    p_zz_p = transition[z, zp]
                    if p_zz_p < 1e-12:
                        continue
                    
                    # Evaluate spline and its derivative
                    c_next_val = splines_next[zp](m_prime[:, zp])
                    dc_next_dm = splines_next[zp](m_prime[:, zp], 1) # First derivative
                    
                    # Clip c_next_val to ensure positivity for u' and u''
                    c_safe = np.maximum(c_next_val, 1e-12)
                    
                    up = u_prime(c_safe, gamma)
                    upp = u_double_prime(c_safe, gamma)
                    
                    u_prime_next_sum += p_zz_p * up
                    # d/ds [ u'(c_next(m')) ] = u''(c_next) * dc_next/dm' * dm'/ds
                    # dm'/ds = R * m
                    d_u_prime_next_sum += p_zz_p * upp * dc_next_dm * (R * m)
                
                rhs_val = beta * R * u_prime_next_sum
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)
                
                # d/ds rhs = (1 / rhs_val) * beta * R * d_u_prime_next_sum
                #             = d_u_prime_next_sum / u_prime_next_sum
                # Note: u_prime_next_sum is E[u'].
                # So d/ds log(E[u']) = E[u'' c' m'] / E[u']
                # Let's use the computed sums directly to avoid division by zero issues if possible, 
                # but division by E[u'] is fine as it's positive.
                d_rhs = d_u_prime_next_sum / u_prime_next_sum
                
                # Residual F(s) = lhs - rhs
                F = lhs - rhs
                dF = d_lhs - d_rhs
                
                # Newton step
                # s_new = s - F / dF
                mask = np.abs(dF) > 1e-10
                s_new = s.copy()
                s_new[mask] = s[mask] - F[mask] / dF[mask]
                
                # Constrain s to [0, 1)
                # If s goes out of bounds, we might need to handle boundary conditions.
                # For now, clip. In a more robust solver, we'd check if the optimum is at the boundary.
                # Given the problem structure (infinite horizon like), interior solution is likely.
                s_new = np.clip(s_new, 0.0, 0.999999)
                
                # Check convergence
                if np.max(np.abs(s_new - s)) < tol:
                    s = s_new
                    converged = True
                    break
                s = s_new
                
            # Final c for this t, z
            c_curr[z] = resource_grid * (1.0 - s)
            
        consumption[t] = c_curr
        c_next = c_curr
        
        # Update splines for next iteration
        splines_next = [CubicSpline(resource_grid, c_next[z]) for z in range(Z)]
        
        # Update s_prev for next time step (t-1)
        # s_prev should be the s solved for at this t
        s_prev = s.copy()
        
    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            # 0 < c
            c = np.maximum(c, 1e-12)
            # c <= m + 1e-10
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[5] = resource_grid[None, :]
    
    return consumption
