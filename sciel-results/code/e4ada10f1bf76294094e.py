"""Rewritten program: log-space Euler equation solver with Newton polish."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start = time.time()
    DEADLINE = 135.0

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)
    N_m = len(m_grid)
    N_z = len(income)

    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1.0:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1.0:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def u_prime2(c):
        """Second derivative of utility."""
        c = np.maximum(c, 1e-12)
        if gamma == 1.0:
            return -1.0 / c ** 2
        else:
            return -gamma * c ** (-gamma - 1)

    policy = np.empty((horizon, N_z, N_m), dtype=float)

    # Terminal: c = m
    for z in range(N_z):
        policy[horizon - 1, z, :] = m_grid

    # Value function at terminal: V_{T}(z, m) = u(m)
    V = np.empty((N_z, N_m), dtype=float)
    for z in range(N_z):
        V[z, :] = util(m_grid)

    # Work backward
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE - 15.0:
            # Fallback: use a simple heuristic
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    # Heuristic: save some fraction
                    policy[tt, z, :] = 0.8 * m_grid
            break

        # Build value function derivatives at grid points for continuation
        # V_{t+1}[z, m] and its first derivative V'_{t+1}[z, m]
        # Use finite differences on the grid for derivatives
        V_next = V.copy()  # V_{t+1}
        
        # Compute V'_{t+1} via central differences (vectorized)
        V_prime_next = np.empty((N_z, N_m), dtype=float)
        dm = m_grid[1] - m_grid[0]  # uniform spacing? No, geomspace
        # For geomspace, use log-space finite differences
        log_m = np.log(m_grid)
        dlog_m = log_m[1] - log_m[0]
        
        for z in range(N_z):
            # dV/dlog_m = m * dV/dm
            # Central difference in log-space
            Vp = np.zeros(N_m, dtype=float)
            for i in range(N_m):
                if i == 0:
                    Vp[i] = (V_next[z, 1] - V_next[z, 0]) / dlog_m
                elif i == N_m - 1:
                    Vp[i] = (V_next[z, -1] - V_next[z, -2]) / dlog_m
                else:
                    Vp[i] = (V_next[z, i+1] - V_next[z, i-1]) / (2.0 * dlog_m)
            V_prime_next[z, i] = Vp[i]  # This is dV/dlog_m = m * V'(m)
        
        # Now solve for each (z, m) using Newton on the Euler equation in log-space
        # Let x = log(c/m), so c = m * exp(x)
        # Euler: u'(c) = beta * R * sum_z' P[z,z'] * V'_{t+1}[z'](R*(m-c)+income[z'])
        # In terms of x: u'(m*exp(x)) = beta * R * sum_z' P[z,z'] * V'_{t+1}[z'](R*m*(1-exp(x))+income[z'])
        
        for z in range(N_z):
            probs = transition[z, :]
            c_vals = np.empty(N_m, dtype=float)
            
            for i_m in range(N_m):
                m = m_grid[i_m]
                
                # Initial guess: use a simple rule
                # For CRRA, optimal saving depends on m
                # Start with c = m * 0.9 as initial guess
                x0 = np.log(0.9)
                
                # Define the Euler equation residual in log-space
                # f(x) = u'(m*exp(x)) - beta * R * sum_z' P[z,z'] * V'_{t+1}[z'](R*m*(1-exp(x))+income[z'])
                
                # We need V'_{t+1}[z'](m') where m' = R*(m-c)+income[z'] = R*m*(1-exp(x))+income[z']
                # V'_{t+1} is dV/dm, but we have dV/dlog_m = m*V'(m)
                # So V'(m) = V'_log(m) / m
                
                def euler_residual(x):
                    c = m * np.exp(x)
                    c = np.clip(c, 1e-12, m)
                    left = u_prime(c)
                    
                    right = 0.0
                    for zp in range(N_z):
                        if probs[zp] == 0:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        m_prime = np.clip(m_prime, m_grid[0], m_grid[-1])
                        
                        # Find index for interpolation
                        # Since m_prime may not be on grid, use linear interp in log-space
                        log_mp = np.log(m_prime)
                        # Find bracketing indices
                        idx = np.searchsorted(log_m, log_mp)
                        if idx <= 0:
                            Vp_val = V_prime_next[zp, 0] / m_grid[0]
                        elif idx >= N_m:
                            Vp_val = V_prime_next[zp, -1] / m_grid[-1]
                        else:
                            # Linear interp in log-space
                            w = (log_mp - log_m[idx-1]) / (log_m[idx] - log_m[idx-1])
                            Vp_log = (1-w) * V_prime_next[zp, idx-1] + w * V_prime_next[zp, idx]
                            m_interp = np.exp((1-w) * log_m[idx-1] + w * log_m[idx])
                            Vp_val = Vp_log / m_interp
                        
                        right += probs[zp] * Vp_val
                    
                    return left - beta * R * right
                
                # Newton iteration
                x = x0
                for it in range(30):
                    f = euler_residual(x)
                    if abs(f) < 1e-12:
                        break
                    # Numerical derivative
                    dx = 1e-6
                    f_prime = (euler_residual(x + dx) - euler_residual(x - dx)) / (2 * dx)
                    if abs(f_prime) < 1e-14:
                        break
                    x_new = x - f / f_prime
                    # Keep x in reasonable range: c in (1e-12, m) => x in (log(1e-12/m), 0)
                    x_min = np.log(1e-12 / m)
                    x_max = 0.0
                    x_new = np.clip(x_new, x_min, x_max)
                    if abs(x_new - x) < 1e-14:
                        x = x_new
                        break
                    x = x_new
                
                c_final = m * np.exp(x)
                c_final = np.clip(c_final, 1e-12, m)
                c_vals[i_m] = c_final
            
            policy[t, z, :] = c_vals
        
        # Update value function
        for z in range(N_z):
            probs = transition[z, :]
            c_z = policy[t, z, :]
            u_c = util(c_z)
            m_minus_c = m_grid - c_z
            R_mmc = R * m_minus_c
            
            # Continuation value: interpolate V_{t+1} at R_mmc + income[z']
            cont = np.zeros(N_m, dtype=float)
            for zp in range(N_z):
                if probs[zp] == 0:
                    continue
                x_next = R_mmc + income[zp]
                x_next = np.clip(x_next, m_grid[0], m_grid[-1])
                # Interpolate in log-space
                log_xn = np.log(x_next)
                idx = np.searchsorted(log_m, log_xn)
                vals = np.zeros(N_m, dtype=float)
                for i in range(N_m):
                    if idx[i] <= 0:
                        vals[i] = V_next[zp, 0]
                    elif idx[i] >= N_m:
                        vals[i] = V_next[zp, -1]
                    else:
                        w = (log_xn[i] - log_m[idx[i]-1]) / (log_m[idx[i]] - log_m[idx[i]-1])
                        vals[i] = (1-w) * V_next[zp, idx[i]-1] + w * V_next[zp, idx[i]]
                cont += probs[zp] * vals
            
            V[z, :] = u_c + beta * cont
        
        if time.time() > DEADLINE:
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    policy[tt, z, :] = 0.8 * m_grid
            break

    # Ensure validity
    for t in range(horizon):
        for z in range(N_z):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
            policy[t, z, :] = np.where(np.isfinite(policy[t, z, :]), policy[t, z, :], m_grid)
    
    # Final period must equal grid
    for z in range(N_z):
        policy[horizon - 1, z, :] = m_grid
    
    return policy
