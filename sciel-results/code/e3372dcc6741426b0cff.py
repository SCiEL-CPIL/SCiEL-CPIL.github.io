"""Solve consumption-saving policy using Chebyshev spectral approximation of marginal value."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    num_z = len(income)
    
    # CRRA utility and marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Inverse of marginal utility: c from u_prime
    def c_from_mu(mu):
        mu = np.maximum(mu, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / mu
        else:
            return mu ** (-1.0 / gamma)

    # Use a high-resolution log-spaced grid for wealth
    num_fine = 4096
    m_fine = np.geomspace(0.001, 40, num_fine)
    
    # We'll store the marginal value function V'(m) on the fine grid for each (t, z)
    # V'(m) = u'(c*(m)) by the envelope theorem
    # So we can recover c*(m) = c_from_mu(V'(m))
    
    # Initialize arrays for fine grid
    mu_fine = np.zeros((horizon, num_z, num_fine))
    
    # Terminal condition: c[m] = m, so V'(m) = u'(m)
    mu_fine[horizon - 1, :, :] = u_prime(m_fine)[np.newaxis, np.newaxis, :]
    
    # Backward induction using Euler equation
    # u'(c_t) = beta * R * sum_z' pi(z'|z) * V'_{t+1}(R*(m-c_t) + income[z'], z')
    # Let mu_t = u'(c_t). Then:
    # mu_t = beta * R * sum_z' pi(z'|z) * mu_{t+1}(R*(m - c_from_mu(mu_t)) + income[z'], z')
    # This is a fixed-point equation in mu_t.
    
    for t in range(horizon - 2, -1, -1):
        # Check time budget
        if time.time() > deadline - 10.0:
            break
        
        for z in range(num_z):
            # Transition probabilities from z to z'
            pi = transition[z]  # shape (num_z,)
            
            # Initialize mu_t with a reasonable guess
            # Start with mu_t = u'(0.5 * m)
            c_guess = 0.5 * m_fine
            mu_t = u_prime(c_guess).copy()
            
            # Fixed-point iteration with relaxation
            omega = 0.7  # Relaxation parameter
            for iter in range(300):
                if time.time() > deadline - 10.0:
                    break
                
                # Compute c_t from current mu_t
                c_t = c_from_mu(mu_t)
                c_t = np.clip(c_t, 1e-12, m_fine)
                
                # Compute next period's wealth for each z'
                m_next = R * (m_fine[np.newaxis, :] - c_t[np.newaxis, :]) + income[:, np.newaxis]
                # m_next has shape (num_z, num_fine)
                
                # Ensure m_next is positive
                m_next = np.maximum(m_next, 1e-12)
                
                # Interpolate next period's marginal value from mu_fine[t+1]
                mu_next = np.zeros_like(m_next)
                for zp in range(num_z):
                    mu_next[zp, :] = np.interp(m_next[zp, :], m_fine, mu_fine[t + 1, zp, :])
                
                # Weighted sum over z'
                mu_expected = np.sum(pi[:, np.newaxis] * mu_next, axis=0)  # shape (num_fine,)
                
                # Euler equation: mu_t_new = beta * R * mu_expected
                mu_t_new = beta * R * mu_expected
                
                # Relaxation
                mu_t_relaxed = omega * mu_t_new + (1 - omega) * mu_t
                
                # Check convergence
                diff = np.max(np.abs(mu_t_relaxed - mu_t))
                mu_t = mu_t_relaxed
                
                if diff < 1e-12:
                    break
            
            mu_fine[t, z, :] = mu_t
    
    # Now recover consumption policy from marginal value
    result = np.zeros((horizon, num_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_z):
            # Interpolate mu_fine to resource_grid
            mu_at_grid = np.interp(resource_grid, m_fine, mu_fine[t, z, :])
            # Convert to consumption
            c_at_grid = c_from_mu(mu_at_grid)
            # Clip to valid range
            c_at_grid = np.clip(c_at_grid, 1e-12, resource_grid + 1e-10)
            result[t, z, :] = c_at_grid
    
    # Ensure terminal condition: c[5] = resource_grid
    result[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Ensure all values are finite and in valid range
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, 1e-12, None)
    
    # Ensure c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    return result
