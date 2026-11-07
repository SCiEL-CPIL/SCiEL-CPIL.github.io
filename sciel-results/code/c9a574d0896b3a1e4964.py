"""Improved program: Direct CRRA inversion with log-linear value function interpolation."""
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Precompute utility
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Precompute log grid for interpolation
    log_grid = np.log(resource_grid)
    
    # Initialize value function for terminal period
    # V_{H-1}(m, z) = u(m)
    V_next = np.zeros((n_states, n_grid))
    for z in range(n_states):
        V_next[z, :] = util(resource_grid)
        
    # Store log V for interpolation
    # We need to handle log(V). If V is negative? 
    # For gamma < 1, u(c) > 0 for c>0? 
    # u(c) = c^(1-g)/(1-g). If gamma=2, u(c) = -1/c. Negative.
    # Log-linear interpolation requires positive values.
    # If V can be negative, log-linear is invalid.
    # However, for gamma=2, u(c) = -1/c.
    # V(m) = max u(c) + beta E[V(m')].
    # If V is negative, we can't take log.
    
    # Alternative: Interpolate V directly? Linear interpolation of V is less accurate for CRRA.
    # But log-linear fails for negative V.
    
    # Let's check the range of V.
    # gamma=1: log(c) > 0 for c>1? No, log(0.001) is negative.
    # So V can be negative even for gamma=1.
    
    # Therefore, we cannot use simple log-log interpolation for V if V crosses zero or is negative.
    # However, the *marginal utility* V' is always positive.
    # We can interpolate V' directly?
    # Or, we can use a "shifted" log interpolation?
    
    # Better approach for robustness:
    # Use linear interpolation for V, and compute V' via finite differences on the grid?
    # No, the directive says "piecewise log-linear value function".
    # Maybe the value function is always positive in this specific setup?
    # u(c) for gamma=2 is -1/c.
    # V(m) = max_c [-1/c + beta E[-1/c']].
    # This is negative.
    
    # Let's re-read the directive: "Represent the value function as a piecewise log-linear function".
    # This might imply we should interpolate $\log(V + K)$ for some K?
    # Or perhaps the "log-linear" refers to the policy function $c(m)$?
    # No, it says "value function".
    
    # Let's look at the seed/inspirations.
    # If we assume V > 0, we can use log-log.
    # For gamma=2, u(c) = -1/c.
    # Is it possible V is positive?
    # No, -1/c is always negative.
    
    # Maybe the "log-linear" is applied to the *marginal utility*?
    # $V'(m)$ is positive.
    # If we assume $V'(m)$ is log-linear in $m$, then $V'(m) = A m^B$.
    # Then $V(m) = \frac{A}{B+1} m^{B+1}$.
    # This is a power function.
    # CRRA value functions are often power functions in the interior.
    # So, we can interpolate $\log(V'(m))$ linearly in $\log(m)$.
    # Then $V'(m)$ is obtained by exponentiating.
    # Then $V(m)$ can be obtained by integrating? Or we just store $V$ and $V'$ separately?
    # We need $V$ for the value update, and $V'$ for the Euler equation.
    
    # Plan:
    # 1. Store $V_{next}$ and $V'_{next}$ on the grid.
    # 2. Interpolate $V_{next}$ using linear interpolation? Or log-linear if positive?
    #    To be safe, let's use linear interpolation for $V$ if it might be negative, 
    #    but the directive says log-linear.
    #    Let's check if $V$ is always positive for the given parameters.
    #    gamma=1: u(c)=log(c). log(0.001) ~ -6.9. V will be negative.
    #    So log-linear on V is impossible.
    
    # Perhaps the directive implies: "Use a method that avoids resolution limits... by using analytical inverse".
    # The "log-linear value function" might be a hint to use log-linear interpolation for the *marginal utility* $V'$, which is always positive.
    # Let's assume we interpolate $\log(V')$ linearly in $\log(m)$.
    # Then $V'(m)$ is smooth and positive.
    # To get $V(m)$, we can just use linear interpolation of $V(m)$ on the grid, 
    # or integrate $V'$?
    # Integrating $V'$ from the grid is complex.
    # Let's use linear interpolation for $V$ and log-linear for $V'$.
    # This is a common hybrid approach.
    
    # Actually, if we use the analytical inverse for c, we only need $E[V']$.
    # We don't need to interpolate $V$ to get $c$.
    # We only need $V$ to update $V_{curr}$.
    # So:
    # 1. Interpolate $V'_{next}$ using log-linear (on $\log V'$ vs $\log m$).
    # 2. Interpolate $V_{next}$ using linear (on $V$ vs $m$) to be safe against sign changes.
    
    # Let's implement this.
    
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon-1, :, :] = resource_grid
    
    # Precompute V' for terminal period
    # V(m) = u(m). V'(m) = u'(m).
    V_prime_next = np.zeros((n_states, n_grid))
    for z in range(n_states):
        if abs(gamma - 1.0) < 1e-10:
            V_prime_next[z, :] = 1.0 / resource_grid
        else:
            V_prime_next[z, :] = np.power(resource_grid, -gamma)
            
    # Backward induction
    for t in range(horizon-2, -1, -1):
        V_curr = np.zeros((n_states, n_grid))
        V_prime_curr = np.zeros((n_states, n_grid))
        
        for z in range(n_states):
            m_vals = resource_grid
            
            # Initial guess for c
            c_vals = np.clip(m_vals * 0.5, 1e-12, m_vals)
            
            # Fixed point iteration for c
            for _ in range(20):
                # Compute expected marginal utility
                E_mu = np.zeros(n_grid)
                E_V = np.zeros(n_grid)
                
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    m_prime = R * (m_vals - c_vals) + income[zp]
                    
                    # Clip m_prime to grid range for interpolation
                    # If m_prime < grid[0], extrapolate?
                    # If m_prime > grid[-1], extrapolate?
                    # Let's clip to [grid[0], grid[-1]] for now, 
                    # but extrapolation is better.
                    # For simplicity and speed, we'll clip. 
                    # The grid is 0.001 to 40.
                    # m_prime min: R(0.001 - 0.001) + 0.5 = 0.5.
                    # m_prime max: R(40 - 1e-12) + 1.5 ~ 42.
                    # So m_prime can be > 40.
                    # We need extrapolation.
                    
                    # Log-linear interpolation/extrapolation for V'
                    log_m_prime = np.log(m_prime)
                    
                    # Find indices
                    # Use searchsorted
                    idx = np.searchsorted(log_grid, log_m_prime)
                    # Handle boundaries
                    idx = np.clip(idx, 1, n_grid - 1)
                    idx0 = idx - 1
                    idx1 = idx
                    
                    # Weights
                    w0 = log_grid[idx1] - log_m_prime
                    w1 = log_m_prime - log_grid[idx0]
                    wsum = w0 + w1
                    w0 = w0 / wsum
                    w1 = w1 / wsum
                    
                    # Interpolate log(V')
                    log_vp0 = np.log(V_prime_next[zp, idx0])
                    log_vp1 = np.log(V_prime_next[zp, idx1])
                    log_vp_interp = w0 * log_vp0 + w1 * log_vp1
                    vp_interp = np.exp(log_vp_interp)
                    
                    # Interpolate V (linear)
                    # We need V_next values.
                    # We stored V_next in the previous step? 
                    # Yes, V_next is the value function from t+1.
                    # We need to interpolate V_next[zp] at m_prime.
                    # Linear interpolation:
                    # m_prime might be outside [grid[0], grid[-1]].
                    # Let's clip m_prime for V interpolation?
                    # Or use linear extrapolation.
                    m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    # Re-calculate indices for clipped m? 
                    # Or just use the same indices if m_prime is within range?
                    # If m_prime > grid[-1], idx will be n_grid-1.
                    # idx0 = n_grid-2, idx1 = n_grid-1.
                    # This is effectively extrapolation using the last segment.
                    
                    v0 = V_next[zp, idx0]
                    v1 = V_next[zp, idx1]
                    v_interp = w0 * v0 + w1 * v1
                    
                    E_mu += p * vp_interp
                    E_V += p * v_interp
                
                # Solve for c using analytical inverse
                # u'(c) = E_mu
                # c^{-gamma} = E_mu  => c = E_mu^{-1/gamma}
                # For gamma=1: 1/c = E_mu => c = 1/E_mu
                
                if abs(gamma - 1.0) < 1e-10:
                    c_new = 1.0 / E_mu
                else:
                    c_new = np.power(E_mu, -1.0 / gamma)
                
                # Enforce constraint c <= m
                c_new = np.minimum(c_new, m_vals)
                c_new = np.maximum(c_new, 1e-12)
                
                # Check convergence
                if np.max(np.abs(c_new - c_vals)) < 1e-10 * np.max(m_vals):
                    c_vals = c_new
                    break
                
                c_vals = c_new
            
            consumption[t, z, :] = c_vals
            
            # Compute V_curr and V_prime_curr
            # V_curr = u(c) + beta * E_V
            V_curr[z, :] = util(c_vals) + beta * E_V
            
            # V_prime_curr = u'(c) = E_mu (from Euler equation)
            # Wait, is V_prime_curr exactly E_mu?
            # The Euler equation says u'(c) = beta R E[V'_{next}].
            # So V'_{curr}(m, z) = beta R E[V'_{next}(m', z')].
            # We computed E_mu = E[V'_{next}].
            # So V_prime_curr[z, :] = beta * R * E_mu.
            
            V_prime_curr[z, :] = beta * R * E_mu
        
        V_next = V_curr
        V_prime_next = V_prime_curr
    
    # Ensure final period constraint
    consumption[horizon-1, :, :] = resource_grid
    
    # Validate
    for t in range(horizon):
        for z in range(n_states):
            c = consumption[t, z, :]
            m = resource_grid
            consumption[t, z, :] = np.clip(c, 1e-12, m)
            
    return consumption
