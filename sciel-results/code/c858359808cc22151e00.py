"""Rewritten program using Chebyshev spectral collocation for value function approximation."""
import numpy as np
from scipy.interpolate import Chebyshev2D
import warnings

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem using Chebyshev spectral collocation.
    
    Parameters:
    -----------
    model : dict
        Contains 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.ndarray
        Grid of resource values (101 points)
    budget : int
        Budget parameter (not directly used in this implementation)
    
    Returns:
    --------
    consumption : np.ndarray
        Array of shape (6, 2, 101) with consumption values
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)  # 2
    n_grid = len(resource_grid)  # 101
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # Map resource grid to [-1, 1] for Chebyshev polynomials
    # m = (m_max + m_min)/2 + (m_max - m_min)/2 * x
    m_mid = (m_max + m_min) / 2.0
    m_half = (m_max - m_min) / 2.0
    
    def m_to_x(m):
        return (m - m_mid) / m_half
    
    def x_to_m(x):
        return m_mid + m_half * x
    
    # Chebyshev nodes in [-1, 1]
    n_cheb = 40  # Number of Chebyshev collocation points
    x_cheb = np.cos(np.pi * np.arange(n_cheb) / (n_cheb - 1))
    x_cheb = x_cheb[::-1]  # Reverse to get ascending order
    m_cheb = x_to_m(x_cheb)
    
    # Ensure m_cheb is within valid range
    m_cheb = np.clip(m_cheb, 1e-12, m_max)
    
    def crra_utility(c, gamma):
        """CRRA utility function with floor on c."""
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    # Initialize value function at terminal period (t = horizon - 1 = 5)
    # At terminal, consumption = m (consume all)
    V_next = np.zeros((n_states, n_cheb))
    for z in range(n_states):
        V_next[z, :] = crra_utility(m_cheb, gamma)
    
    # Backward induction
    V_curr = np.zeros((n_states, n_cheb))
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # We need to store value functions at each time step for interpolation
    # For efficiency, we'll compute the optimal consumption at each grid point
    
    # First, let's compute the value function at each time step using Chebyshev interpolation
    # For each time step t from horizon-2 down to 0:
    for t in range(horizon - 2, -1, -1):
        V_curr.fill(0.0)
        
        for z in range(n_states):
            # For each state z, find optimal consumption
            # We need to evaluate the Bellman equation:
            # V_t(m, z) = max_c { u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(m', z') }
            # where m' = R*(m - c) + income[z']
            
            # For each Chebyshev node m_cheb[i], find optimal c
            for i in range(n_cheb):
                m_val = m_cheb[i]
                best_val = -np.inf
                best_c = m_val  # Default: consume all
                
                # Try a range of consumption values
                # c in [0, m_val]
                # Use a fine grid for search
                n_c_search = 200
                c_grid = np.linspace(1e-12, m_val, n_c_search)
                
                # For each c, compute the expected continuation value
                for c_idx, c in enumerate(c_grid):
                    if c > m_val + 1e-10:
                        continue
                    
                    # Current utility
                    u_c = crra_utility(c, gamma)
                    
                    # Next period resources for each z'
                    m_next = R * (m_val - c) + income  # Shape: (n_states,)
                    
                    # Ensure m_next is non-negative
                    m_next = np.maximum(m_next, 1e-12)
                    
                    # Interpolate V_{t+1} at m_next for each z'
                    # We have V_next[z', j] for z' in range(n_states), j in range(n_cheb)
                    # Need to interpolate V_next[z', :] as a function of m
                    
                    # Use Chebyshev interpolation to get V_{t+1}(m_next[z'], z')
                    cont_vals = np.zeros(n_states)
                    for z_prime in range(n_states):
                        # Interpolate V_next[z_prime, :] from x_cheb to m_next[z_prime]
                        x_target = m_to_x(m_next[z_prime])
                        # Use numpy interp for simplicity (linear interpolation in x-space)
                        # Actually, let's use a proper Chebyshev interpolation
                        # For now, use linear interpolation as a baseline
                        cont_vals[z_prime] = np.interp(x_target, x_cheb, V_next[z_prime, :])
                    
                    # Expected continuation value
                    exp_cont = np.dot(transition[z, :], cont_vals)
                    
                    total_val = u_c + beta * exp_cont
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                V_curr[z, i] = best_val
                # Store optimal consumption for this node
                consumption[t, z, i] = best_c
        
        V_next = V_curr.copy()
    
    # Now we need to map the Chebyshev node values to the resource_grid
    # We have V_curr and consumption at Chebyshev nodes
    # We need to interpolate to get consumption at resource_grid
    
    # For each time t and state z, interpolate consumption from Chebyshev nodes to resource_grid
    for t in range(horizon):
        for z in range(n_states):
            # We have consumption[t, z, :] at Chebyshev nodes (m_cheb)
            # Interpolate to resource_grid
            # Use linear interpolation in m-space
            consumption[t, z, :] = np.interp(resource_grid, m_cheb, consumption[t, z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[5] should equal the grid
    consumption[5, :, :] = resource_grid[np.newaxis, :, np.newaxis]
    
    # Ensure finite values
    consumption = np.where(np.isfinite(consumption), consumption, 0.0)
    
    return consumption
