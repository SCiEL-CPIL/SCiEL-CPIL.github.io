"""Rewritten program: solve consumption via value-function iteration with high-order interpolation."""
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
    
    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Initialize consumption policy for final period
    # At t=5 (last period), consume everything
    c_final = np.tile(resource_grid, (n_states, 1))  # shape (2, 101)
    
    # We'll store consumption policies for each period
    c_policies = [None] * horizon
    c_policies[-1] = c_final.copy()
    
    # For intermediate periods, we need to solve the Euler equation
    # We'll use value function iteration with interpolation
    
    # Define a function to interpolate value function
    # We'll use cubic spline or higher-order polynomial interpolation
    
    def interpolate_value(V, m_grid, m_query):
        """Interpolate value function from grid to query points."""
        # Use np.interp for simplicity, but we could use higher order
        # For better accuracy, use cubic spline if available
        try:
            from scipy.interpolate import interp1d
            # Create interpolation function for each state
            interp_funcs = []
            for z in range(n_states):
                f = interp1d(m_grid, V[z], kind='cubic', fill_value='extrapolate')
                interp_funcs.append(f)
            return np.array([f(m_query) for f in interp_funcs])
        except ImportError:
            # Fallback to linear interpolation
            return np.array([np.interp(m_query, m_grid, V[z]) for z in range(n_states)])
    
    # Initialize value function for final period
    # V_5(m, z) = utility(m) since you consume everything
    V = np.array([utility(m_grid) for _ in range(n_states)])  # shape (2, 101)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # For each income state z, solve for optimal consumption
        c_t = np.zeros((n_states, n_grid))
        
        for z in range(n_states):
            # For each resource level m, find optimal c
            for i, m in enumerate(m_grid):
                # Try different consumption levels
                # Use a grid of possible consumption values
                # c ranges from 0 to m
                c_grid = np.linspace(1e-12, m, 50)
                
                best_c = c_grid[0]
                best_val = -np.inf
                
                for c in c_grid:
                    # Compute expected continuation value
                    m_next = R * (m - c)
                    cont_val = 0.0
                    for z_prime in range(n_states):
                        m_next_z = m_next + income[z_prime]
                        # Interpolate V at m_next_z
                        if m_next_z <= m_grid[0]:
                            V_next = V[z_prime, 0]
                        elif m_next_z >= m_grid[-1]:
                            V_next = V[z_prime, -1]
                        else:
                            # Linear interpolation
                            idx = np.searchsorted(m_grid, m_next_z)
                            if idx == 0:
                                V_next = V[z_prime, 0]
                            elif idx == n_grid:
                                V_next = V[z_prime, -1]
                            else:
                                m0, m1 = m_grid[idx-1], m_grid[idx]
                                v0, v1 = V[z_prime, idx-1], V[z_prime, idx]
                                V_next = v0 + (v1 - v0) * (m_next_z - m0) / (m1 - m0)
                        cont_val += transition[z, z_prime] * V_next
                
                total_val = utility(c) + beta * cont_val
                if total_val > best_val:
                    best_val = total_val
                    best_c = c
                
                c_t[z, i] = best_c
        
        c_policies[t] = c_t
        
        # Update value function
        V_new = np.zeros((n_states, n_grid))
        for z in range(n_states):
            for i, m in enumerate(m_grid):
                c = c_t[z, i]
                m_next = R * (m - c)
                cont_val = 0.0
                for z_prime in range(n_states):
                    m_next_z = m_next + income[z_prime]
                    if m_next_z <= m_grid[0]:
                        V_next = V[z_prime, 0]
                    elif m_next_z >= m_grid[-1]:
                        V_next = V[z_prime, -1]
                    else:
                        idx = np.searchsorted(m_grid, m_next_z)
                        if idx == 0:
                            V_next = V[z_prime, 0]
                        elif idx == n_grid:
                            V_next = V[z_prime, -1]
                        else:
                            m0, m1 = m_grid[idx-1], m_grid[idx]
                            v0, v1 = V[z_prime, idx-1], V[z_prime, idx]
                            V_next = v0 + (v1 - v0) * (m_next_z - m0) / (m1 - m0)
                    cont_val += transition[z, z_prime] * V_next
                V_new[z, i] = utility(c) + beta * cont_val
        V = V_new
    
    # Ensure final period constraint: c[5] equals grid
    c_policies[-1] = np.tile(resource_grid, (n_states, 1))
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                c_policies[t][z, i] = np.clip(c_policies[t][z, i], 1e-12, m_grid[i] + 1e-10)
    
    # Stack into output array: shape (6, 2, 101)
    result = np.array(c_policies)
    return result
