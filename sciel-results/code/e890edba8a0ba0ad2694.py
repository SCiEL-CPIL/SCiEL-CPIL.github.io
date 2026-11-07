"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import make_interp_spline

def solve_consumption(model, resource_grid, budget):
    """
    Solve the finite-horizon consumption-saving problem using Chebyshev
    spectral collocation for the value function representation.
    
    Returns consumption[t, income_state, resource_index] with shape (6, 2, 101).
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Chebyshev nodes and weights on [0, 1]
    # We'll map resource_grid to [0, 1] via log-scale since grid is geometric
    # Actually, let's use the actual resource_grid values directly with Chebyshev
    # on the log-scale for better conditioning
    
    # Map resource grid to a uniform interval for Chebyshev approximation
    # Use log-space since resource_grid is geometric
    log_grid = np.log(resource_grid)
    log_min = log_grid[0]
    log_max = log_grid[-1]
    
    # Normalize to [-1, 1]
    def to_cheb(m):
        return (np.log(m) - log_min) / (log_max - log_min) * 2.0 - 1.0
    
    def from_cheb(x):
        return np.exp(log_min + (x + 1.0) / 2.0 * (log_max - log_min))
    
    # Chebyshev points of degree n
    def chebyshev_points(n):
        """Return n Chebyshev-Gauss-Lobatto points on [-1, 1]."""
        k = np.arange(n)
        return np.cos(np.pi * (n - 1 - k) / (n - 1))
    
    # Degree of polynomial
    deg = 25
    n_cheb = deg + 1
    cheb_x = chebyshev_points(n_cheb)
    
    # Build Vandermonde matrix for Chebyshev polynomials
    def cheb_vandermonde(x, n):
        """Compute Chebyshev polynomial values T_0(x), ..., T_{n-1}(x)."""
        T = np.zeros((len(x), n))
        T[:, 0] = 1.0
        if n > 1:
            T[:, 1] = x
            for k in range(2, n):
                T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
        return T
    
    # Precompute Chebyshev basis at Chebyshev points
    T_cheb = cheb_vandermonde(cheb_x, n_cheb)
    
    # For each time step, we need to represent V_t(m, z) for each state z
    # V_t(m, z) = max_c { u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c)+income[z'], z') }
    
    # Terminal condition: c[m] = m (consume all)
    # So V_{horizon}(m, z) = u(m)
    
    # We'll work backwards from t = horizon-1 to t = 0
    
    # Store value functions as Chebyshev coefficients for each state
    # V_coeffs[t, z] = array of shape (n_cheb,)
    
    # At terminal: V_{horizon}(m, z) = u(m) for all z
    # Fit u(m) at Chebyshev points
    m_cheb = from_cheb(cheb_x)
    u_terminal = utility(m_cheb)
    
    # Fit terminal value function
    # Solve T_cheb @ coeffs = u_terminal
    try:
        V_terminal_coeffs = np.linalg.solve(T_cheb, u_terminal)
    except np.linalg.LinAlgError:
        V_terminal_coeffs = np.linalg.lstsq(T_cheb, u_terminal, rcond=None)[0]
    
    # For each state, terminal value is the same (u(m))
    V_coeffs_next = np.array([V_terminal_coeffs.copy() for _ in range(n_states)])
    
    consumption = np.zeros((horizon, n_states, n_grid))
    
    for t in range(horizon - 1, -1, -1):
        V_coeffs_curr = np.zeros_like(V_coeffs_next)
        
        for z in range(n_states):
            # We need to find optimal c for each m on the grid
            # V_t(m, z) = max_c { u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c)+income[z'], z') }
            
            # Use Chebyshev collocation: evaluate at Chebyshev points
            # For each Chebyshev point m_cheb[i], find optimal c
            
            V_curr_cheb = np.zeros(n_cheb)
            
            for i in range(n_cheb):
                m_val = m_cheb[i]
                
                # Search over consumption: c in [1e-12, m_val]
                # Use a fine grid for the search
                # The number of search points depends on budget
                n_search = min(budget, 200)
                c_grid = np.linspace(1e-12, m_val, n_search)
                
                # Compute continuation value for each c
                # m' = R * (m_val - c) + income[z']
                # For each z', compute V_{t+1}(m', z') using Chebyshev interpolation
                
                # m' values for each c and each z'
                m_prime = np.zeros((n_search, n_states))
                for zp in range(n_states):
                    m_prime[:, zp] = R * (m_val - c_grid) + income[zp]
                
                # Ensure m' >= 0 (clip to small positive)
                m_prime = np.maximum(m_prime, 1e-12)
                
                # Evaluate V_{t+1}(m', z') for each z'
                # Map m' to Chebyshev space
                m_prime_cheb = to_cheb(m_prime)
                
                # Clip to [-1, 1] for Chebyshev evaluation (extrapolation may be needed)
                # For now, clip to avoid issues
                m_prime_cheb_clipped = np.clip(m_prime_cheb, -1.0, 1.0)
                
                V_cont = np.zeros((n_search, n_states))
                T_prime = cheb_vandermonde(m_prime_cheb_clipped, n_cheb)
                
                for zp in range(n_states):
                    V_cont[:, zp] = T_prime @ V_coeffs_next[zp]
                
                # Weighted sum over z'
                probs = transition[z]
                expected_V = np.sum(probs * V_cont, axis=1)
                
                # Total value for each c
                total_value = utility(c_grid) + beta * expected_V
                
                # Find optimal c
                idx_best = np.argmax(total_value)
                c_opt = c_grid[idx_best]
                V_curr_cheb[i] = total_value[idx_best]
            
            # Fit V_curr_cheb to Chebyshev coefficients
            try:
                V_coeffs_curr[z] = np.linalg.solve(T_cheb, V_curr_cheb)
            except np.linalg.LinAlgError:
                V_coeffs_curr[z] = np.linalg.lstsq(T_cheb, V_curr_cheb, rcond=None)[0]
            
            # Also compute consumption on the actual resource_grid for output
            for j in range(n_grid):
                m_val = resource_grid[j]
                
                n_search = min(budget, 200)
                c_grid = np.linspace(1e-12, m_val, n_search)
                
                m_prime = np.zeros((n_search, n_states))
                for zp in range(n_states):
                    m_prime[:, zp] = R * (m_val - c_grid) + income[zp]
                
                m_prime = np.maximum(m_prime, 1e-12)
                m_prime_cheb = to_cheb(m_prime)
                m_prime_cheb_clipped = np.clip(m_prime_cheb, -1.0, 1.0)
                
                V_cont = np.zeros((n_search, n_states))
                T_prime = cheb_vandermonde(m_prime_cheb_clipped, n_cheb)
                
                for zp in range(n_states):
                    V_cont[:, zp] = T_prime @ V_coeffs_next[zp]
                
                probs = transition[z]
                expected_V = np.sum(probs * V_cont, axis=1)
                
                total_value = utility(c_grid) + beta * expected_V
                
                idx_best = np.argmax(total_value)
                c_opt = c_grid[idx_best]
                
                # Ensure 0 < c <= m + 1e-10
                c_opt = np.clip(c_opt, 1e-12, m_val + 1e-10)
                consumption[t, z, j] = c_opt
        
        V_coeffs_next = V_coeffs_curr
    
    # Terminal period: consume all
    for z in range(n_states):
        consumption[horizon - 1, z, :] = resource_grid
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_states):
            for j in range(n_grid):
                consumption[t, z, j] = np.clip(consumption[t, z, j], 1e-12, resource_grid[j] + 1e-10)
    
    # Final constraint: c[5] (last period) must equal grid within 1e-9
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, :]
    
    return consumption
# EVOLVE-BLOCK-END
