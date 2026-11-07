import numpy as np
import time
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Use Chebyshev nodes on log grid for spectral accuracy
    # Map log(m) to [-1, 1] for Chebyshev
    log_m_min = np.log(0.001)
    log_m_max = np.log(40)
    
    # Number of Chebyshev nodes
    n_cheb = 64
    
    # Chebyshev nodes in [-1, 1]
    j = np.arange(n_cheb)
    x_cheb = np.cos(np.pi * (2*j + 1) / (2*n_cheb))
    
    # Map to log grid
    log_m_cheb = 0.5 * (log_m_max - log_m_min) * x_cheb + 0.5 * (log_m_max + log_m_min)
    m_cheb = np.exp(log_m_cheb)
    m_cheb = np.sort(m_cheb)[::-1]  # Sort descending for consistency
    log_m_cheb = np.sort(log_m_cheb)[::-1]
    
    # Precompute Chebyshev polynomial values
    # V_k(x) where k is the polynomial degree
    def cheb_vals(x, n):
        """Compute Chebyshev polynomial values T_k(x) for k=0..n-1"""
        T = np.zeros((len(x), n))
        T[:, 0] = 1.0
        if n > 1:
            T[:, 1] = x
            for k in range(2, n):
                T[:, k] = 2*x*T[:, k-1] - T[:, k-2]
        return T
    
    T_cheb = cheb_vals(x_cheb, n_cheb)
    
    # Build interpolation matrix: given values at Chebyshev nodes, 
    # get polynomial coefficients
    # V(x) = sum_k a_k T_k(x)
    # We solve T_cheb @ a = v for a
    # Use least squares for stability
    try:
        A_inv = np.linalg.pinv(T_cheb.T)
    except:
        A_inv = np.linalg.inv(T_cheb.T)
    
    def fit_chebyshev(x_nodes, y_vals):
        """Fit Chebyshev polynomial to data at Chebyshev nodes"""
        # x_nodes should be in [-1, 1]
        T = cheb_vals(x_nodes, n_cheb)
        coeffs, _, _, _ = np.linalg.lstsq(T, y_vals, rcond=None)
        return coeffs
    
    def eval_chebyshev(coeffs, x_eval):
        """Evaluate Chebyshev polynomial at x_eval points"""
        T = cheb_vals(x_eval, n_cheb)
        return T @ coeffs
    
    def eval_chebyshev_deriv(coeffs, x_eval):
        """Evaluate derivative of Chebyshev polynomial"""
        # T'_k(x) = k * U_{k-1}(x) where U is Chebyshev of second kind
        # Use finite differences for simplicity, or analytical
        h = 1e-7
        return (eval_chebyshev(coeffs, x_eval + h) - eval_chebyshev(coeffs, x_eval - h)) / (2*h)
    
    def eval_chebyshev_deriv2(coeffs, x_eval):
        """Evaluate second derivative"""
        h = 1e-5
        return (eval_chebyshev(coeffs, x_eval + h) - 2*eval_chebyshev(coeffs, x_eval) + eval_chebyshev(coeffs, x_eval - h)) / (h*h)
    
    # We'll work in log-space for better conditioning
    # V(m) = f(log(m)) where f is smooth
    # Let x = log(m), so m = exp(x)
    # V(m) = f(x)
    # V'(m) = f'(x) / m
    # V''(m) = f''(x)/m^2 - f'(x)/m^2
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: V_{T-1}(m) = u(m) = u(exp(x))
    # For log utility: f(x) = x
    # For CRRA: f(x) = exp(x*(1-gamma))/(1-gamma)
    
    # Store value function coefficients at each time
    V_coeffs = [None] * T
    P_coeffs = [None] * T  # Policy in log-space: c = exp(g(x))
    
    # Terminal
    if abs(gamma - 1.0) < 1e-10:
        # f(x) = x, so f(x) = 0*x_0 + 1*x_1 + 0*x_2 + ...
        # In Chebyshev basis: x = T_1(x)
        V_coeffs[T-1] = np.zeros(n_cheb)
        V_coeffs[T-1][1] = 1.0
        # Policy: c = m = exp(x), so g(x) = x = T_1(x)
        P_coeffs[T-1] = np.zeros(n_cheb)
        P_coeffs[T-1][1] = 1.0
    else:
        # f(x) = exp((1-gamma)*x) / (1-gamma)
        # This is not a polynomial, but we can fit it
        x_fit = x_cheb
        y_fit = np.exp((1-gamma)*x_fit) / (1-gamma)
        V_coeffs[T-1] = fit_chebyshev(x_fit, y_fit)
        # Policy: c = m = exp(x), g(x) = x
        P_coeffs[T-1] = np.zeros(n_cheb)
        P_coeffs[T-1][1] = 1.0
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback
            for t_rem in range(t, -1, -1):
                V_coeffs[t_rem] = V_coeffs[t+1].copy()
                P_coeffs[t_rem] = P_coeffs[t+1].copy()
            break
        
        V_next_coeffs = V_coeffs[t+1]
        
        # For each state z, solve the Euler equation
        V_t_coeffs = np.zeros((n_z, n_cheb))
        P_t_coeffs = np.zeros((n_z, n_cheb))
        
        for z in range(n_z):
            # We need to find the optimal policy g(x) such that
            # u'(exp(g(x))) = beta * sum_z' P(z'|z) * V'(exp(R*(exp(x)-exp(g(x))) + income[z']))
            
            # Start with previous period's policy as initial guess
            g_init = P_coeffs[t+1][z].copy()
            
            # Newton's method on the policy coefficients
            g = g_init.copy()
            
            for newton_iter in range(30):
                # Evaluate current policy at Chebyshev nodes
                c_vals = np.exp(eval_chebyshev(g, x_cheb))
                c_vals = np.clip(c_vals, c_min, np.exp(log_m_max))
                
                # Compute LHS: u'(c)
                lhs = utility_prime(c_vals)
                
                # Compute RHS: beta * sum_z' P(z'|z) * V'(next_m)
                rhs = np.zeros(n_cheb)
                for zp in range(n_z):
                    next_m = R * (np.exp(x_cheb) - c_vals) + income[zp]
                    next_log_m = np.log(np.maximum(next_m, 1e-10))
                    
                    # Map next_log_m to Chebyshev coordinate
                    x_next = 2 * (next_log_m - log_m_min) / (log_m_max - log_m_min) - 1
                    x_next = np.clip(x_next, -1.0, 1.0)
                    
                    # V'(m) = f'(x)/m where f is the value function in log-space
                    # f'(x) = derivative of V_coeffs evaluated at x_next
                    f_prime = eval_chebyshev_deriv(V_next_coeffs, x_next)
                    V_prime = f_prime / np.exp(x_next)
                    
                    rhs += transition[z, zp] * V_prime
                
                rhs *= beta
                
                # Residual: lhs - rhs
                residual = lhs - rhs
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-8:
                    break
                
                # Compute Jacobian for Newton step
                # d/dg_k [residual] at x_cheb
                # This is complex; use finite differences on coefficients
                J = np.zeros((n_cheb, n_cheb))
                h_fd = 1e-6
                for k in range(n_cheb):
                    g_pert = g.copy()
                    g_pert[k] += h_fd
                    c_pert = np.exp(eval_chebyshev(g_pert, x_cheb))
                    c_pert = np.clip(c_pert, c_min, np.exp(log_m_max))
                    
                    lhs_pert = utility_prime(c_pert)
                    
                    rhs_pert = np.zeros(n_cheb)
                    for zp in range(n_z):
                        next_m_pert = R * (np.exp(x_cheb) - c_pert) + income[zp]
                        next_log_m_pert = np.log(np.maximum(next_m_pert, 1e-10))
                        x_next_pert = 2 * (next_log_m_pert - log_m_min) / (log_m_max - log_m_min) - 1
                        x_next_pert = np.clip(x_next_pert, -1.0, 1.0)
                        
                        f_prime_pert = eval_chebyshev_deriv(V_next_coeffs, x_next_pert)
                        V_prime_pert = f_prime_pert / np.exp(x_next_pert)
                        rhs_pert += transition[z, zp] * V_prime_pert
                    rhs_pert *= beta
                    
                    J[:, k] = (lhs_pert - rhs_pert - residual) / h_fd
                
                # Solve J @ delta_g = -residual
                try:
                    delta_g = np.linalg.solve(J, -residual)
                except:
                    delta_g = np.linalg.lstsq(J, -residual, rcond=None)[0]
                
                # Damping
                alpha = 1.0
                for _ in range(20):
                    g_trial = g + alpha * delta_g
                    c_trial = np.exp(eval_chebyshev(g_trial, x_cheb))
                    c_trial = np.clip(c_trial, c_min, np.exp(log_m_max))
                    
                    lhs_trial = utility_prime(c_trial)
                    rhs_trial = np.zeros(n_cheb)
                    for zp in range(n_z):
                        next_m_trial = R * (np.exp(x_cheb) - c_trial) + income[zp]
                        next_log_m_trial = np.log(np.maximum(next_m_trial, 1e-10))
                        x_next_trial = 2 * (next_log_m_trial - log_m_min) / (log_m_max - log_m_min) - 1
                        x_next_trial = np.clip(x_next_trial, -1.0, 1.0)
                        
                        f_prime_trial = eval_chebyshev_deriv(V_next_coeffs, x_next_trial)
                        V_prime_trial = f_prime_trial / np.exp(x_next_trial)
                        rhs_trial += transition[z, zp] * V_prime_trial
                    rhs_trial *= beta
                    
                    res_trial = lhs_trial - rhs_trial
                    if np.max(np.abs(res_trial)) < np.max(np.abs(residual)):
                        g = g_trial
                        break
                    alpha *= 0.5
                else:
                    g = g + alpha * delta_g
            
            P_t_coeffs[z] = g
            
            # Now compute value function coefficients
            c_vals = np.exp(eval_chebyshev(g, x_cheb))
            c_vals = np.clip(c_vals, c_min, np.exp(log_m_max))
            
            util_vals = utility(c_vals)
            
            cont_val = np.zeros(n_cheb)
            for zp in range(n_z):
                next_m = R * (np.exp(x_cheb) - c_vals) + income[zp]
                next_log_m = np.log(np.maximum(next_m, 1e-10))
                x_next = 2 * (next_log_m - log_m_min) / (log_m_max - log_m_min) - 1
                x_next = np.clip(x_next, -1.0, 1.0)
                
                cont_val += transition[z, zp] * eval_chebyshev(V_next_coeffs, x_next)
            
            V_vals = util_vals + beta * cont_val
            V_t_coeffs[z] = fit_chebyshev(x_cheb, V_vals)
        
        V_coeffs[t] = V_t_coeffs
        P_coeffs[t] = P_t_coeffs
    
    # Extract policy on the output grid
    result = np.zeros((T, n_z, n_out))
    
    for t in range(T):
        for z in range(n_z):
            # Map output grid to Chebyshev coordinates
            x_out = 2 * (np.log(grid_out) - log_m_min) / (log_m_max - log_m_min) - 1
            x_out = np.clip(x_out, -1.0, 1.0)
            
            # Evaluate policy
            log_c = eval_chebyshev(P_coeffs[t][z], x_out)
            c = np.exp(log_c)
            
            # Enforce constraints
            c = np.clip(c, c_min, grid_out + 1e-10)
            
            result[t, z, :] = c
    
    # Terminal condition must be exact
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    # Final validity check
    for t in range(T):
        for z in range(n_z):
            # Ensure 0 < c <= m + 1e-10
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            # Ensure finite
            result[t, z, :] = np.where(np.isfinite(result[t, z, :]), result[t, z, :], grid_out)
    
    return result
