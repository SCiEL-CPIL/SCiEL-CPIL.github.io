import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_z = len(income)
    target_grid = resource_grid
    n_target = len(target_grid)
    
    # Dense grid for internal computation
    # Max m' approx 1.02 * 40 + 1.5 = 42.3
    m_max_internal = 50.0
    m_min_internal = 0.001
    N_dense = 2000
    m_dense = np.geomspace(m_min_internal, m_max_internal, N_dense)
    
    # Utility and derivative
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)
            
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        return c_safe ** (-gamma)

    # V[t, z, :] stores value function on m_dense
    # Policy[t, z, :] stores optimal consumption on m_dense
    V = np.zeros((horizon, n_z, N_dense))
    Policy = np.zeros((horizon, n_z, N_dense))
    
    # Terminal period t = horizon - 1 (index 5)
    t_last = horizon - 1
    for z in range(n_z):
        V[t_last, z, :] = u(m_dense)
        Policy[t_last, z, :] = m_dense.copy()
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # Interpolate V[t+1, z'] for each z'
            # We need derivatives of V[t+1, z'] with respect to m'
            splines = []
            for zp in range(n_z):
                # Ensure strict monotonicity for CubicSpline
                # m_dense is strictly increasing
                spl = CubicSpline(m_dense, V[t+1, zp, :])
                splines.append(spl)
            
            def get_cont_value_and_deriv(m_prime_vec):
                # m_prime_vec is array
                vals = np.zeros_like(m_prime_vec)
                derivs = np.zeros_like(m_prime_vec)
                for zp in range(n_z):
                    p = transition[z, zp]
                    if p > 0:
                        # Clip m_prime to grid range to avoid extrapolation issues if any
                        # CubicSpline handles extrapolation by default (constant slope), 
                        # but let's be safe.
                        mp = np.clip(m_prime_vec, m_dense[0], m_dense[-1])
                        vals += p * splines[zp](mp)
                        derivs += p * splines[zp](mp, 1)
                return vals, derivs
            
            def objective_deriv(c):
                # FOC: U'(c) - beta * R * sum_z' P(z'|z) V'_{t+1}(m', z') = 0
                c_safe = max(c, 1e-12)
                u_p = u_prime(c_safe)
                
                # m' = R(m - c) + income[z']
                # We need to evaluate for each m in m_dense? No, this is for a specific m.
                # This function is used inside brentq for a specific m.
                pass
            
            # We need to solve for c for each m in m_dense
            c_opt = np.zeros(N_dense)
            
            for i_m, m_val in enumerate(m_dense):
                # If m_val is very small, c might be 0? 
                # But U'(0) is inf, so c > 0.
                
                def f(c):
                    # c is scalar
                    c_safe = max(c, 1e-12)
                    u_p = u_prime(c_safe)
                    
                    # Calculate expected marginal continuation value
                    # m' = R(m_val - c) + income[zp]
                    m_prime_vec = R * (m_val - c) + income
                    # Clip to grid
                    m_prime_clipped = np.clip(m_prime_vec, m_dense[0], m_dense[-1])
                    
                    # Sum over z' of P(z'|z) * V'_{t+1}(m', z')
                    # We can use the splines directly
                    sum_deriv = 0.0
                    for zp in range(n_z):
                        p = transition[z, zp]
                        if p > 0:
                            # splines[zp](m_prime_clipped[zp], 1)
                            # Note: splines[zp] is a CubicSpline object
                            sum_deriv += p * splines[zp](m_prime_clipped[zp], 1)
                    
                    return u_p - beta * R * sum_deriv

                # Check boundaries
                # f(c) = U'(c) - beta * R * E[V']
                # As c -> 0, U'(c) -> inf (for gamma>0). So f(0) > 0.
                # As c -> m, U'(c) is finite. 
                # If f(m) > 0, then optimal c = m (corner).
                # If f(m) < 0, there is a root in (0, m).
                
                # Evaluate f at m
                # Note: if m is very small, c=m is the max.
                f_m = f(m_val)
                
                if f_m > 0:
                    # Optimal c is m
                    c_opt[i_m] = m_val
                else:
                    # Root in (1e-12, m_val)
                    try:
                        c_opt[i_m] = brentq(f, 1e-12, m_val, xtol=1e-12)
                    except ValueError:
                        # If no sign change (e.g. numerical issues), fallback to m
                        c_opt[i_m] = m_val
                        # Or maybe 0? But f(0)>0 usually.
                        # Let's check f(1e-12)
                        f_low = f(1e-12)
                        if f_low < 0:
                            c_opt[i_m] = 1e-12
                        else:
                            c_opt[i_m] = m_val

            Policy[t, z, :] = c_opt
            # Calculate V[t, z, :]
            # V = u(c) + beta * E[V_{t+1}]
            c_vals = Policy[t, z, :]
            u_vals = u(c_vals)
            
            # Calculate continuation value
            m_prime_vecs = R * (m_dense - c_vals) + income # Shape (N_dense, n_z)
            m_prime_clipped = np.clip(m_prime_vecs, m_dense[0], m_dense[-1])
            
            cont_val = np.zeros(N_dense)
            for zp in range(n_z):
                p = transition[z, zp]
                if p > 0:
                    cont_val += p * splines[zp](m_prime_clipped[:, zp])
            
            V[t, z, :] = u_vals + beta * cont_val

    # Interpolate Policy from m_dense to target_grid
    # Policy shape: (horizon, n_z, N_dense)
    # Target shape: (horizon, n_z, n_target)
    
    final_policy = np.zeros((horizon, n_z, n_target))
    
    for t in range(horizon):
        for z in range(n_z):
            # Use linear interpolation for policy? 
            # The policy is smooth. np.interp is linear.
            # Cubic spline interpolation of the policy might be better?
            # But policy is bounded by m.
            # Let's use np.interp first, it's robust.
            # Actually, since we solved on a dense grid, linear interp error is small.
            final_policy[t, z, :] = np.interp(target_grid, m_dense, Policy[t, z, :])
            
            # Enforce constraints
            final_policy[t, z, :] = np.clip(final_policy[t, z, :], 1e-12, target_grid + 1e-10)
            
    # Enforce terminal condition exactly
    final_policy[t_last, :, :] = target_grid[None, :]
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a simple policy? 
        # Unlikely to happen with this efficient method.
        pass
        
    return final_policy
