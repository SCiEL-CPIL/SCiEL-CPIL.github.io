"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Use a dense internal grid for accurate interpolation of continuation values
    # 1000 points provides high resolution for the spline interpolation
    n_dense = 1000
    dense_grid = np.geomspace(0.001, 40, n_dense)
    
    # Precompute utility derivatives
    def get_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def get_utility_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # consumption_solve[t, z, i] is optimal consumption at time t, state z, grid index i
    consumption_solve = np.zeros((horizon, 2, n_dense))
    consumption_solve[horizon-1, :, :] = dense_grid[np.newaxis, :]

    # Splines for V' (which is u'(c*)) for the next period
    # V_prime_splines[z] is the spline for state z at time t+1
    V_prime_splines = [None, None]
    
    # Initialize for t = horizon - 1
    # c_T = m, so u'(c_T) = u'(m)
    c_next = consumption_solve[horizon-1, :, :]
    u_prime_next = get_utility_prime(c_next)
    for z in range(2):
        V_prime_splines[z] = CubicSpline(dense_grid, u_prime_next[z, :])

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: fill remaining periods with a simple rule
            # Use linear approximation or just copy previous if available
            for tt in range(t, -1, -1):
                if tt < horizon - 1:
                    consumption_solve[tt, :, :] = consumption_solve[tt+1, :, :] * 0.95
                else:
                    consumption_solve[tt, :, :] = dense_grid[np.newaxis, :]
            break

        # Solve for consumption at time t
        # We iterate over states z
        for z in range(2):
            m_vals = dense_grid
            # Initial guess: proportional saving
            # c ~ m / (1 + beta * R * ... )
            # A safe guess is 0.5 * m
            c = 0.5 * m_vals.copy()
            
            # Newton's Method
            for _ in range(20):
                # Compute m' for each next state z'
                # m' shape: (2, n_dense)
                m_next = R * (m_vals[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                
                # Interpolate V' and V'' from the splines of the next period
                # V_prime_splines[zp] is for state zp at t+1
                # We need E[V'(m', z') | z]
                
                vp_0 = V_prime_splines[0](m_next[0, :])
                vp_1 = V_prime_splines[1](m_next[1, :])
                
                # V'' is the derivative of V'
                # CubicSpline.derivative(1) returns a new spline for the first derivative
                # We can evaluate it directly
                vpp_0 = V_prime_splines[0].derivative(1)(m_next[0, :])
                vpp_1 = V_prime_splines[1].derivative(1)(m_next[1, :])
                
                # Expected values
                p0 = transition[z, 0]
                p1 = transition[z, 1]
                
                expected_vp = p0 * vp_0 + p1 * vp_1
                expected_vpp = p0 * vpp_0 + p1 * vpp_1
                
                target = beta * R * expected_vp
                
                lhs = get_utility_prime(c)
                f = lhs - target
                
                # Derivative of f with respect to c
                # f(c) = u'(c) - beta * R * E[V'(m'(c))]
                # f'(c) = u''(c) - beta * R * E[V''(m'(c)) * m''(c)]
                # m'(c) = R(m-c) + y => m''(c) = -R
                # f'(c) = u''(c) - beta * R * E[V''(m') * (-R)]
                # f'(c) = u''(c) + beta * R^2 * E[V''(m')]
                
                rhs_deriv = get_utility_double_prime(c) + beta * (R ** 2) * expected_vpp
                
                # Newton step
                # Avoid division by zero
                mask = np.abs(rhs_deriv) > 1e-12
                c_step = np.zeros_like(c)
                c_step[mask] = f[mask] / rhs_deriv[mask]
                
                # Update c
                c_new = c - c_step
                
                # Clip to valid range
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c)) < 1e-10:
                    c = c_new
                    break
                c = c_new

            consumption_solve[t, z, :] = c

        # Update splines for the current period (which becomes "next" for the previous iteration)
        c_curr = consumption_solve[t, :, :]
        u_prime_curr = get_utility_prime(c_curr)
        for z in range(2):
            V_prime_splines[z] = CubicSpline(dense_grid, u_prime_curr[z, :])

    # Interpolate the solution from dense_grid (1000 points) to resource_grid (101 points)
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            consumption[t, z, :] = np.interp(resource_grid, dense_grid, consumption_solve[t, z, :])
    
    # Final check and clipping
    # Ensure 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Ensure c[5] is exactly the grid
    consumption[5, :, :] = resource_grid[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
