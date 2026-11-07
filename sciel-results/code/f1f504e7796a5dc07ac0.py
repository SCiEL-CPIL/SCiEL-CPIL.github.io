import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # Fine grid for value function representation
    fine_M = 1201
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # Precompute utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Backward induction on fine grid
    consumption_fine = np.zeros((horizon, Z, fine_M))
    V = np.zeros((horizon + 1, Z, fine_M))

    # Precompute indices for interpolation
    log_fine_grid = np.log(fine_grid)
    
    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            for m_idx in range(fine_M):
                m = fine_grid[m_idx]

                if t == horizon - 1:
                    # Last period: must consume all
                    c = m
                    V[t, z, m_idx] = utility(c)
                    consumption_fine[t, z, m_idx] = c
                    continue

                # Use Euler equation to find optimal c via Newton's method
                # Euler: u'(c) = beta * E[u'(c') * R]
                # where c' depends on z'
                
                # Initial guess: use a fraction based on beta
                # For CRRA, the steady state saving rate is roughly (beta*R)^(1/gamma) - 1
                c_guess = m * min(0.8, max(0.2, 1.0 - (beta * R) ** (1.0 / max(gamma, 0.1)) + 0.1))
                c_guess = min(max(c_guess, 1e-8), m * 0.999)
                
                # Newton iterations
                for newton_iter in range(10):
                    # Check time
                    if time.time() > deadline - 30.0:
                        break
                    
                    # Compute LHS: u'(c)
                    lhs = marginal_utility(c_guess)
                    
                    # Compute RHS: beta * E[u'(c') * R]
                    rhs = 0.0
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c_guess) + income[zp]
                        if m_prime < 1e-12:
                            m_prime = 1e-12
                        # Interpolate V[t+1, zp, :] to get c_{t+1}
                        if m_prime >= fine_grid[-1]:
                            c_next = fine_grid[-1]
                        elif m_prime <= fine_grid[0]:
                            c_next = fine_grid[0]
                        else:
                            c_next = np.interp(m_prime, fine_grid, consumption_fine[t + 1, zp, :])
                        c_next = min(max(c_next, 1e-12), m_prime)
                        rhs += p * marginal_utility(c_next) * R
                    
                    rhs *= beta
                    
                    # If we're past the last period that has consumption computed, use analytical
                    if t + 1 == horizon - 1:
                        # Next period is last, so c' = m'
                        rhs = 0.0
                        for zp in range(Z):
                            p = transition[z, zp]
                            if p < 1e-15:
                                continue
                            m_prime = R * (m - c_guess) + income[zp]
                            m_prime = max(m_prime, 1e-12)
                            rhs += p * marginal_utility(m_prime) * R
                        rhs *= beta
                    
                    # Newton step: f(c) = u'(c) - beta*E[u'(c')*R] = 0
                    # f'(c) = u''(c) - beta*E[u''(c')*R' * dc'/dc]
                    # u''(c) = -gamma * c^(-gamma-1) for gamma != 1, -1/c^2 for gamma=1
                    # dc'/dc = -R (since m' = R*(m-c) + income, dm'/dc = -R)
                    
                    # Numerical derivative for robustness
                    h = 1e-6 * max(c_guess, 1e-8)
                    c_plus = min(c_guess + h, m * 0.9999)
                    c_minus = max(c_guess - h, 1e-10)
                    
                    def euler_residual(c_val):
                        ml = marginal_utility(c_val)
                        mr = 0.0
                        for zp in range(Z):
                            p = transition[z, zp]
                            if p < 1e-15:
                                continue
                            mp = R * (m - c_val) + income[zp]
                            mp = max(mp, 1e-12)
                            if t + 1 == horizon - 1:
                                mr += p * marginal_utility(mp) * R
                            else:
                                if mp >= fine_grid[-1]:
                                    cn = fine_grid[-1]
                                elif mp <= fine_grid[0]:
                                    cn = fine_grid[0]
                                else:
                                    cn = np.interp(mp, fine_grid, consumption_fine[t + 1, zp, :])
                                cn = min(max(cn, 1e-12), mp)
                                mr += p * marginal_utility(cn) * R
                        return ml - beta * mr
                    
                    f_val = euler_residual(c_guess)
                    f_plus = euler_residual(c_plus)
                    f_minus = euler_residual(c_minus)
                    
                    # Numerical derivative
                    if c_plus > c_minus:
                        f_deriv = (f_plus - f_minus) / (c_plus - c_minus)
                    else:
                        f_deriv = -gamma * c_guess ** (-gamma - 1) if abs(gamma - 1) > 1e-10 else -1.0 / (c_guess ** 2)
                    
                    if abs(f_deriv) < 1e-15:
                        break
                    
                    # Newton update
                    c_new = c_guess - f_val / f_deriv
                    c_new = min(max(c_new, 1e-10), m * 0.9999)
                    
                    if abs(c_new - c_guess) < 1e-10 * max(1.0, c_guess):
                        c_guess = c_new
                        break
                    c_guess = c_new
                
                # Final validation: ensure 0 < c <= m
                c_opt = min(max(c_guess, 1e-10), m)
                
                # Compute value
                u_c = utility(c_opt)
                cont = 0.0
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime = max(m_prime, 1e-12)
                    if m_prime >= fine_grid[-1]:
                        v_next = V[t + 1, zp, -1]
                    elif m_prime <= fine_grid[0]:
                        v_next = V[t + 1, zp, 0]
                    else:
                        v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                    cont += p * v_next
                
                V[t, z, m_idx] = u_c + beta * cont
                consumption_fine[t, z, m_idx] = c_opt

        # Check deadline
        if time.time() > deadline - 20.0:
            break

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
