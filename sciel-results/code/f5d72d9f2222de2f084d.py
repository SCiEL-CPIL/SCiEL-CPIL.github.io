import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # Use a fine geometric grid for the value function
    fine_M = 1201
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # Precompute utility and marginal utility
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

    # Backward induction
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            m_vals = fine_grid
            c_guess = np.zeros(fine_M)

            # Initial guess: use a coarse grid search
            n_c = 200
            c_fracs = np.linspace(0.002, 1.0, n_c)

            best_val = -np.inf * np.ones(fine_M)
            best_c = np.zeros(fine_M)

            for c_frac in c_fracs:
                c_vals = m_vals * c_frac
                c_vals = np.minimum(c_vals, m_vals)
                c_vals = np.maximum(c_vals, 1e-12)

                m_minus_c = m_vals - c_vals
                m_minus_c = np.maximum(m_minus_c, 0)

                cont = np.zeros(fine_M)
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime = R * m_minus_c + income[zp]
                    m_prime = np.maximum(m_prime, 1e-12)
                    m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                    v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                    cont += p * v_next

                total = utility(c_vals) + beta * cont

                mask = total > best_val
                best_val[mask] = total[mask]
                best_c[mask] = c_vals[mask]

            c_guess = best_c
            consumption_fine[t, z, :] = c_guess

            # Newton refinement on the Euler equation
            for newton_iter in range(5):
                if time.time() > deadline - 30.0:
                    break

                residual = np.zeros(fine_M)
                deriv = np.zeros(fine_M)
                
                h = 1e-5
                for i in range(fine_M):
                    c = c_guess[i]
                    m = m_vals[i]

                    mu_c = marginal_utility(c)

                    exp_mu_cp = 0.0
                    exp_term = 0.0
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])

                        c_prime = np.interp(m_prime, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime = np.maximum(c_prime, 1e-12)

                        mu_cp = marginal_utility(c_prime)
                        exp_mu_cp += p * mu_cp

                        # Derivative calculation
                        m_prime_plus = np.clip(m_prime + h, fine_grid[0], fine_grid[-1])
                        m_prime_minus = np.clip(m_prime - h, fine_grid[0], fine_grid[-1])
                        c_prime_plus = np.interp(m_prime_plus, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime_minus = np.interp(m_prime_minus, fine_grid, consumption_fine[t + 1, zp, :])
                        dc_dm = (c_prime_plus - c_prime_minus) / (m_prime_plus - m_prime_minus + 1e-15)

                        if abs(gamma - 1.0) < 1e-10:
                            u2_cp = -1.0 / (c_prime ** 2)
                        else:
                            u2_cp = -gamma * c_prime ** (-gamma - 1)

                        exp_term += p * u2_cp * dc_dm * (-R)

                    residual[i] = mu_c - beta * R * exp_mu_cp

                    if abs(gamma - 1.0) < 1e-10:
                        u2_c = -1.0 / (c ** 2)
                    else:
                        u2_c = -gamma * c ** (-gamma - 1)

                    deriv[i] = u2_c - beta * R * exp_term

                # Check convergence
                if np.max(np.abs(residual)) < 1e-9:
                    break

                for i in range(fine_M):
                    if abs(deriv[i]) > 1e-12:
                        c_new = c_guess[i] - residual[i] / deriv[i]
                        c_guess[i] = np.clip(c_new, 1e-12, m_vals[i])

            consumption_fine[t, z, :] = c_guess

            # Compute V[t, z, :] using the optimal c
            c_vals = c_guess
            m_minus_c = np.maximum(m_vals - c_vals, 0)
            cont = np.zeros(fine_M)
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                m_prime = R * m_minus_c + income[zp]
                m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                cont += p * v_next
            V[t, z, :] = utility(c_vals) + beta * cont

        if time.time() > deadline - 20.0:
            break

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
