import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # Fine grid for value function representation
    # Use a dense grid for better interpolation accuracy
    fine_M = 501
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # c-grid for optimization: use fractions of m
    # For each m, c ranges from small fraction to m
    # We'll use a fixed number of c-points per m
    c_frac = np.linspace(0.001, 1.0, 200)

    # Precompute utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Backward induction on fine grid
    # consumption_fine[t, z, m_idx] for fine grid
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Terminal condition: at t=horizon, c = m
    # We work backwards from t = horizon-1 to 0
    # V_{t+1}[z', m'] is the continuation value

    # Initialize: at t = horizon, the "next period" value is 0 (terminal)
    # Actually, at t = horizon - 1, the next period is t = horizon where c = m
    # So V_horizon[m] = utility(m) for all m (but this is the terminal utility)
    # Let me think more carefully.

    # The problem: max sum_{t=0}^{horizon-1} beta^t * u(c_t)
    # Subject to: m_{t+1} = R*(m_t - c_t) + income[z_{t+1}]
    # Terminal: c_{horizon} = m_{horizon} (but this isn't in the sum)
    # Wait, the constraint says c[5] equals grid within 1e-9, and horizon=6
    # So t=0..5, and at t=5 (last period), c must equal m.

    # Actually re-reading: "c[5] equal to the grid within 1e-9"
    # So at the final period t=5, consumption = resources.

    # Let's define value function V_t(m, z) = max expected utility from t to horizon-1
    # V_horizon(m, z) = 0 (no more periods)
    # V_t(m, z) = max_c { u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c)+income[z'], z') }
    # For t = horizon-1 = 5: c = m, so V_5(m,z) = u(m)

    # Let's work with this formulation
    # V is indexed by [t, z, m_idx] on fine grid

    V = np.zeros((horizon + 1, Z, fine_M))
    # V[horizon] = 0

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

                # Try different consumption levels
                # c must satisfy 0 < c <= m
                # m' = R*(m - c) + income[z']
                # Need m' > 0 for all z' (or at least handle it)

                best_val = -np.inf
                best_c = m  # default: consume all

                # c ranges from small to m
                c_vals = m * c_frac
                c_vals = c_vals[c_vals <= m + 1e-10]
                c_vals = c_vals[c_vals > 0]

                if len(c_vals) == 0:
                    V[t, z, m_idx] = utility(1e-12)
                    consumption_fine[t, z, m_idx] = 1e-12
                    continue

                for c in c_vals:
                    if c > m + 1e-10:
                        continue
                    m_minus_c = m - c
                    if m_minus_c < -1e-10:
                        continue

                    u_c = utility(c)

                    # Expected continuation value
                    cont = 0.0
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * m_minus_c + income[zp]
                        if m_prime < 1e-12:
                            m_prime = 1e-12
                        # Interpolate V[t+1, zp, :] at m_prime
                        if m_prime >= fine_grid[-1]:
                            v_next = V[t + 1, zp, -1]
                        elif m_prime <= fine_grid[0]:
                            v_next = V[t + 1, zp, 0]
                        else:
                            v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                        cont += p * v_next

                    total = u_c + beta * cont
                    if total > best_val:
                        best_val = total
                        best_c = c

                V[t, z, m_idx] = best_val
                consumption_fine[t, z, m_idx] = best_c

        # Check deadline
        if time.time() > deadline - 20.0:
            # Need to extract what we have and return
            break

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            # Interpolate consumption_fine[t, z, :] onto resource_grid
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            # c must be > 0
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            # c must be <= m + 1e-10
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
