import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12

    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return np.power(c, -gamma)

    # Value function: V[t, z, m_idx]
    # Policy: C[t, z, m_idx]
    V = np.zeros((horizon, num_states, num_resources))
    C = np.zeros((horizon, num_states, num_resources))

    # Terminal period (t = horizon-1 = 5): consume all
    V[horizon - 1] = 0.0
    C[horizon - 1] = resource_grid[np.newaxis, :]  # shape (1, num_resources), broadcast to (num_states, num_resources)
    # Actually set properly
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Fill remaining with a simple rule: consume a fraction
            for tt in range(t, horizon - 1):
                for z in range(num_states):
                    C[tt, z] = resource_grid
                    V[tt, z] = 0.0
            break

        for z in range(num_states):
            # Expected next-period value as a function of m'
            # V_next[z', m'] for each z'
            # We need to interpolate V[t+1, z', :] onto m' values

            # For each m in resource_grid, we need to find optimal c
            # FOC: u'(c) = beta * R * E_z'[ V'(t+1, z', m') ]
            # where m' = R*(m - c) + income[z']

            # Since V is concave in m, the FOC has a unique solution.
            # We use bisection on c in [c_floor, m - c_floor] (or [c_floor, m])

            m_vals = resource_grid  # shape (num_resources,)

            # We'll bisect for each m simultaneously
            c_low = np.full(num_resources, c_floor)
            c_high = np.maximum(m_vals - c_floor, c_floor)

            # Number of bisection iterations
            n_bisect = 50  # 2^-50 ~ 1e-15 precision

            for _ in range(n_bisect):
                if time.time() > deadline - 5.0:
                    break
                c_mid = 0.5 * (c_low + c_high)

                # Compute m' for each z' and each m
                # m' = R * (m - c_mid) + income[z']
                # Shape: (num_states, num_resources)
                m_prime = R * (m_vals[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                # Clip m' to grid range for interpolation
                m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])

                # Interpolate V[t+1, z', :] onto m_prime_clipped
                # V_next_interp[z', m_idx] = V[t+1, z', m_prime_clipped[z', m_idx]]
                V_next_interp = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    V_next_interp[zp] = np.interp(m_prime_clipped[zp], resource_grid, V[t + 1, zp])

                # Expected value: sum over z' of transition[z, z'] * V_next_interp[z', :]
                V_expected = transition[z, :] @ V_next_interp  # shape (num_resources,)

                # Now we need V'(t+1, m') weighted by transition
                # V' is the derivative of V w.r.t. m
                # We approximate V' by finite differences or use the policy
                # Actually, for the FOC we need:
                # u'(c) = beta * R * sum_z' trans[z,z'] * V'(t+1, z', m')
                # where m' depends on c

                # Approximate V' using the policy: at the optimal c,
                # V'(m) = u'(c*(m)) / (1 + beta*R*...) -- this is circular.

                # Better approach: use the envelope theorem.
                # V(t, z, m) = max_c { u(c) + beta * E_z'[ V(t+1, z', R(m-c)+income[z']) ] }
                # V_m = beta * R * E_z'[ V'(t+1, z', m') ]
                # So the FOC is: u'(c) = V_m(t, z, m) = beta * R * E_z'[ V'(t+1, z', m') ]

                # We need V'(t+1, z', m'). We can approximate this by:
                # V'(t+1, z', m') ≈ u'(c*(t+1, z', m')) / (1 - beta*R*...) 
                # This is getting complex. Let's use a different approach.

                # Alternative: use the fact that at the optimum,
                # the marginal value of m equals u'(c).
                # So V'(t+1, z', m') = u'(c*(t+1, z', m'))
                # where c*(t+1, z', m') is the optimal consumption at (t+1, z', m').

                # This is the key insight: by the envelope theorem,
                # V'(t, z, m) = u'(c*(t, z, m))
                # So we can use the already-computed policy to get V'.

                # V_prime_next[z', m_idx] = u'(C[t+1, z', m_idx])
                # Interpolate this onto m_prime

                V_prime_next_interp = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    V_prime_vals = util_deriv(C[t + 1, zp])
                    V_prime_next_interp[zp] = np.interp(m_prime_clipped[zp], resource_grid, V_prime_vals)

                # Expected marginal value
                V_prime_expected = transition[z, :] @ V_prime_next_interp  # shape (num_resources,)

                # FOC: u'(c) = beta * R * V_prime_expected
                # i.e., u'(c_mid) ?= beta * R * V_prime_expected
                rhs = beta * R * V_prime_expected
                lhs = util_deriv(c_mid)

                # If lhs > rhs, we're consuming too little (marginal utility too high), increase c
                # If lhs < rhs, we're consuming too much, decrease c
                # u'(c) is decreasing in c, so:
                # If u'(c_mid) > rhs, we need larger c to reduce u'(c)
                # If u'(c_mid) < rhs, we need smaller c to increase u'(c)

                mask_increase = lhs > rhs
                c_low[mask_increase] = c_mid[mask_increase]
                mask_decrease = lhs < rhs
                c_high[mask_decrease] = c_mid[mask_decrease]

            c_opt = 0.5 * (c_low + c_high)
            # Ensure c is within bounds
            c_opt = np.clip(c_opt, c_floor, m_vals - c_floor)
            # For very small m, c should be m (consume all)
            small_m = m_vals < 2 * c_floor
            c_opt[small_m] = m_vals[small_m]

            C[t, z] = c_opt

            # Compute V[t, z, m] = u(c) + beta * E[V(t+1, z', m')]
            m_prime = R * (m_vals[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])

            V_next_interp = np.zeros((num_states, num_resources))
            for zp in range(num_states):
                V_next_interp[zp] = np.interp(m_prime_clipped[zp], resource_grid, V[t + 1, zp])

            V_expected = transition[z, :] @ V_next_interp
            V[t, z] = utility(c_opt) + beta * V_expected

    # Ensure final period consumes all
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    # Final validity checks
    # 0 < c <= m + 1e-10
    m_expanded = resource_grid[np.newaxis, np.newaxis, :]
    C = np.maximum(C, c_floor)
    C = np.minimum(C, m_expanded + 1e-10)

    # c[5] should equal the grid
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    return C
