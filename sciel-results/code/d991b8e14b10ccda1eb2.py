import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']

    num_states = len(income)
    orig_grid = resource_grid
    num_resources = len(resource_grid)
    c_floor = 1e-12

    # Dense working grid for high accuracy
    n_work = 1001
    work_grid = np.geomspace(0.001, 40, n_work)
    m_min = work_grid[0]
    m_max = work_grid[-1]

    # Utility and derivatives
    if gamma == 1:
        def utility(c):
            return np.log(np.maximum(c, c_floor))
        def u_prime(c):
            return 1.0 / np.maximum(c, c_floor)
        def u_double_prime(c):
            return -1.0 / (np.maximum(c, c_floor) ** 2)
    else:
        def utility(c):
            return np.power(np.maximum(c, c_floor), 1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            return np.power(np.maximum(c, c_floor), -gamma)
        def u_double_prime(c):
            return -gamma * np.power(np.maximum(c, c_floor), -gamma - 1)

    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))

    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid

    # Precompute transition probabilities
    nonzero_probs = []
    for z in range(num_states):
        nz = [zp for zp in range(num_states) if transition[z, zp] > 1e-15]
        nonzero_probs.append(nz)

    # Backward induction using Newton-Raphson with analytic derivatives
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        # V'_{t+1}(z', m) = u'(c*(z', m))
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = u_prime(policy_work[t + 1, zp, :])

        for z in range(num_states):
            probs = transition[z]
            nz = nonzero_probs[z]

            m = work_grid

            # Initial guess: smooth heuristic
            if gamma == 1:
                alpha = 0.25 / (m * 0.3 + 0.5)
                alpha = np.clip(alpha, 0.02, 0.45)
            else:
                alpha = 0.3 / (m * 0.2 + 0.8)
                alpha = np.clip(alpha, 0.03, 0.5)
            c = m * (1.0 - alpha)
            c = np.clip(c, c_floor, m)

            # Newton-Raphson iterations
            max_iter = 30
            tol = 1e-13
            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # f(c) = u'(c) - beta*R*E[V'(m_next)]
                lhs = u_prime(c)

                rhs = np.zeros(n_work)
                for zp in nz:
                    m_next = R * (m - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_min, m_max)
                    vprime_vals = np.interp(m_next_clipped, work_grid, vp_next[zp, :])
                    rhs += probs[zp] * vprime_vals
                rhs = beta * R * rhs

                f_val = lhs - rhs

                # Analytic derivative: f'(c) = u''(c) - beta*R*E[V''(m_next)*(-R)]
                # V''(m_next) = u''(c*(m_next)) * dc/dm_next
                # We approximate dc/dm_next using the policy gradient
                # More precisely: d/dc [beta*R*E[V'(m_next)]] = beta*R*E[V''(m_next) * dm_next/dc]
                # dm_next/dc = -R
                # So f'(c) = u''(c) + beta*R^2*E[V''(m_next)]
                # where V''(m_next) = u''(c*(m_next)) * (dc/dm_next)
                # But we don't have dc/dm_next directly. Instead, use:
                # f'(c) = u''(c) + beta*R*E[d/dm_next V'(m_next) * (-R)]
                # = u''(c) - beta*R^2*E[V''(m_next)]
                # where V''(m_next) is the second derivative of the value function.
                # By the envelope theorem, V'(m) = u'(c*(m)), so
                # V''(m) = u''(c*(m)) * c'(m)
                # We can estimate c'(m) from the policy on the grid.

                # Simpler: use numerical derivative of f with respect to c
                dc = np.maximum(np.abs(c) * 1e-6, 1e-12)
                c_plus = np.clip(c + dc, c_floor, m)

                lhs_plus = u_prime(c_plus)
                rhs_plus = np.zeros(n_work)
                for zp in nz:
                    m_next_plus = R * (m - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, m_min, m_max)
                    vprime_vals_plus = np.interp(m_next_plus_clipped, work_grid, vp_next[zp, :])
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus

                f_plus = lhs_plus - rhs_plus
                df_dc = (f_plus - f_val) / dc

                # Newton step
                newton_c = c - f_val / np.where(np.abs(df_dc) > 1e-15, df_dc, 1e-15)

                # Clamp to valid range
                newton_c = np.clip(newton_c, c_floor, m)

                # Check convergence
                delta_c = np.abs(newton_c - c)
                c = newton_c

                if np.max(delta_c) < tol:
                    break

            policy_work[t, z, :] = c

            # Compute value function
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp in nz:
                m_next = R * (m - c) + income[zp]
                m_next_clipped = np.clip(m_next, m_min, m_max)
                v_next = np.interp(m_next_clipped, work_grid, value_work[t + 1, zp, :])
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val

    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, orig_grid)

    # Terminal period must equal grid exactly
    for z in range(num_states):
        policy[horizon - 1, z, :] = orig_grid

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], c_floor)
            policy[t, z, :] = np.minimum(policy[t, z, :], orig_grid + 1e-10)

    return policy
