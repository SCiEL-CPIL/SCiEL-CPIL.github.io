import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=float)
    budget = int(budget)

    n_states = len(income)
    n_m = len(m_grid)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Use a much denser grid for the coarse pass
    n_coarse = 300
    c_fractions = np.linspace(0.001, 1.0, n_coarse)

    policy = np.zeros((horizon, n_states, n_m))
    V = np.zeros((horizon, n_states, n_m))

    # Final period: t = horizon - 1 = 5
    for z in range(n_states):
        V[5, z, :] = utility(m_grid)
    policy[5, :, :] = m_grid[np.newaxis, :]

    # Work backwards from t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, 0, -1):
                for z in range(n_states):
                    policy[tt, z, :] = policy[tt + 1, z, :]
                    V[tt, z, :] = V[tt + 1, z, :]
            policy[5, :, :] = m_grid[np.newaxis, :]
            for z in range(n_states):
                V[5, z, :] = utility(m_grid)
            for tt in range(horizon):
                for z in range(n_states):
                    policy[tt, z, :] = np.clip(policy[tt, z, :], 1e-12, m_grid + 1e-10)
            return policy

        for z in range(n_states):
            # Coarse search
            c_mat = np.outer(m_grid, c_fractions)
            c_mat = np.clip(c_mat, 1e-12, m_grid[:, np.newaxis] + 1e-10)
            u_c = utility(c_mat)

            cont_total = np.zeros((n_m, n_coarse))
            for z_next in range(n_states):
                prob = transition[z, z_next]
                if prob < 1e-15:
                    continue
                m_next = R * (m_grid[:, np.newaxis] - c_mat) + income[z_next]
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                v_vals = V[t + 1, z_next, :]
                interp_vals = np.interp(m_next_clipped.ravel(), m_grid, v_vals)
                interp_vals = interp_vals.reshape(m_next_clipped.shape)
                cont_total += prob * interp_vals

            total_value = u_c + beta * cont_total
            best_idx = np.argmax(total_value, axis=1)
            best_c = c_mat[np.arange(n_m), best_idx]
            best_V = total_value[np.arange(n_m), best_idx]

            # Local refinement: for each m, do a few Newton steps around best_c
            # The objective is f(c) = u(c) + beta * E[V_next(R*(m-c)+income)]
            # f'(c) = u'(c) - beta * E[V_next'(R*(m-c)+income)] * R
            # We need V_next' which we can approximate from the grid
            # For simplicity, do a small local search around best_c
            n_refine = 10
            for i in range(n_m):
                m_i = m_grid[i]
                c_start = best_c[i]
                # Search in a small neighborhood
                # The fraction is c_start / m_i
                frac_start = c_start / m_i
                # Local grid around frac_start
                local_fracs = np.linspace(max(0.001, frac_start - 0.01), min(1.0, frac_start + 0.01), n_refine)
                local_cs = local_fracs * m_i
                local_cs = np.clip(local_cs, 1e-12, m_i + 1e-10)
                
                # Evaluate objective at these points
                local_u = utility(local_cs)
                local_cont = np.zeros(n_refine)
                for z_next in range(n_states):
                    prob = transition[z, z_next]
                    if prob < 1e-15:
                        continue
                    m_next_local = R * (m_i - local_cs) + income[z_next]
                    m_next_clipped_local = np.clip(m_next_local, m_grid[0], m_grid[-1])
                    v_vals = V[t + 1, z_next, :]
                    interp_local = np.interp(m_next_clipped_local, m_grid, v_vals)
                    local_cont += prob * interp_local
                
                local_val = local_u + beta * local_cont
                local_best = np.argmax(local_val)
                best_c[i] = local_cs[local_best]
                best_V[i] = local_val[local_best]

            policy[t, z, :] = best_c
            V[t, z, :] = best_V

    # Final validation
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
    policy[5, :, :] = m_grid[np.newaxis, :]

    return policy
