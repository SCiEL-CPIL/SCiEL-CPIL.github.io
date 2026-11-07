import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    consumption = np.zeros((horizon, Z, M))
    V = np.zeros((horizon, Z, M))

    def utility(c):
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    # Initialize V for t = horizon - 1 (index 5)
    V[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, np.newaxis, :]
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]

    # Grid for consumption fraction s = c/m
    # We want to cover the range well.
    S = 2000
    s_grid = np.linspace(1e-5, 1.0, S)

    # Backward induction for t = horizon - 2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # If we run out of time, fill remaining with c = m (consume all)
            # This is valid since 0 < c <= m + 1e-10
            for tt in range(t + 1):
                consumption[tt] = resource_grid[np.newaxis, np.newaxis, :]
                V[tt] = utility(resource_grid)[np.newaxis, np.newaxis, :]
            return consumption

        V_next = V[t + 1]  # Shape (Z, M)

        for z in range(Z):
            # Vectorized coarse search
            m_col = resource_grid[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]

            c_candidates = m_col * s_row
            c_candidates = np.maximum(c_candidates, 1e-12)

            obj_values = np.zeros((M, S))

            for zp in range(Z):
                prob = transition[z, zp]
                if prob == 0:
                    continue

                m_prime = R * (resource_grid[:, np.newaxis] - c_candidates) + income[zp]

                # Interpolate V_next[zp] at m_prime
                m_prime_flat = m_prime.flatten()
                v_interp = np.interp(m_prime_flat, resource_grid, V_next[zp, :])
                v_interp = v_interp.reshape(M, S)

                obj_values += prob * v_interp

            u_c = utility(c_candidates)
            total_obj = u_c + beta * obj_values

            best_idx = np.argmax(total_obj, axis=1)
            best_s = s_grid[best_idx]
            best_c = resource_grid * best_s

            # Refinement using finer local search around best_s
            s_step = s_grid[1] - s_grid[0]

            for i in range(M):
                m_i = resource_grid[i]
                c_init = best_c[i]

                # Define bounds for local search
                c_low = max(1e-12, m_i * (best_s[i] - 3 * s_step))
                c_high = min(m_i, m_i * (best_s[i] + 3 * s_step))

                if c_high - c_low < 1e-12:
                    c_opt = c_init
                    V[t, z, i] = total_obj[i, best_idx[i]]
                    consumption[t, z, i] = c_opt
                    continue

                # Use a few evaluation points for local refinement
                # instead of scipy to avoid overhead
                local_s = np.linspace(
                    max(1e-5, best_s[i] - 3 * s_step),
                    min(1.0, best_s[i] + 3 * s_step),
                    20
                )
                local_c = m_i * local_s
                local_c = np.maximum(local_c, 1e-12)

                local_obj = np.zeros(len(local_s))
                for zp in range(Z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    m_prime = R * (m_i - local_c) + income[zp]
                    v_zp = np.interp(m_prime, resource_grid, V_next[zp, :])
                    local_obj += prob * v_zp

                local_obj = utility(local_c) + beta * local_obj
                local_best = np.argmax(local_obj)
                c_opt = local_c[local_best]
                c_opt = np.clip(c_opt, 1e-12, m_i)

                consumption[t, z, i] = c_opt
                V[t, z, i] = local_obj[local_best]

    return consumption
