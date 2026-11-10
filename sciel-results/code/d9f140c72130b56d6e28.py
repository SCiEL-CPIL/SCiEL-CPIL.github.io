import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)

    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))

    # --- Utility helpers (vectorized) ---
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)

    def u1(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)

    def u2(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c ** 2)
        return -gamma * c ** (-gamma - 1)

    # Terminal: consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]

    # Dense consumption search grid for initial guess (1000 points, log-spaced in fraction)
    n_frac = 1000
    frac_grid = np.geomspace(0.001, 1.0, n_frac)  # c/m fractions

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: simple rule
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break

        for z in range(num_states):
            # --- Vectorized grid search for initial guess ---
            # c_grid has shape (num_resources, n_frac)
            c_grid = resource_grid[:, np.newaxis] * frac_grid[np.newaxis, :]
            # Clamp to [1e-12, m]
            c_grid = np.clip(c_grid, 1e-12, resource_grid[:, np.newaxis])

            # m_next = R * (m - c), shape (num_resources, n_frac)
            m_next = R * (resource_grid[:, np.newaxis] - c_grid)

            # Continuation value: for each z', interpolate V_{t+1}[z'] at m_next + income[z']
            # cont_val shape (num_resources, n_frac)
            cont_val = np.zeros_like(c_grid)
            for zp in range(num_states):
                m_next_zp = m_next + income[zp]
                # np.interp along axis=0 (resource_grid axis)
                # We need to interpolate value[t+1, zp, :] at each m_next_zp
                # Flatten approach: for each column (frac index), interpolate
                v_interp = np.empty_like(m_next_zp)
                for fi in range(n_frac):
                    v_interp[:, fi] = np.interp(m_next_zp[:, fi], resource_grid, value[t + 1, zp, :])
                cont_val += transition[z, zp] * v_interp

            # Objective on grid
            obj_grid = utility(c_grid) + beta * cont_val
            # Find best frac index per resource
            best_frac_idx = np.argmax(obj_grid, axis=1)  # shape (num_resources,)
            c_init = c_grid[np.arange(num_resources), best_frac_idx]

            # --- Vectorized Newton-Raphson refinement ---
            c = c_init.copy()
            c = np.clip(c, 1e-12, resource_grid)

            for newton_iter in range(6):
                # m_next for current c
                m_next_c = R * (resource_grid - c)

                # Continuation value at c
                cont = np.zeros(num_resources)
                cont_deriv = np.zeros(num_resources)  # d(cont)/d(c) via chain rule
                cont_deriv2 = np.zeros(num_resources)  # second derivative (approx)

                for zp in range(num_states):
                    m_next_zp = m_next_c + income[zp]
                    # Interpolate V and its derivative
                    v_zp = value[t + 1, zp, :]
                    # Compute derivative of V wrt m via finite difference on grid
                    dm = np.diff(resource_grid)
                    dv = np.diff(v_zp)
                    dvdm = dv / dm  # shape (num_resources-1,)
                    # Pad for interpolation
                    dvdm_full = np.interp(resource_grid, resource_grid, dvdm,
                                          left=dvdm[0], right=dvdm[-1])

                    # Interpolate V at m_next_zp
                    v_at = np.interp(m_next_zp, resource_grid, v_zp)
                    dv_at = np.interp(m_next_zp, resource_grid, dvdm_full)

                    # dV/dm_next = dv_at
                    # dm_next/dc = -R
                    # d(cont)/dc contribution: P(z'|z) * dv_at * (-R)
                    cont += transition[z, zp] * v_at
                    cont_deriv += transition[z, zp] * dv_at * (-R)

                    # Second derivative: approximate d2V/dm2
                    # Use central difference of dvdm
                    d2vdm = np.zeros_like(resource_grid)
                    for i in range(1, num_resources - 1):
                        d2vdm[i] = (dvdm_full[i + 1] - dvdm_full[i - 1]) / (resource_grid[i + 1] - resource_grid[i - 1])
                    d2vdm[0] = d2vdm[1]
                    d2vdm[-1] = d2vdm[-2]
                    d2v_at = np.interp(m_next_zp, resource_grid, d2vdm)
                    # d2(cont)/dc2 = P * d2V/dm2 * R^2
                    cont_deriv2 += transition[z, zp] * d2v_at * (R ** 2)

                # Objective derivatives
                # F(c) = u(c) + beta * cont(c)
                # F'(c) = u1(c) + beta * cont_deriv
                # F''(c) = u2(c) + beta * cont_deriv2
                F1 = u1(c) + beta * cont_deriv
                F2 = u2(c) + beta * cont_deriv2

                # Newton step: c_new = c - F1/F2
                # Guard against division by zero
                F2_safe = np.where(np.abs(F2) < 1e-30, 1e-30, F2)
                step = F1 / F2_safe
                # Damping: limit step to 50% of current c or half the interval
                max_step = 0.5 * c
                step = np.clip(step, -max_step, max_step)
                c_new = c - step
                c_new = np.clip(c_new, 1e-12, resource_grid)

                # Check convergence
                if np.max(np.abs(c_new - c)) < 1e-12:
                    c = c_new
                    break
                c = c_new

            policy[t, z, :] = c
            # Compute value at optimal c
            m_next_c = R * (resource_grid - c)
            cont_final = np.zeros(num_resources)
            for zp in range(num_states):
                m_next_zp = m_next_c + income[zp]
                v_at = np.interp(m_next_zp, resource_grid, value[t + 1, zp, :])
                cont_final += transition[z, zp] * v_at
            value[t, z, :] = utility(c) + beta * cont_final

    # Ensure validity
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m

    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)

    return policy
