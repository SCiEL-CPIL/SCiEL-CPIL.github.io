import numpy as np
import time
import scipy.interpolate

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Internal grid for value function
    # Focus resolution on the scoring region [0.2, 8]
    # Create a dense grid in [0.1, 10] and sparse elsewhere
    m_low = np.geomspace(0.001, 0.1, 50)
    m_mid = np.geomspace(0.1, 10.0, 1500)
    m_high = np.geomspace(10.0, 50.0, 50)
    m_internal = np.concatenate([m_low, m_mid, m_high])
    # Ensure unique and sorted
    m_internal = np.unique(m_internal)
    n_m_int = len(m_internal)

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    # Policy storage: C[t, z, m_idx]
    C = np.zeros((horizon, n_z, n_m_int))

    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    C[horizon - 1, :, :] = m_internal[np.newaxis, :]

    # Grid for consumption fraction s = c/m
    # 1000 points is likely sufficient and fast
    n_s = 1000
    s_grid = np.linspace(0.0001, 1.0, n_s)

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Pre-compute interpolators for V[t+1] for each z'
        # Use PchipInterpolator for monotonicity and stability
        interpolators = []
        for zp in range(n_z):
            interpolators.append(scipy.interpolate.PchipInterpolator(m_internal, V[t + 1, zp, :]))

        for z in range(n_z):
            m_col = m_internal[:, np.newaxis]  # (n_m_int, 1)
            s_row = s_grid[np.newaxis, :]      # (1, n_s)

            c_mat = m_col * s_row              # (n_m_int, n_s)
            c_mat = np.maximum(c_mat, 1e-12)

            u_c = utility(c_mat)               # (n_m_int, n_s)

            total_val = np.zeros_like(u_c)     # (n_m_int, n_s)

            for zp in range(n_z):
                m_next = R * (m_col - c_mat) + income[zp]  # (n_m_int, n_s)
                
                # Clip to grid range for interpolation evaluation
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])

                # Evaluate interpolator
                v_next = interpolators[zp](m_next_clipped)
                
                total_val += beta_trans[z, zp] * v_next

            total_val += u_c

            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
            C[t, z, :] = c_mat[np.arange(n_m_int), best_idx]

            # Enforce c <= m
            C[t, z, :] = np.minimum(C[t, z, :], m_internal)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)

            # Time check
            if time.time() - start_time > deadline - 5:
                # If we run out of time, we can't continue safely.
                # However, with n_m_int ~ 1600 and n_s=1000, it should be fast.
                # 1600*1000 = 1.6e6. 5*2*1.6e6 = 16e6 ops.
                # This is very fast in numpy.
                break

        if time.time() - start_time > deadline - 5:
            break

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))

    for t in range(horizon):
        for z in range(n_z):
            # Use PchipInterpolator for final interpolation as well
            if n_m_int > 3:
                spl = scipy.interpolate.PchipInterpolator(m_internal, C[t, z, :])
                c_out = spl(m_grid_out)
            else:
                c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out

            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
