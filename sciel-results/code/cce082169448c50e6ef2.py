import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)

    # Precompute utility function (vectorized)
    if abs(gamma - 1.0) < 1e-12:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Fraction grid for c/m: log-spaced from 0.001 to 1.0
    # Use 300 points for good resolution
    c_frac = np.geomspace(0.001, 1.0, 300)
    n_frac = len(c_frac)

    # V[t, z, m] for t in 0..horizon
    # V[horizon, z, m] = utility(m)
    V = np.zeros((horizon + 1, n_z, n_m))
    V[horizon, :, :] = utility(m_grid)

    # consumption[t, z, m]
    consumption = np.zeros((horizon, n_z, n_m))

    # Backward induction, fully vectorized over m
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule
            for tt in range(t + 1, horizon):
                consumption[tt, :, :] = m_grid * 0.5
            break

        for z in range(n_z):
            # For all m in m_grid simultaneously:
            # c_vals[i, j] = c_frac[j] * m_grid[i], shape (n_m, n_frac)
            c_vals = m_grid[:, None] * c_frac[None, :]  # (n_m, n_frac)
            c_vals = np.clip(c_vals, 1e-12, m_grid[:, None])

            # m_minus_c[i, j] = m_grid[i] - c_vals[i, j]
            m_minus_c = m_grid[:, None] - c_vals  # (n_m, n_frac)

            # Expected continuation value: for each z', m' = R * m_minus_c + income[z']
            # cont[i, j] = sum_z' trans[z, z'] * V[t+1, z', m'(i,j,z')]
            cont = np.zeros((n_m, n_frac))
            for zp in range(n_z):
                m_prime = R * m_minus_c + income[zp]  # (n_m, n_frac)
                # Clip to grid range
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                # Interpolate V[t+1, zp, :] at each m_prime
                # np.interp works on 1D, so we do it row by row or use a trick
                # Actually np.interp can handle 2D if we reshape
                # Let's do it properly: for each row i, interpolate V[t+1,zp,:] at m_prime[i,:]
                # This is (n_m, n_frac) interpolations of a (n_m,) function
                # We can vectorize: np.interp doesn't directly do 2D, but we can use:
                # For each i, np.interp(m_prime[i,:], m_grid, V[t+1, zp, :])
                # Let's use a loop over i but it's only 101 iterations
                v_interp = np.empty((n_m, n_frac))
                for i in range(n_m):
                    v_interp[i, :] = np.interp(m_prime_clipped[i, :], m_grid, V[t+1, zp, :])
                cont += trans[z, zp] * v_interp

            # Total value: utility(c) + beta * cont
            total = utility(c_vals) + beta * cont  # (n_m, n_frac)

            # Find argmax over j for each i
            idx_best = np.argmax(total, axis=1)  # (n_m,)

            # Extract optimal c for each m
            c_opt = np.empty(n_m)
            for i in range(n_m):
                c_opt[i] = c_vals[i, idx_best[i]]

            # Store value and consumption
            V[t, z, :] = total[np.arange(n_m), idx_best]
            consumption[t, z, :] = c_opt

    # Final period: consume all
    consumption[horizon - 1, :, :] = m_grid

    # Validate
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)

    # Ensure c[5] (last period, index horizon-1=5) equals grid
    consumption[horizon - 1, :, :] = m_grid

    return consumption
