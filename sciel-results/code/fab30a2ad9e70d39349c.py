"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)

    # Final period: consume everything
    C = np.zeros((horizon, n_z, n_grid))
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]
    V_next = np.zeros((n_z, n_grid))  # V at t+1, all zeros for final period

    # Backward induction with log-linear value function approximation
    # For CRRA, the value function is approximately log-linear in m
    # V(m) ≈ A + B * log(m)
    # We'll use this to get smooth derivatives for the FOC

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        # For each state z, find optimal consumption for each m on the grid
        for z in range(n_z):
            for i_m in range(n_grid):
                m = grid[i_m]
                if m <= 1e-12:
                    C[t, z, i_m] = 1e-12
                    continue

                # Search over consumption ratio c/m in [0.001, 1]
                # Use a fine grid in ratio space
                n_ratio = 200
                ratios = np.linspace(0.001, 1.0, n_ratio)
                c_vals = ratios * m
                c_vals = np.clip(c_vals, 1e-12, m + 1e-10)

                best_c = c_vals[0]
                best_val = -np.inf

                # Vectorized evaluation for speed
                for idx in range(len(c_vals)):
                    c = c_vals[idx]
                    m_next_base = R * (m - c)
                    cont_val = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next_z = m_next_base + income[z_next]
                        if m_next_z < grid[0]:
                            v_next = np.interp(grid[0], grid, V_next[z_next])
                        elif m_next_z > grid[-1]:
                            v_next = np.interp(grid[-1], grid, V_next[z_next])
                        else:
                            v_next = np.interp(m_next_z, grid, V_next[z_next])
                        cont_val += prob * v_next
                    total_val = u(c) + beta * cont_val
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c

                C[t, z, i_m] = best_c

        # Update V_next for the next iteration (moving backward)
        V_curr = np.zeros((n_z, n_grid))
        for z in range(n_z):
            for i_m in range(n_grid):
                m = grid[i_m]
                c = C[t, z, i_m]
                m_next_base = R * (m - c)
                cont_val = 0.0
                for z_next in range(n_z):
                    prob = transition[z, z_next]
                    m_next_z = m_next_base + income[z_next]
                    if m_next_z < grid[0]:
                        v_next = np.interp(grid[0], grid, V_next[z_next])
                    elif m_next_z > grid[-1]:
                        v_next = np.interp(grid[-1], grid, V_next[z_next])
                    else:
                        v_next = np.interp(m_next_z, grid, V_next[z_next])
                    cont_val += prob * v_next
                V_curr[z, i_m] = u(c) + beta * cont_val
        V_next = V_curr

    return _finalize(C, grid, horizon, n_z, n_grid)

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

def _finalize(C, grid, horizon, n_z, n_grid):
    # Ensure c[5] (final period) equals grid
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]

    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            m_vals = grid
            c_vals = C[t, z, :]
            c_vals = np.clip(c_vals, 1e-12, m_vals + 1e-10)
            C[t, z, :] = c_vals

    return C
# EVOLVE-BLOCK-END
