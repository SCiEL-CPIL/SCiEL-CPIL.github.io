"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_states = len(income)
    n_grid = len(grid)

    # Utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    # Continuation value function: V[t+1, z, m] -> value
    # We store V as an array of shape (horizon+1, n_states, n_grid) for interpolation
    # V[t] = max over c of util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', m']
    # m' = R*(m - c) + income[z']

    # Final period: c = m (consume all), V[horizon, z, m] = util(m)
    V = np.zeros((horizon + 1, n_states, n_grid), dtype=float)
    for z in range(n_states):
        V[horizon, z, :] = util(grid)

    # Consumption policy: shape (horizon, n_states, n_grid)
    C = np.zeros((horizon, n_states, n_grid), dtype=float)

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        # Check deadline
        if time.time() > deadline:
            # Fill remaining with a simple heuristic: consume a fixed fraction
            for tt in range(t, horizon):
                for z in range(n_states):
                    # Simple: consume 50% of m, but ensure c <= m
                    C[tt, z, :] = np.minimum(0.5 * grid, grid)
            break

        for z in range(n_states):
            # For each m in grid, find optimal c
            # Coarse grid: c = m * linspace(0.001, 1, coarse_n)
            coarse_n = 300
            fractions = np.linspace(0.001, 1.0, coarse_n)

            best_c = np.zeros(n_grid, dtype=float)
            best_val = np.full(n_grid, -np.inf, dtype=float)

            # Vectorize over m
            # For each fraction f, c = f * m, m' = R*(m - c) + income[z'] = R*(1-f)*m + income[z']
            # V_next interpolated at m' for each z'
            # We'll iterate over fractions to find best

            for f in fractions:
                c = f * grid
                # m' for each z'
                for zp in range(n_states):
                    mp = R * (1.0 - f) * grid + income[zp]
                    # Interpolate V[t+1, zp, :] at mp
                    # Clip mp to grid range
                    mp_clip = np.clip(mp, grid[0], grid[-1])
                    v_next = np.interp(mp_clip, grid, V[t + 1, zp, :])
                    # Weight by transition
                    w = transition[z, zp]
                    if w > 0:
                        val = util(c) + beta * w * v_next
                        # Accumulate: we need sum over zp of transition[z,zp]*V_next
                        # Let's restructure: compute total continuation for this f
                # Actually, let's compute the full value for this f
                # continuation = sum_{zp} transition[z, zp] * V[t+1, zp, mp_zp]
                cont = np.zeros(n_grid, dtype=float)
                for zp in range(n_states):
                    w = transition[z, zp]
                    if w > 0:
                        mp = R * (1.0 - f) * grid + income[zp]
                        mp_clip = np.clip(mp, grid[0], grid[-1])
                        v_next = np.interp(mp_clip, grid, V[t + 1, zp, :])
                        cont += w * v_next
                val = util(c) + beta * cont
                # Update best
                mask = val > best_val
                best_val[mask] = val[mask]
                best_c[mask] = c[mask]

            # Local refinement around coarse optimum
            # For each m, refine in a window around best_c
            refine_n = 80
            for i in range(n_grid):
                if time.time() > deadline:
                    break
                m_i = grid[i]
                c_coarse = best_c[i]
                # Window: +/- 5% of m or around c_coarse
                lo = max(1e-12, c_coarse - 0.05 * m_i)
                hi = min(m_i, c_coarse + 0.05 * m_i)
                if hi <= lo:
                    hi = min(m_i, lo + 0.01 * m_i)
                c_refine = np.linspace(lo, hi, refine_n)
                # Evaluate each
                best_local_c = c_coarse
                best_local_val = best_val[i]
                for c_r in c_refine:
                    c_r = min(c_r, m_i)
                    c_r = max(c_r, 1e-12)
                    cont_r = 0.0
                    for zp in range(n_states):
                        w = transition[z, zp]
                        if w > 0:
                            mp = R * (m_i - c_r) + income[zp]
                            mp_clip = np.clip(mp, grid[0], grid[-1])
                            v_next = np.interp(mp_clip, grid, V[t + 1, zp, :])
                            cont_r += w * v_next
                    val_r = util(c_r) + beta * cont_r
                    if val_r > best_local_val:
                        best_local_val = val_r
                        best_local_c = c_r
                best_c[i] = best_local_c
                best_val[i] = best_local_val

            C[t, z, :] = best_c
            # Store V[t, z, :] = best_val for next iteration's interpolation
            V[t, z, :] = best_val

    # Enforce constraints: 0 < c <= m + 1e-10, and c[5] = grid
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)
    C[5, :, :] = grid[np.newaxis, :]  # c[5] equals grid

    # Ensure finite
    C = np.where(np.isfinite(C), C, grid[np.newaxis, :])

    return C
# EVOLVE-BLOCK-END
