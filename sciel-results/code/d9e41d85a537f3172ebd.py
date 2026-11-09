"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']

    m_grid = resource_grid
    M = len(m_grid)

    # Utility: exact CRRA as specified
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Value and policy arrays
    V = np.zeros((H, 2, M))
    C = np.zeros((H, 2, M))

    # Terminal period: t = H-1 = 5, c = m
    V[H - 1, :, :] = utility(m_grid)[np.newaxis, np.newaxis, :]
    C[H - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Helper to build spline for a given value vector
    def build_spline(v_vec):
        return CubicSpline(m_grid, v_vec, bc_type='natural')

    # Pre-build terminal splines
    splines = [None] * H
    splines[H - 1] = [build_spline(V[H - 1, zp, :]) for zp in range(2)]

    # Backward induction with vectorized grid search
    for t in range(H - 2, -1, -1):
        if time.time() > deadline:
            break

        # Build splines for next period's value functions
        splines[t] = [build_spline(V[t + 1, zp, :]) for zp in range(2)]

        for z in range(2):
            if time.time() > deadline:
                break

            p0 = transition[z, 0]
            p1 = transition[z, 1]
            y0 = income[0]
            y1 = income[1]

            # Use a dense grid of consumption fractions: c = frac * m
            # frac in [0.001, 1.0], use 200 points for good resolution
            n_frac = 200
            fracs = np.linspace(0.001, 1.0, n_frac)
            
            best_c = np.zeros(M)
            best_v = np.zeros(M)

            for i in range(M):
                m = m_grid[i]
                c_candidates = fracs * m
                c_candidates = np.clip(c_candidates, 1e-12, m)
                
                # Compute m' for each candidate
                m_prime_0 = R * (m - c_candidates) + y0
                m_prime_1 = R * (m - c_candidates) + y1
                
                # Evaluate continuation values
                v0 = splines[t + 1][0](m_prime_0)
                v1 = splines[t + 1][1](m_prime_1)
                e_v = p0 * v0 + p1 * v1
                
                # Total value
                u_c = utility(c_candidates)
                total_v = u_c + beta * e_v
                
                # Find best
                idx = np.argmax(total_v)
                best_c[i] = c_candidates[idx]
                best_v[i] = total_v[idx]

            C[t, z, :] = best_c
            V[t, z, :] = best_v

    # Post-loop validity check and fallback
    for t in range(H):
        for z in range(2):
            if np.all(C[t, z, :] == 0):
                C[t, z, :] = m_grid
            else:
                C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid)

    return C
# EVOLVE-BLOCK-END
