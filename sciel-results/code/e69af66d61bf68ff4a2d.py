"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Precompute grid spacing for finite differences
    V_prime = np.zeros((horizon, n_states, n_grid))
    
    # Terminal derivative
    V_prime[horizon - 1, :, :] = marginal_utility(grid)

    # Backward induction using vectorized grid search over consumption fractions
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # Use a grid of consumption fractions for vectorized evaluation
            # Fractions from small to 1.0
            n_frac = min(200, budget)
            fracs = np.linspace(0.005, 1.0, n_frac)
            
            # C candidates: shape (n_frac, n_grid)
            C_cand = fracs[:, None] * grid[None, :]  # (n_frac, n_grid)
            C_cand = np.clip(C_cand, 1e-12, grid[None, :])
            
            # Compute m_prime for each (frac, m) pair
            # m_prime = R * (m - c) + income[z']
            # Shape: (n_frac, n_grid, n_states)
            m_c = grid[None, :] - C_cand  # (n_frac, n_grid)
            m_prime_all = R * m_c[:, :, None] + income[None, None, :]  # (n_frac, n_grid, n_states)
            
            # Clip to grid range
            m_prime_clipped = np.clip(m_prime_all, grid[0], grid[-1])
            
            # Compute continuation values using vectorized interpolation
            # For each z', we need to interpolate V[t+1, z', :] over m_prime
            cont_vals = np.zeros((n_frac, n_grid, n_states))
            for zp in range(n_states):
                # np.interp doesn't vectorize over the first two dims easily
                # Use searchsorted + linear interpolation
                idx = np.searchsorted(grid, m_prime_clipped[:, :, zp], side='right') - 1
                idx = np.clip(idx, 0, n_grid - 2)
                idx1 = idx + 1
                x0 = grid[idx]
                x1 = grid[idx1]
                y0 = V[t + 1, zp, idx]
                y1 = V[t + 1, zp, idx1]
                # Avoid division by zero
                denom = np.where(x1 - x0 > 0, x1 - x0, 1.0)
                cont_vals[:, :, zp] = y0 + (m_prime_clipped[:, :, zp] - x0) / denom * (y1 - y0)
            
            # Expected continuation value: (n_frac, n_grid)
            expected_cont = np.sum(transition[z][None, None, :] * cont_vals, axis=2)
            
            # Total value: u(c) + beta * expected_cont
            u_cand = utility(C_cand)  # (n_frac, n_grid)
            total_val = u_cand + beta * expected_cont  # (n_frac, n_grid)
            
            # Find optimal fraction for each m
            best_idx = np.argmax(total_val, axis=0)  # (n_grid,)
            best_frac = fracs[best_idx]  # (n_grid,)
            
            C[t, z, :] = best_frac * grid
            V[t, z, :] = total_val[best_idx, np.arange(n_grid)]
            
            # Compute V_prime for this period using central differences
            if n_grid > 2:
                V_prime[t, z, 1:-1] = (V[t, z, 2:] - V[t, z, :-2]) / (grid[2:] - grid[:-2])
            if n_grid > 1:
                V_prime[t, z, 0] = (V[t, z, 1] - V[t, z, 0]) / (grid[1] - grid[0])
                V_prime[t, z, -1] = (V[t, z, -1] - V[t, z, -2]) / (grid[-1] - grid[-2])

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
