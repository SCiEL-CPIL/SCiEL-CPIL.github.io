"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Value function and consumption policy stored on the resource grid
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # Vectorized evaluation over all grid points
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 2.0:
                    frac = 1.0 / (1.0 + beta * R)
                    c_opt = np.clip(m * frac, 1e-12, m)
                    C[t, z, i_m] = c_opt
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c_opt) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont
                    continue

                # Define the objective to maximize: u(c) + beta * E[V_{t+1}(R*(m-c) + income[z'])]
                def neg_objective(c):
                    c = max(c, 1e-12)
                    if c >= m - 1e-12:
                        return 0.0  # penalty for infeasible
                    u_c = utility(np.array([c]))[0]
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    return -(u_c + beta * expected_cont)

                # Use bounded minimization over c in [1e-12, m]
                try:
                    result = minimize_scalar(neg_objective, bounds=(1e-12, m), method='bounded', 
                                            options={'xatol': 1e-8, 'maxiter': 100})
                    c_opt = result.x
                except:
                    c_opt = m / (1.0 + beta * R)
                
                # Ensure valid range
                c_opt = np.clip(c_opt, 1e-12, m)
                
                C[t, z, i_m] = c_opt
                
                # Compute the value
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
