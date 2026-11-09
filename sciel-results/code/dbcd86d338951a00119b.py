"""Solve consumption-saving policy via homothetic CRRA reduction + fine-grid backward induction."""
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

    # For CRRA, the optimal consumption fraction f = c/m is independent of m (homothetic).
    # We solve for f* at each (t, z) using the FOC in terms of the fraction.
    #
    # The Bellman equation:
    # V_t(m, z) = max_c { u(c) + beta * E[V_{t+1}(R*(m-c) + income[z'], z')] }
    #
    # With CRRA, V_t(m, z) = A_t(z) * m^(1-gamma)/(1-gamma) for gamma != 1
    # or V_t(m, z) = B_t(z) * log(m) + D_t(z) for gamma = 1
    #
    # The FOC gives: u'(c) = beta * R * E[V'_{t+1}(R*(m-c) + income[z'], z')]
    #
    # For the fraction approach, we can use the fact that at the optimum,
    # c = f* * m where f* depends only on (t, z) and the parameters.
    #
    # However, due to the additive income, the problem is not perfectly homothetic.
    # So we use a hybrid approach: solve for the optimal consumption on a fine grid
    # using the FOC, but vectorize over m for speed.

    # Strategy: Use a fine internal grid (e.g., 500 points) for backward induction,
    # then interpolate the resulting policy onto the required 101-point grid.
    # This gives much better accuracy than solving on the coarse 101-point grid.

    # Fine grid for computation
    n_fine = 500
    fine_grid = np.geomspace(grid[0], grid[-1], n_fine)

    # Value function and consumption policy on fine grid
    V_fine = np.zeros((horizon, n_states, n_fine))
    C_fine = np.zeros((horizon, n_states, n_fine))

    # Terminal: consume all
    V_fine[horizon - 1, :, :] = utility(fine_grid)
    C_fine[horizon - 1, :, :] = fine_grid

    # Derivative of value function
    V_prime_fine = np.zeros((horizon, n_states, n_fine))
    V_prime_fine[horizon - 1, :, :] = marginal_utility(fine_grid)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            # Fill remaining with heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C_fine[t2, z, :] = np.clip(fine_grid * frac, 1e-12, fine_grid)
                    V_fine[t2, z, :] = utility(C_fine[t2, z, :])
            break

        for z in range(n_states):
            # For each m on the fine grid, find optimal c
            # Vectorize the FOC evaluation
            m_vals = fine_grid
            c_candidates = m_vals[:, None] * np.linspace(0.001, 1.0, 301)[None, :]
            c_candidates = np.clip(c_candidates, 1e-12, m_vals[:, None])

            # For each m, compute the objective for each candidate c
            # Objective: u(c) + beta * E[V_{t+1}(R*(m-c) + income[z'], z')]
            # This is expensive for 500 * 301 = 150,000 evaluations per (t, z)
            # Let's use a smarter approach: solve the FOC directly

            # FOC: u'(c) = beta * R * sum_{z'} P(z'|z) * V'_{t+1}(R*(m-c) + income[z'], z')
            # We solve this for c given m.

            # For efficiency, use a coarse-to-fine search
            # First, find approximate solution on a coarse grid of c/m ratios
            # Then refine

            # Actually, let's just do a direct 1D optimization for each m
            # But that's 500 optimizations per (t, z) = 500 * 2 * 5 = 5000 total
            # Each optimization is fast with scipy, so this should be OK

            # Precompute the expected continuation value function as a function of m_prime
            # E[V_{t+1}(m_prime, z')] = sum_{z'} P(z'|z) * V_{t+1}(m_prime, z')
            # This is a weighted average of the two state value functions

            # Build the expected continuation value on the fine grid
            E_cont = np.zeros(n_fine)
            for zp in range(n_states):
                E_cont += transition[z, zp] * V_fine[t + 1, zp, :]

            E_cont_prime = np.zeros(n_fine)
            for zp in range(n_states):
                E_cont_prime += transition[z, zp] * V_prime_fine[t + 1, zp, :]

            # Now for each m, solve: u'(c) = beta * R * E_cont_prime(R*(m-c) + income[z'])
            # where E_cont_prime is interpolated at the appropriate point

            for i_m in range(n_fine):
                if time.time() > deadline - 5.0:
                    frac = 1.0 / (1.0 + beta * R)
                    c_opt = np.clip(fine_grid[i_m] * frac, 1e-12, fine_grid[i_m])
                    C_fine[t, z, i_m] = c_opt
                    m_prime_vals = R * (fine_grid[i_m] - c_opt) + income
                    m_prime_clipped = np.clip(m_prime_vals, fine_grid[0], fine_grid[-1])
                    exp_cont = np.sum(transition[z, :] * np.interp(m_prime_clipped, fine_grid, E_cont))
                    V_fine[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * exp_cont
                    continue

                m = fine_grid[i_m]

                def foc_residual(c):
                    c = max(c, 1e-12)
                    mu_c = marginal_utility(np.array([c]))[0]
                    m_prime_vals = R * (m - c) + income
                    m_prime_clipped = np.clip(m_prime_vals, fine_grid[0], fine_grid[-1])
                    exp_vp = np.sum(transition[z, :] * np.interp(m_prime_clipped, fine_grid, E_cont_prime))
                    return mu_c - beta * R * exp_vp

                # Bisection on [1e-12, m]
                lo = 1e-12
                hi = m * 0.999999

                res_lo = foc_residual(lo)
                res_hi = foc_residual(hi)

                if res_lo * res_hi > 0:
                    # No sign change; pick boundary
                    if res_lo < 0:
                        c_opt = lo
                    else:
                        c_opt = hi
                else:
                    # Bisection
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        res_mid = foc_residual(mid)
                        if res_lo * res_mid <= 0:
                            hi = mid
                        else:
                            lo = mid
                            res_lo = res_mid
                    c_opt = (lo + hi) / 2

                c_opt = np.clip(c_opt, 1e-12, m)
                C_fine[t, z, i_m] = c_opt

                # Compute value
                m_prime_vals = R * (m - c_opt) + income
                m_prime_clipped = np.clip(m_prime_vals, fine_grid[0], fine_grid[-1])
                exp_cont = np.sum(transition[z, :] * np.interp(m_prime_clipped, fine_grid, E_cont))
                V_fine[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * exp_cont

        # Compute V_prime for this period using central differences
        for z in range(n_states):
            if n_fine > 2:
                V_prime_fine[t, z, 1:-1] = (V_fine[t, z, 2:] - V_fine[t, z, :-2]) / (fine_grid[2:] - fine_grid[:-2])
            if n_fine > 1:
                V_prime_fine[t, z, 0] = (V_fine[t, z, 1] - V_fine[t, z, 0]) / (fine_grid[1] - fine_grid[0])
                V_prime_fine[t, z, -1] = (V_fine[t, z, -1] - V_fine[t, z, -2]) / (fine_grid[-1] - fine_grid[-2])

    # Interpolate the fine-grid policy onto the required 101-point grid
    C = np.zeros((horizon, n_states, n_grid))
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.interp(grid, fine_grid, C_fine[t, z, :])

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
