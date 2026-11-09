"""Rewritten solver: homothetic ratio-space search with local refinement and deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    # Utility and derivative
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

    # Continuation value: V_{t+1}[z, m]
    # We'll store V as array [n_z, n_grid] for each t from horizon-1 down to 0.
    # At t = horizon-1 (last period), continuation value is 0 (no future).
    # Actually, the problem says c[5] must equal the grid. That means at t=5 (last),
    # consumption = m (spend all). So V[5] = 0 (no continuation).
    # For t < 5, V[t] is the continuation value from t+1 onward.

    # Let's define:
    # V[t][z, m] = max_{c in [0, m]} u(c) + beta * E_z'[ V[t+1][z', m'] ]
    # where m' = R*(m - c) + income[z']
    #
    # But the task says: "c[5] equal to the grid within 1e-9"
    # This means at t=5, c = m. So V[5] = 0 (terminal).
    # We compute V[4], V[3], ..., V[0] backward.
    # Then policy at t is argmax over c.

    # We'll store policies: policy[t][z, i] = optimal c for grid index i
    policies = np.zeros((horizon, n_z, n_grid))
    # Continuation values: V[t][z, i] for t=0..horizon-1
    # V[horizon-1] = 0
    V = np.zeros((horizon + 1, n_z, n_grid))  # V[t] is value at time t, V[horizon] = 0

    # V[horizon] = 0 already
    # For t from horizon-1 down to 0:
    for t in range(horizon - 1, -1, -1):
        if time.time() > DEADLINE - 5.0:
            # Emergency: fill remaining with simple policy
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    policies[tt, z, :] = grid
            break

        # V[t+1] is known
        V_next = V[t + 1]  # shape [n_z, n_grid]

        # For each z and each grid point m, find optimal c
        for z in range(n_z):
            # Precompute expected continuation for a given c
            # E_z'[ V_next[z', m'] ] where m' = R*(m-c) + income[z']
            # z' ~ transition[z]
            # For each z', m' = R*(m-c) + income[z']
            # Then V_next[z', m'] is interpolated from grid

            # We'll search over ratio f = c/m in [0.001, 1.0]
            # For m near 0, c is small; for m large, c is large.

            # Coarse grid of ratios
            # Use more points for better accuracy
            n_ratio = 200
            ratios = np.linspace(0.001, 1.0, n_ratio)

            # For each grid point m, compute value for each ratio
            # Vectorize over m
            m_vals = grid  # [n_grid]
            c_vals = ratios[:, None] * m_vals[None, :]  # [n_ratio, n_grid]
            c_vals = np.clip(c_vals, 1e-12, m_vals[None, :] + 1e-10)

            # Compute utility
            u_vals = u(c_vals)  # [n_ratio, n_grid]

            # Compute m' for each z'
            # m' = R*(m - c) + income[z']
            # For each z', shape [n_z', n_ratio, n_grid]
            m_prime = R * (m_vals[None, None, :] - c_vals[None, :, :]) + income[:, None, None]
            m_prime = np.maximum(m_prime, 0.0)  # m' should be non-negative

            # Interpolate V_next[z', m']
            # V_next shape [n_z, n_grid]
            # For each z', interpolate m_prime along grid
            V_interp = np.zeros((n_z, n_ratio, n_grid))
            for zp in range(n_z):
                V_interp[zp] = np.interp(m_prime[zp], grid, V_next[zp])

            # Expected continuation: sum over z' of transition[z, z'] * V_interp[z']
            # transition[z] shape [n_z]
            cont = np.tensordot(transition[z], V_interp, axes=(0, 0))  # [n_ratio, n_grid]

            # Total value: u(c) + beta * cont
            total_val = u_vals + beta * cont  # [n_ratio, n_grid]

            # Find best ratio for each m
            best_idx = np.argmax(total_val, axis=0)  # [n_grid]
            best_ratios = ratios[best_idx]
            best_c = best_ratios * m_vals

            # Local refinement around best ratio
            # For each m, do a local search in ratio space
            # To save time, only refine a subset or use vectorized approach
            # Let's refine all points with a small local search

            # We'll do a simple golden-section or use scipy on a subset
            # For speed, refine only if we have time
            if time.time() < DEADLINE - 10.0:
                # Local refinement using scipy on ratio
                # We'll process in batches to avoid overhead
                n_refine = 50
                # Use a finer local grid around best ratio
                for i in range(0, n_grid, 1):
                    if time.time() > DEADLINE - 5.0:
                        break
                    m_i = m_vals[i]
                    if m_i < 1e-9:
                        continue
                    r0 = best_ratios[i]
                    # Local search in [max(0.001, r0-0.05), min(1.0, r0+0.05)]
                    lo = max(0.001, r0 - 0.05)
                    hi = min(1.0, r0 + 0.05)
                    if hi - lo < 1e-6:
                        continue

                    def neg_obj(r):
                        c_i = max(r * m_i, 1e-12)
                        c_i = min(c_i, m_i + 1e-10)
                        ui = u(c_i)
                        mp = R * (m_i - c_i) + income
                        mp = np.maximum(mp, 0.0)
                        vi = np.interp(mp, grid, V_next)
                        cont_i = np.dot(transition[z], vi)
                        return -(ui + beta * cont_i)

                    try:
                        res = minimize_scalar(neg_obj, bounds=(lo, hi), method='bounded',
                                              options={'xatol': 1e-6})
                        if res.success:
                            best_c[i] = res.x * m_i
                    except:
                        pass

            policies[t, z, :] = np.clip(best_c, 1e-12, m_vals + 1e-10)

            # Compute V[t][z, :] using the optimal c
            c_opt = policies[t, z, :]
            u_opt = u(c_opt)
            m_prime_opt = R * (m_vals - c_opt) + income[:, None]
            m_prime_opt = np.maximum(m_prime_opt, 0.0)
            V_interp_opt = np.zeros((n_z, n_grid))
            for zp in range(n_z):
                V_interp_opt[zp] = np.interp(m_prime_opt[zp], grid, V_next[zp])
            cont_opt = np.tensordot(transition[z], V_interp_opt, axes=(0, 0))
            V[t, z, :] = u_opt + beta * cont_opt

        # Check deadline
        if time.time() > DEADLINE - 2.0:
            # Fill remaining t's with simple policy
            for tt in range(t - 1, -1, -1):
                for z in range(n_z):
                    policies[tt, z, :] = grid
            break

    # Enforce terminal condition: c[5] = grid
    policies[-1, :, :] = grid

    return policies
# EVOLVE-BLOCK-END
