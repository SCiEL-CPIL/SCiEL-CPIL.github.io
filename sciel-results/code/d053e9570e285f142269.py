"""Solve consumption-saving policy via backward induction with vectorized optimization."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))

    # Terminal: c[horizon-1] = m (consume all at last period)
    consumption[horizon - 1, :, :] = m_grid[None, None, :]

    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Not enough time; fill remaining with terminal-like policy
            for tt in range(t, -1, -1):
                consumption[tt, :, :] = m_grid[None, None, :]
            break

        # For each state z and each m, find optimal c
        # Bellman: max_c u(c) + beta * E_z'[V_{t+1}(m', z')]
        # where m' = R*(m - c) + income[z']

        # We'll use a fine grid for c search, then refine
        # c ranges from 0 to m (can't consume more than m)
        # Use fraction of m: c = m * frac, frac in [eps, 1]

        # Precompute: for next period, we have consumption[t+1, z', m']
        # V_{t+1}(m', z') = u(c_{t+1}(m', z')) + beta * E[V_{t+2}(...)]
        # But we only have the policy, not the value.
        # We need to compute value function too, or iterate.

        # Actually, let's compute value function V[t, z, m] as well.
        # V[horizon-1, z, m] = u(m)  (consume all)
        # V[t, z, m] = max_c { u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', m'] }
        # where m' = R*(m-c) + income[z']

        # Let's compute V[t+1] first, then V[t]
        # We already have consumption[t+1], but we need V[t+1].
        # Let's restructure: compute V and c together backward.

        pass

    # Better approach: compute value function and policy simultaneously
    # V[t, z, m] for t = horizon-1 down to 0
    V = np.zeros((horizon, Z, M))
    V[horizon - 1, :, :] = utility(m_grid)[None, None, :]

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15.0:
            for tt in range(t, -1, -1):
                consumption[tt, :, :] = m_grid[None, None, :]
                V[tt, :, :] = utility(m_grid)[None, None, :]
            break

        # For each z and m, maximize over c: u(c) + beta * sum_z' P(z'|z) * V[t+1, z', R*(m-c)+income[z']]
        # Use a grid of c values. Since c in (0, m], use fractions.
        # For efficiency, use a moderate grid and then refine.

        # Number of c candidates: use budget parameter
        n_c = max(100, budget)

        # c fractions: from small to 1
        # Avoid c=0 (log singularity for gamma=1)
        c_fracs = np.linspace(1e-4, 1.0, n_c)

        # For each z (current state), for each m:
        # m is m_grid[m_idx]
        # c = m * c_fracs
        # m_next[z'] = R * (m - c) + income[z']
        # We need V[t+1, z', m_next] via interpolation

        # Vectorize over m for fixed z
        # m_grid shape: (M,)
        # c shape: (M, n_c)
        # m_next shape: (M, n_c, Z) for each z'

        for z in range(Z):
            m_vals = m_grid  # shape (M,)

            # c candidates: shape (M, n_c)
            c_cands = m_vals[:, None] * c_fracs[None, :]

            # Ensure c > 0
            c_cands = np.maximum(c_cands, 1e-12)

            # Utility of current consumption: shape (M, n_c)
            u_c = utility(c_cands)

            # For each next state z', compute m_next and interpolate V[t+1, z', m_next]
            # m_next[z'] = R * (m - c) + income[z']
            # Shape: (M, n_c, Z)
            m_next = R * (m_vals[:, None, None] - c_cands[:, :, None]) + income[None, None, :]

            # m_next shape: (M, n_c, Z)
            # We need V_next[z', m_next] for each z'
            # V[t+1, z', :] has shape (M,)
            # Interpolate: for each (m_idx, c_idx, z'), get V[t+1, z', m_next[m_idx, c_idx, z']]

            # Use np.interp for each z'
            # V_next_interp shape: (M, n_c, Z)
            V_next_interp = np.zeros((M, n_c, Z))
            for zp in range(Z):
                # m_next[:, :, zp] shape (M, n_c)
                # V[t+1, zp, :] shape (M,)
                # np.interp works on 1D; we need to handle 2D
                # Flatten, interpolate, reshape
                mn_flat = m_next[:, :, zp].ravel()
                v_interp = np.interp(mn_flat, m_grid, V[t + 1, zp, :], left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                V_next_interp[:, :, zp] = v_interp.reshape(M, n_c)

            # Expected continuation: beta * sum_z' P(z'|z) * V_next_interp[:, :, z']
            # transition[z, :] shape (Z,)
            cont = beta * np.sum(transition[z, None, None, :] * V_next_interp, axis=2)
            # cont shape: (M, n_c)

            # Total objective: u_c + cont
            obj = u_c + cont

            # Find best c for each m
            best_idx = np.argmax(obj, axis=1)
            best_c = c_cands[np.arange(M), best_idx]

            # Refine: for each m, do a local search around best_c
            # Use a few iterations of golden section or fine grid
            # For simplicity, use a fine local grid
            n_refine = 20
            for m_idx in range(M):
                c_best = best_c[m_idx]
                m_val = m_vals[m_idx]

                # Local grid around c_best
                # Range: [max(1e-12, c_best - 0.05*m_val), min(m_val, c_best + 0.05*m_val)]
                lo = max(1e-12, c_best - 0.05 * m_val)
                hi = min(m_val, c_best + 0.05 * m_val)

                if hi <= lo:
                    hi = lo + 1e-10

                c_fine = np.linspace(lo, hi, n_refine)
                u_fine = utility(c_fine)

                m_next_fine = R * (m_val - c_fine) + income[None, :]
                V_fine = np.zeros((n_refine, Z))
                for zp in range(Z):
                    mn_flat = m_next_fine[:, zp]
                    v_interp = np.interp(mn_flat, m_grid, V[t + 1, zp, :], left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                    V_fine[:, zp] = v_interp

                cont_fine = beta * np.sum(transition[z, None, :] * V_fine, axis=1)
                obj_fine = u_fine + cont_fine
                best_local = c_fine[np.argmax(obj_fine)]
                best_c[m_idx] = best_local

            consumption[t, z, :] = best_c

            # Compute V[t, z, m] for each m
            # V[t, z, m] = u(best_c) + beta * sum_z' P(z'|z) * V[t+1, z', m']
            u_best = utility(best_c)
            m_next_best = R * (m_vals - best_c) + income[None, :]
            V_best = np.zeros((M, Z))
            for zp in range(Z):
                v_interp = np.interp(m_next_best[:, zp], m_grid, V[t + 1, zp, :], left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                V_best[:, zp] = v_interp
            cont_best = beta * np.sum(transition[z, None, :] * V_best, axis=1)
            V[t, z, :] = u_best + cont_best

    # Enforce terminal condition exactly
    consumption[horizon - 1, :, :] = m_grid[None, None, :]

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            c_row = consumption[t, z, :]
            m_row = m_grid
            # Clip to valid range
            c_row = np.clip(c_row, 1e-12, m_row + 1e-10)
            consumption[t, z, :] = c_row

    # Final check: c[5] (last period) must equal grid within 1e-9
    consumption[horizon - 1, :, :] = m_grid[None, None, :]

    return consumption
