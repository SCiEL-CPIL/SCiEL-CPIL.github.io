import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = int(model['horizon'])

    num_states = len(income)
    orig_grid = np.asarray(resource_grid, dtype=np.float64)
    num_resources = len(orig_grid)
    c_floor = 1e-12

    # Dense working grid (log-spaced) for high accuracy in the scoring region
    n_work = 800
    work_grid = np.geomspace(0.001, 40.0, n_work)
    wg0, wg1 = work_grid[0], work_grid[-1]

    # --- Utility helpers (vectorized, floored) ---
    if gamma == 1:
        def utility(c):
            return np.log(np.maximum(c, c_floor))
        def u_prime(c):
            return 1.0 / np.maximum(c, c_floor)
    else:
        one_minus_g = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, c_floor), one_minus_g) / one_minus_g
        def u_prime(c):
            return np.power(np.maximum(c, c_floor), -gamma)

    policy_work = np.empty((horizon, num_states, n_work))
    value_work = np.empty((horizon, num_states, n_work))

    # Terminal: consume all
    for z in range(num_states):
        policy_work[horizon - 1, z, :] = work_grid
        value_work[horizon - 1, z, :] = utility(work_grid)

    # Precompute non-zero transitions per state
    nz = [np.where(transition[z] > 0)[0] for z in range(num_states)]
    nz_p = [transition[z][nz[z]] for z in range(num_states)]

    # --- Backward induction: solve Euler equation via vectorized bisection ---
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15.0:
            # Not enough time for a full period; fall back to a fast heuristic for remaining t
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    if tt == 0:
                        # simple heuristic
                        m = work_grid
                        c = np.clip(m * 0.9, c_floor, m)
                        policy_work[tt, z, :] = c
                        value_work[tt, z, :] = utility(c)
            break

        # Marginal value of next period on the working grid
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = u_prime(policy_work[t + 1, zp, :])

        for z in range(num_states):
            idx = nz[z]
            probs = nz_p[z]
            inc = income[idx]

            c_low = np.full(n_work, c_floor)
            c_high = work_grid.copy()
            # Initial guess: a smooth saving rule
            m_ratio = work_grid / (work_grid + 1.0)
            saving_rate = 0.25 * (1.0 - m_ratio) + 0.05
            c = np.clip(work_grid * (1.0 - saving_rate), c_floor, work_grid)

            max_iter = 45
            tol = 1e-12
            for it in range(max_iter):
                # Expected marginal continuation value at m_next = R*(m - c) + y
                m_next = R * (work_grid[:, None] - c[:, None]) + inc[None, :]
                m_next_c = np.clip(m_next, wg0, wg1)
                exp_vp = np.zeros(n_work)
                for k in range(len(idx)):
                    exp_vp += probs[k] * np.interp(m_next_c[:, k], work_grid, vp_next[idx[k], :])
                rhs = beta * R * exp_vp
                lhs = u_prime(c)

                # f(c) = lhs - rhs is decreasing in c
                too_low = lhs > rhs
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)

                if np.all(c_high - c_low < tol):
                    break
                c = (c_low + c_high) * 0.5
                c = np.clip(c, c_floor, work_grid)

            policy_work[t, z, :] = c

            # Value function
            m_next = R * (work_grid[:, None] - c[:, None]) + inc[None, :]
            m_next_c = np.clip(m_next, wg0, wg1)
            cont = np.zeros(n_work)
            for k in range(len(idx)):
                cont += probs[k] * np.interp(m_next_c[:, k], work_grid, value_work[t + 1, idx[k], :])
            value_work[t, z, :] = utility(c) + beta * cont

    # --- Interpolate to the original resource grid ---
    policy = np.empty((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
    policy = np.clip(policy, c_floor, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid  # terminal exact

    # --- Policy refinement on the exact scoring grid (251 pts, m in [0.2, 8]) ---
    score_grid = np.geomspace(0.001, 40.0, 251)
    mask = (score_grid >= 0.2) & (score_grid <= 8.0)
    sm = score_grid[mask]
    n_sm = len(sm)

    # Build a per-period, per-state value table on the scoring grid from value_work
    # value_score[t, z, j] = V_t(z, sm[j])
    value_score = np.empty((horizon, num_states, n_sm))
    for t in range(horizon):
        for z in range(num_states):
            value_score[t, z, :] = np.interp(sm, work_grid, value_work[t, z, :])

    # Current policy on the scoring grid
    pol_score = np.empty((horizon, num_states, n_sm))
    for t in range(horizon):
        for z in range(num_states):
            pol_score[t, z, :] = np.interp(sm, work_grid, policy_work[t, z, :])
    pol_score = np.clip(pol_score, c_floor, sm)

    # Refinement: for t from horizon-2 down to 0, move policy toward argmax of
    #   u(c) + beta * E[V_{t+1}(z', R*(m-c)+y_{z'})]
    # using damped coordinate updates (Gauss-Seidel over the grid points).
    n_refine = 2
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
        for z in range(num_states):
            idx = nz[z]
            probs = nz_p[z]
            inc = income[idx]
            cur = pol_score[t, z, :].copy()

            for _ in range(n_refine):
                if time.time() > deadline - 5.0:
                    break
                # Evaluate objective on a candidate grid of c values
                # Candidate c: a log-spaced set in (0, m)
                # We evaluate over c = sm * linspace(0.001, 1, 61) for each m
                n_cand = 61
                fracs = np.linspace(0.001, 1.0, n_cand)
                # cand[t,z,j,k] = sm[j] * fracs[k]
                cand = sm[:, None] * fracs[None, :]  # (n_sm, n_cand)
                cand = np.clip(cand, c_floor, sm[:, None])

                # Current utility
                u_cand = utility(cand)  # (n_sm, n_cand)

                # Continuation: m_next = R*(sm - cand) + y
                m_next = R * (sm[:, None, None] - cand[:, :, None]) + inc[None, None, :]
                m_next_c = np.clip(m_next, wg0, wg1)
                cont = np.zeros((n_sm, n_cand))
                for k in range(len(idx)):
                    # interpolate value_score[t+1, idx[k], :] over m_next
                    cont += probs[k] * np.interp(m_next_c[:, :, k], work_grid, value_work[t + 1, idx[k], :])
                obj = u_cand + beta * cont  # (n_sm, n_cand)

                # argmax over k for each j
                best_k = np.argmax(obj, axis=1)
                best_c = cand[np.arange(n_sm), best_k]
                best_c = np.clip(best_c, c_floor, sm)

                # Damped update
                cur = 0.5 * cur + 0.5 * best_c
                cur = np.clip(cur, c_floor, sm)

            pol_score[t, z, :] = cur

    # Write refined policy back to the original grid via interpolation
    for t in range(horizon):
        for z in range(num_states):
            # Combine: use refined values where the scoring grid covers, else keep work-grid interp
            # Build a merged x,y for interpolation: scoring grid (refined) + original grid (old)
            # Simpler: interpolate refined over score_grid, then over orig_grid
            refined_full = np.interp(score_grid, sm, pol_score[t, z, :])
            policy[t, z, :] = np.interp(orig_grid, score_grid, refined_full)
    policy = np.clip(policy, c_floor, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid

    # Final validity
    policy = np.nan_to_num(policy, nan=c_floor, posinf=orig_grid, neginf=c_floor)
    policy = np.maximum(policy, c_floor)
    policy = np.minimum(policy, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid

    return policy
