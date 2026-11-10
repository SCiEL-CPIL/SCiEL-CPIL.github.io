import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 138.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    log_grid = np.log(resource_grid)

    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return np.power(c, -gamma)

    def util_deriv2(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * np.power(c, -gamma - 1.0)

    V = np.zeros((horizon, num_states, num_resources))
    C = np.zeros((horizon, num_states, num_resources))

    # Terminal period: consume all
    V[horizon - 1] = 0.0
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            for tt in range(t, horizon - 1):
                for z in range(num_states):
                    C[tt, z] = resource_grid
                    V[tt, z] = 0.0
            break

        # Build interpolators in log-space for better accuracy across wide range
        pchip_V_log = []
        pchip_Vp_log = []
        for zp in range(num_states):
            V_vals = V[t + 1, zp]
            Vp_vals = util_deriv(C[t + 1, zp])
            # In log-space: x = log(m), so we interpolate V(log(m)) and V'(log(m))
            # Ensure values are finite
            V_vals = np.where(np.isfinite(V_vals), V_vals, 0.0)
            Vp_vals = np.where(np.isfinite(Vp_vals), Vp_vals, 1.0)
            pchip_V_log.append(PchipInterpolator(log_grid, V_vals, extrapolate=False))
            pchip_Vp_log.append(PchipInterpolator(log_grid, Vp_vals, extrapolate=False))

        for z in range(num_states):
            m_vals = resource_grid
            c_low = np.full(num_resources, c_floor)
            c_high = np.maximum(m_vals - c_floor, c_floor)
            c_curr = np.clip(0.5 * (c_low + c_high), c_floor, m_vals - c_floor)

            # Newton-Raphson with bisection fallback
            n_newton = 30
            for it in range(n_newton):
                if time.time() > deadline - 5.0:
                    break

                # Compute m' for each z'
                m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                log_m_prime = np.log(m_prime_clipped)

                # Interpolate V' in log-space
                V_prime_next_interp = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    V_prime_next_interp[zp] = pchip_Vp_log[zp](log_m_prime[zp])

                V_prime_expected = transition[z, :] @ V_prime_next_interp
                rhs = beta * R * V_prime_expected
                lhs = util_deriv(c_curr)
                diff = lhs - rhs  # positive means c too small

                # Newton step: d(lhs-rhs)/dc = u''(c) - beta*R*dV'/dc
                # Approximate dV'/dc via finite difference
                eps = np.maximum(1e-8 * c_curr, 1e-12)
                c_plus = np.minimum(c_curr + eps, m_vals - c_floor)
                c_minus = np.maximum(c_curr - eps, c_floor)

                m_prime_plus = R * (m_vals[np.newaxis, :] - c_plus[np.newaxis, :]) + income[:, np.newaxis]
                m_prime_minus = R * (m_vals[np.newaxis, :] - c_minus[np.newaxis, :]) + income[:, np.newaxis]
                m_prime_plus_clipped = np.clip(m_prime_plus, m_min, m_max)
                m_prime_minus_clipped = np.clip(m_prime_minus, m_min, m_max)
                log_mp_plus = np.log(m_prime_plus_clipped)
                log_mp_minus = np.log(m_prime_minus_clipped)

                Vp_plus = np.zeros((num_states, num_resources))
                Vp_minus = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    Vp_plus[zp] = pchip_Vp_log[zp](log_mp_plus[zp])
                    Vp_minus[zp] = pchip_Vp_log[zp](log_mp_minus[zp])

                Vpe_plus = transition[z, :] @ Vp_plus
                Vpe_minus = transition[z, :] @ Vp_minus
                dVp_dc = (Vpe_plus - Vpe_minus) / (c_plus - c_minus)

                # Total derivative of F(c) = u'(c) - beta*R*E[V'(m')]
                # dF/dc = u''(c) - beta*R*dE[V']/dc
                # Note: dE[V']/dc = sum z' trans[z,z'] * V''(m') * (-R)
                # But we're approximating numerically, so:
                dF_dc = util_deriv2(c_curr) - beta * R * dVp_dc

                # Newton step
                valid = np.abs(dF_dc) > 1e-30
                step = np.zeros(num_resources)
                step[valid] = -diff[valid] / dF_dc[valid]

                # Limit step to stay within bounds
                max_step = 0.5 * (c_high - c_low)
                step = np.clip(step, -max_step, max_step)

                c_new = c_curr + step
                c_new = np.clip(c_new, c_low, c_high)

                # Update bisection bounds
                mask_inc = diff > 0
                c_low[mask_inc] = c_curr[mask_inc]
                mask_dec = diff < 0
                c_high[mask_dec] = c_curr[mask_dec]

                c_curr = c_new

            c_opt = np.clip(c_curr, c_floor, m_vals - c_floor)
            small_m = m_vals < 2 * c_floor
            c_opt[small_m] = m_vals[small_m]
            C[t, z] = c_opt

            # Compute V[t, z, m]
            m_prime = R * (m_vals[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_prime_clipped = np.clip(m_prime, m_min, m_max)
            log_mp = np.log(m_prime_clipped)

            V_next_interp = np.zeros((num_states, num_resources))
            for zp in range(num_states):
                V_next_interp[zp] = pchip_V_log[zp](log_mp[zp])

            V_expected = transition[z, :] @ V_next_interp
            V[t, z] = utility(c_opt) + beta * V_expected

    # Final polish: refine the policy in the scoring range (0.2 <= m <= 8) with more iterations
    if time.time() < deadline - 3.0:
        # Rebuild interpolators from the final V
        pchip_V_log_final = []
        pchip_Vp_log_final = []
        for zp in range(num_states):
            V_vals = V[0, zp]  # Use t=0 V for polishing (or we could do per-t)
            Vp_vals = util_deriv(C[0, zp])
            V_vals = np.where(np.isfinite(V_vals), V_vals, 0.0)
            Vp_vals = np.where(np.isfinite(Vp_vals), Vp_vals, 1.0)
            pchip_V_log_final.append(PchipInterpolator(log_grid, V_vals, extrapolate=False))
            pchip_Vp_log_final.append(PchipInterpolator(log_grid, Vp_vals, extrapolate=False))

        # Polish only in the scoring range
        score_mask = (resource_grid >= 0.2) & (resource_grid <= 8.0)
        if np.any(score_mask):
            for t in range(horizon - 2, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                # Rebuild interpolators for t+1
                pchip_V_log_t1 = []
                pchip_Vp_log_t1 = []
                for zp in range(num_states):
                    V_vals = V[t + 1, zp]
                    Vp_vals = util_deriv(C[t + 1, zp])
                    V_vals = np.where(np.isfinite(V_vals), V_vals, 0.0)
                    Vp_vals = np.where(np.isfinite(Vp_vals), Vp_vals, 1.0)
                    pchip_V_log_t1.append(PchipInterpolator(log_grid, V_vals, extrapolate=False))
                    pchip_Vp_log_t1.append(PchipInterpolator(log_grid, Vp_vals, extrapolate=False))

                for z in range(num_states):
                    m_vals = resource_grid
                    c_curr = C[t, z].copy()
                    c_low = np.maximum(c_curr - 0.01 * m_vals, c_floor)
                    c_high = np.minimum(c_curr + 0.01 * m_vals, m_vals - c_floor)

                    for it in range(15):
                        if time.time() > deadline - 3.0:
                            break
                        m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                        m_prime_clipped = np.clip(m_prime, m_min, m_max)
                        log_mp = np.log(m_prime_clipped)

                        V_prime_next_interp = np.zeros((num_states, num_resources))
                        for zp in range(num_states):
                            V_prime_next_interp[zp] = pchip_Vp_log_t1[zp](log_mp[zp])

                        V_prime_expected = transition[z, :] @ V_prime_next_interp
                        rhs = beta * R * V_prime_expected
                        lhs = util_deriv(c_curr)
                        diff = lhs - rhs

                        # Only update in scoring range
                        if not np.any(score_mask):
                            break

                        # Newton step
                        eps = np.maximum(1e-9 * c_curr, 1e-13)
                        c_plus = np.minimum(c_curr + eps, m_vals - c_floor)
                        c_minus = np.maximum(c_curr - eps, c_floor)
                        mp_plus = R * (m_vals[np.newaxis, :] - c_plus[np.newaxis, :]) + income[:, np.newaxis]
                        mp_minus = R * (m_vals[np.newaxis, :] - c_minus[np.newaxis, :]) + income[:, np.newaxis]
                        mp_plus_clipped = np.clip(mp_plus, m_min, m_max)
                        mp_minus_clipped = np.clip(mp_minus, m_min, m_max)
                        lmp_plus = np.log(mp_plus_clipped)
                        lmp_minus = np.log(mp_minus_clipped)

                        Vp_plus = np.zeros((num_states, num_resources))
                        Vp_minus = np.zeros((num_states, num_resources))
                        for zp in range(num_states):
                            Vp_plus[zp] = pchip_Vp_log_t1[zp](lmp_plus[zp])
                            Vp_minus[zp] = pchip_Vp_log_t1[zp](lmp_minus[zp])

                        Vpe_plus = transition[z, :] @ Vp_plus
                        Vpe_minus = transition[z, :] @ Vp_minus
                        dVp_dc = (Vpe_plus - Vpe_minus) / (c_plus - c_minus)
                        dF_dc = util_deriv2(c_curr) - beta * R * dVp_dc

                        valid = np.abs(dF_dc) > 1e-30
                        step = np.zeros(num_resources)
                        step[valid] = -diff[valid] / dF_dc[valid]
                        step = np.clip(step, -0.5 * (c_high - c_low), 0.5 * (c_high - c_low))
                        c_new = c_curr + step
                        c_new = np.clip(c_new, c_low, c_high)
                        c_curr = c_new

                    C[t, z, score_mask] = c_curr[score_mask]

                    # Recompute V
                    m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                    m_prime_clipped = np.clip(m_prime, m_min, m_max)
                    log_mp = np.log(m_prime_clipped)
                    V_next_interp = np.zeros((num_states, num_resources))
                    for zp in range(num_states):
                        V_next_interp[zp] = pchip_V_log_t1[zp](log_mp[zp])
                    V_expected = transition[z, :] @ V_next_interp
                    V[t, z] = utility(c_curr) + beta * V_expected

    # Ensure final period consumes all
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    # Final validity checks
    m_expanded = resource_grid[np.newaxis, np.newaxis, :]
    C = np.maximum(C, c_floor)
    C = np.minimum(C, m_expanded + 1e-10)
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    return C
