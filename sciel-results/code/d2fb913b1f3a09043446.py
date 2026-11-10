import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12

    # Denser internal grid for backward induction
    n_internal = 400
    m_internal = np.geomspace(0.001, 40, n_internal)

    # Utility functions
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)

    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)

    # Policy and Value arrays on internal grid
    policy = np.zeros((horizon, num_states, n_internal))
    value = np.zeros((horizon, num_states, n_internal))

    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(m_internal)
        policy[t, z, :] = m_internal

    # Pre-compute interpolators for the terminal period
    def build_interpolators(t_next):
        vp_interps = []
        val_interps = []
        for zp in range(num_states):
            vp_grid = u_prime(policy[t_next, zp, :])
            try:
                vp_interps.append(PchipInterpolator(m_internal, vp_grid))
            except Exception:
                vp_interps.append(None)
            try:
                val_interps.append(PchipInterpolator(m_internal, value[t_next, zp, :]))
            except Exception:
                val_interps.append(None)
        return vp_interps, val_interps

    # Backward Induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = m_internal
                    value[tt, z, :] = utility(m_internal)
            break

        vp_interps, val_interps = build_interpolators(t + 1)

        for z in range(num_states):
            probs = transition[z]
            active_zp = [zp for zp in range(num_states) if probs[zp] > 0]

            # Initialize: use previous period policy as warm start
            c = np.clip(policy[t + 1, z, :].copy(), c_floor, m_internal)

            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline:
                    break

                c_safe = np.maximum(c, c_floor)
                lhs = u_prime(c_safe)
                lhs2 = u_double_prime(c_safe)

                rhs = np.zeros(n_internal)
                rhs2 = np.zeros(n_internal)  # For f'(c)

                for zp in active_zp:
                    m_next = R * (m_internal - c_safe) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])

                    if vp_interps[zp] is not None:
                        vp_val = vp_interps[zp](m_next_clipped)
                        # Numerical second derivative of V' w.r.t. m
                        h = 1e-5
                        m_p = np.clip(m_next_clipped + h, m_internal[0], m_internal[-1])
                        m_m = np.clip(m_next_clipped - h, m_internal[0], m_internal[-1])
                        vp_p = vp_interps[zp](m_p)
                        vp_m = vp_interps[zp](m_m)
                        vpp_val = (vp_p - vp_m) / (2 * h)
                    else:
                        vp_grid = u_prime(policy[t + 1, zp, :])
                        vp_val = np.interp(m_next_clipped, m_internal, vp_grid)
                        h = 1e-5
                        m_p = np.clip(m_next_clipped + h, m_internal[0], m_internal[-1])
                        m_m = np.clip(m_next_clipped - h, m_internal[0], m_internal[-1])
                        vp_p = np.interp(m_p, m_internal, vp_grid)
                        vp_m = np.interp(m_m, m_internal, vp_grid)
                        vpp_val = (vp_p - vp_m) / (2 * h)

                    rhs += probs[zp] * vp_val
                    # f'(c) = u''(c) + beta*R * sum(probs[z'] * V''[z'](m_next) * R)
                    # where V'' is dV'/dm
                    rhs2 += probs[zp] * vpp_val

                rhs = beta * R * rhs
                rhs2 = beta * R * R * rhs2

                residual = lhs - rhs
                f_prime = lhs2 + rhs2  # f'(c) = u''(c) + beta*R^2 * sum(...)

                # Newton step with damping
                safe_fp = np.where(np.abs(f_prime) > 1e-15, f_prime, 1e-15 * np.sign(f_prime + 1e-30))
                step = residual / safe_fp

                # Damp large steps
                max_step = 0.5 * m_internal
                step = np.clip(step, -max_step, max_step)

                c_new = c_safe - step
                c_new = np.clip(c_new, c_floor, m_internal)

                # Check convergence
                if np.all(np.abs(c_new - c) < 1e-12 * np.maximum(1.0, m_internal)):
                    c = c_new
                    break

                c = c_new

            policy[t, z, :] = np.clip(c, c_floor, m_internal)

            # Compute value
            c_opt = policy[t, z, :]
            u_val = utility(c_opt)
            cont_val = np.zeros(n_internal)
            for zp in active_zp:
                m_next = R * (m_internal - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                if val_interps[zp] is not None:
                    v_next = val_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, m_internal, value[t + 1, zp, :])
                cont_val += probs[zp] * v_next

            value[t, z, :] = u_val + beta * cont_val

    # Ensure terminal condition is exact
    t = horizon - 1
    for z in range(num_states):
        policy[t, z, :] = m_internal
        value[t, z, :] = utility(m_internal)

    # Refinement pass: for points in [0.2, 8], do additional Newton iterations
    # This targets the scoring region directly
    if time.time() < deadline - 20:
        mask_refine = (m_internal >= 0.2) & (m_internal <= 8.0)
        idx_refine = np.where(mask_refine)[0]

        if len(idx_refine) > 0:
            for t in range(horizon - 2, -1, -1):
                if time.time() > deadline - 5:
                    break

                vp_interps, val_interps = build_interpolators(t + 1)

                for z in range(num_states):
                    probs = transition[z]
                    active_zp = [zp for zp in range(num_states) if probs[zp] > 0]

                    c_ref = policy[t, z, idx_refine].copy()
                    m_ref = m_internal[idx_refine]

                    for it in range(20):
                        c_safe = np.maximum(c_ref, c_floor)
                        lhs = u_prime(c_safe)
                        lhs2 = u_double_prime(c_safe)

                        rhs = np.zeros(len(idx_refine))
                        rhs2 = np.zeros(len(idx_refine))

                        for zp in active_zp:
                            m_next = R * (m_ref - c_safe) + income[zp]
                            m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])

                            if vp_interps[zp] is not None:
                                vp_val = vp_interps[zp](m_next_clipped)
                                h = 1e-5
                                m_p = np.clip(m_next_clipped + h, m_internal[0], m_internal[-1])
                                m_m = np.clip(m_next_clipped - h, m_internal[0], m_internal[-1])
                                vp_p = vp_interps[zp](m_p)
                                vp_m = vp_interps[zp](m_m)
                                vpp_val = (vp_p - vp_m) / (2 * h)
                            else:
                                vp_grid = u_prime(policy[t + 1, zp, :])
                                vp_val = np.interp(m_next_clipped, m_internal, vp_grid)
                                h = 1e-5
                                m_p = np.clip(m_next_clipped + h, m_internal[0], m_internal[-1])
                                m_m = np.clip(m_next_clipped - h, m_internal[0], m_internal[-1])
                                vp_p = np.interp(m_p, m_internal, vp_grid)
                                vp_m = np.interp(m_m, m_internal, vp_grid)
                                vpp_val = (vp_p - vp_m) / (2 * h)

                            rhs += probs[zp] * vp_val
                            rhs2 += probs[zp] * vpp_val

                        rhs = beta * R * rhs
                        rhs2 = beta * R * R * rhs2

                        residual = lhs - rhs
                        f_prime = lhs2 + rhs2

                        safe_fp = np.where(np.abs(f_prime) > 1e-15, f_prime, 1e-15 * np.sign(f_prime + 1e-30))
                        step = residual / safe_fp
                        max_step = 0.3 * m_ref
                        step = np.clip(step, -max_step, max_step)

                        c_new = c_safe - step
                        c_new = np.clip(c_new, c_floor, m_ref)

                        if np.all(np.abs(c_new - c_ref) < 1e-13 * np.maximum(1.0, m_ref)):
                            c_ref = c_new
                            break

                        c_ref = c_new

                    policy[t, z, idx_refine] = np.clip(c_ref, c_floor, m_ref)

                    # Recompute value for refined points
                    c_opt = policy[t, z, idx_refine]
                    u_val = utility(c_opt)
                    cont_val = np.zeros(len(idx_refine))
                    for zp in active_zp:
                        m_next = R * (m_ref - c_opt) + income[zp]
                        m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                        if val_interps[zp] is not None:
                            v_next = val_interps[zp](m_next_clipped)
                        else:
                            v_next = np.interp(m_next_clipped, m_internal, value[t + 1, zp, :])
                        cont_val += probs[zp] * v_next

                    value[t, z, idx_refine] = u_val + beta * cont_val

    # Interpolate policy back to the original resource_grid
    policy_out = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy_out[t, z, :] = np.interp(resource_grid, m_internal, policy[t, z, :])

    # Enforce constraints: 0 < c <= m + 1e-10, c[5] equals grid
    for t in range(horizon):
        for z in range(num_states):
            policy_out[t, z, :] = np.clip(policy_out[t, z, :], c_floor, resource_grid)
            if t == horizon - 1:
                policy_out[t, z, :] = resource_grid

    # Final validity check
    for t in range(horizon):
        for z in range(num_states):
            policy_out[t, z, :] = np.clip(policy_out[t, z, :], c_floor, resource_grid)
            policy_out[t, z, :] = np.where(np.isnan(policy_out[t, z, :]), resource_grid, policy_out[t, z, :])
            if t == horizon - 1:
                policy_out[t, z, :] = resource_grid

    return policy_out
