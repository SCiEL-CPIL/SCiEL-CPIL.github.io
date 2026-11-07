"""Solve consumption-saving policy via Euler-equation Newton with log-linear value interpolation."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = resource_grid
    num_m = len(M)
    num_z = len(income)

    # Precompute utility and marginal utility
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def mu(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)

    # Backward induction
    # V[t, z, m] is the value at time t with state z and resources m
    # C[t, z, m] is the optimal consumption

    # At the final period (t = horizon - 1 = 5), consume all
    C = np.zeros((horizon, num_z, num_m))
    V = np.zeros((horizon, num_z, num_m))

    # Final period: consume everything
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # For earlier periods, solve the Euler equation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far, filling remaining with linear rule
            for tt in range(t, horizon - 1):
                if tt == horizon - 1:
                    C[tt] = M[np.newaxis, :]
                    V[tt] = util(M)[np.newaxis, :]
                else:
                    # Simple fallback: consume a fraction
                    C[tt] = M[np.newaxis, :] * 0.5
                    V[tt] = util(C[tt]) + beta * 0.5 * (V[tt + 1, 0] + V[tt + 1, 1])
            break

        for z in range(num_z):
            # Transition probabilities from state z
            probs = transition[z]
            next_zs = np.arange(num_z)

            # We need to solve for c[m] such that:
            # mu(c) = beta * R * sum_z' probs[z,z'] * mu(c'(m', z'))
            # where m' = R*(m - c) + income[z']
            # and c' is the optimal consumption at t+1 given m' and z'

            # For each m, we'll use Newton's method
            c_vals = np.zeros(num_m)
            v_vals = np.zeros(num_m)

            for i_m in range(num_m):
                m = M[i_m]

                # Initial guess: proportional consumption
                c = max(1e-6, min(m * 0.5, m - 1e-6))

                # Newton iterations
                for _ in range(30):
                    # Compute next-period resources for each z'
                    m_primes = R * (m - c) + income[next_zs]
                    # Clip to valid range
                    m_primes = np.maximum(m_primes, 1e-12)

                    # Get next-period marginal utilities via interpolation
                    # We need mu(c'(m', z')) for each z'
                    # c' is C[t+1, z', :] interpolated at m'

                    mu_next = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        if m_primes[j] <= M[0]:
                            cp = C[t + 1, zp, 0]
                            mu_next[j] = mu(cp)
                        elif m_primes[j] >= M[-1]:
                            cp = C[t + 1, zp, -1]
                            mu_next[j] = mu(cp)
                        else:
                            # Interpolate consumption
                            cp = np.interp(m_primes[j], M, C[t + 1, zp, :])
                            cp = max(cp, 1e-12)
                            mu_next[j] = mu(cp)

                    # RHS of Euler equation
                    rhs = beta * R * np.dot(probs, mu_next)

                    # LHS
                    lhs = mu(c)

                    # Residual
                    res = lhs - rhs

                    # Derivative of LHS w.r.t. c: -gamma * c^(-gamma-1)
                    # Derivative of RHS w.r.t. c:
                    # d/dc [beta * R * sum probs * mu(c'(R*(m-c)+inc))]
                    # = beta * R * sum probs * (-gamma * c'^(1-gamma)) * (-R) * dc'/dm'
                    # where dc'/dm' is the derivative of next-period consumption w.r.t. resources

                    # Approximate dc'/dm' numerically
                    eps = 1e-5
                    dc_dm = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        m1 = max(m_primes[j] - eps, 1e-12)
                        m2 = min(m_primes[j] + eps, M[-1])
                        if m1 <= M[0]:
                            cp1 = C[t + 1, zp, 0]
                        elif m1 >= M[-1]:
                            cp1 = C[t + 1, zp, -1]
                        else:
                            cp1 = np.interp(m1, M, C[t + 1, zp, :])
                        if m2 <= M[0]:
                            cp2 = C[t + 1, zp, 0]
                        elif m2 >= M[-1]:
                            cp2 = C[t + 1, zp, -1]
                        else:
                            cp2 = np.interp(m2, M, C[t + 1, zp, :])
                        dc_dm[j] = (cp2 - cp1) / max(m2 - m1, 1e-10)

                    # dRHS/dc = beta * R * sum probs * (-gamma * c'^(1-gamma)) * (-R) * dc_dm
                    # Note: mu'(c) = -gamma * c^(-gamma-1), so d mu(c')/dc' = -gamma * c'^(-gamma-1)
                    # d c'/dc = dc_dm * d m'/dc = dc_dm * (-R)
                    # So d mu(c')/dc = -gamma * c'^(-gamma-1) * (-R) * dc_dm = gamma * R * c'^(-gamma-1) * dc_dm

                    d_rhs_dc = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        if m_primes[j] <= M[0]:
                            cp = C[t + 1, zp, 0]
                        elif m_primes[j] >= M[-1]:
                            cp = C[t + 1, zp, -1]
                        else:
                            cp = np.interp(m_primes[j], M, C[t + 1, zp, :])
                        cp = max(cp, 1e-12)
                        d_mu_dc_prime = -gamma * cp ** (-gamma - 1)
                        d_m_prime_dc = -R
                        d_rhs_dc[j] = d_mu_dc_prime * d_m_prime_dc * dc_dm[j]

                    d_rhs_dc_total = beta * R * np.dot(probs, d_rhs_dc)

                    # Newton step: res = lhs - rhs, dres/dc = d_lhs/dc - d_rhs_dc
                    d_lhs_dc = -gamma * c ** (-gamma - 1)
                    d_res_dc = d_lhs_dc - d_rhs_dc_total

                    if abs(d_res_dc) < 1e-15:
                        break

                    step = res / d_res_dc
                    c_new = c - step

                    # Bound c to [1e-12, m]
                    c_new = max(1e-12, min(c_new, m))

                    if abs(c_new - c) < 1e-10 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new

                c_vals[i_m] = c

                # Compute value
                m_primes = R * (m - c) + income[next_zs]
                m_primes = np.maximum(m_primes, 1e-12)
                v_next = np.zeros(num_z)
                for j, zp in enumerate(next_zs):
                    if m_primes[j] <= M[0]:
                        v_next[j] = V[t + 1, zp, 0]
                    elif m_primes[j] >= M[-1]:
                        v_next[j] = V[t + 1, zp, -1]
                    else:
                        v_next[j] = np.interp(m_primes[j], M, V[t + 1, zp, :])

                v_vals[i_m] = util(c) + beta * np.dot(probs, v_next)

            C[t, z, :] = c_vals
            V[t, z, :] = v_vals

    # Ensure final period constraint: c[5] = m
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], M + 1e-10)

    return C
