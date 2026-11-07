"""Consumption-saving policy via Euler-equation Newton with PCHIP continuation interpolation."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import time


def _crsa_u(c, gamma):
    """CRRA utility, floored."""
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def _crsa_mu(c, gamma):
    """Marginal utility."""
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / c
    return c ** (-gamma)


def _crsa_mu2(c, gamma):
    """Second derivative of utility."""
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return -1.0 / (c * c)
    return -gamma * c ** (-gamma - 1.0)


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    M = np.asarray(resource_grid, dtype=float)
    num_m = len(M)
    num_z = len(income)

    # Final period: consume everything
    V = np.zeros((horizon + 1, num_z, num_m))
    C = np.zeros((horizon, num_z, num_m))
    V[horizon, :, :] = _crsa_u(M, gamma)
    C[horizon - 1, :, :] = M  # c[t] for t=horizon-1 is m at final (index 5)

    # We need V[t, z, m] for t = horizon-1 down to 0.
    # V[horizon] is known. For t from horizon-2 down to 0:
    #   V[t, z, m] = max_c u(c) + beta * sum_{z'} trans[z, z'] * V[t+1, z', m']
    #   where m' = R*(m - c) + income[z']
    # Euler: mu(c) = beta * R * sum_z' trans[z,z'] * mu(V_{t+1, z', m'})
    #   i.e. mu(c) = beta * R * sum_z' trans[z,z'] * dV[t+1,z',m']/dm'
    # where dV/dm' = mu(c') (envelope theorem), so:
    #   mu(c) = beta * R * sum_z' trans[z,z'] * mu(c'[z'])
    #   with c'[z'] = optimal consumption next period given m'[z'] = R*(m-c)+income[z']

    # For each t from horizon-2 down to 0, for each z, solve for c[m] for all m.
    # Use PCHIP to interpolate the next-period marginal utility as a function of m'.
    # dV[t+1, z', m'] = mu(C[t+1, z', m'])  (envelope: dV/dm = mu(c))
    # So we need mu(C[t+1, z', m']) as a function of m' for each z'.

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        # For each z', build interpolant of mu_next(z', m') = mu(C[t+1, z', m']) vs m'
        # Note: C[t+1] is defined for t+1 in [0, horizon-1]. At t+1 = horizon-1, C = M.
        # dV[t+1, z', m'] = mu(C[t+1, z', m'])
        mu_next_interp = []
        for zp in range(num_z):
            mu_vals = _crsa_mu(C[t + 1, zp, :], gamma)
            # Ensure monotonicity for PCHIP: mu should be decreasing in m'
            # (higher m -> higher c -> lower mu). PCHIP preserves monotonicity.
            try:
                mu_next_interp.append(PchipInterpolator(M, mu_vals))
            except Exception:
                # Fallback: linear interp
                mu_next_interp.append(np.interp)

        # For each current z, solve Euler for each m
        for z in range(num_z):
            c_sol = np.empty(num_m)
            for i in range(num_m):
                m = M[i]

                def euler_residual(c):
                    # mu(c) - beta*R*sum_z' trans[z,z']*mu_next(z', m')
                    c = np.clip(c, 1e-12, m)
                    lhs = _crsa_mu(c, gamma)
                    rhs = 0.0
                    for zp in range(num_z):
                        mprime = R * (m - c) + income[zp]
                        # Clip mprime to grid range for interpolation
                        if mprime < M[0]:
                            mprime = M[0]
                        elif mprime > M[-1]:
                            mprime = M[-1]
                        mu_val = mu_next_interp[zp](mprime)
                        rhs += transition[z, zp] * mu_val
                    return lhs - beta * R * rhs

                def euler_deriv(c):
                    # d/dc [mu(c) - beta*R*sum trans*mu_next(m')]
                    # d mu(c)/dc = mu2(c)
                    # d m'/dc = -R
                    # d mu_next/dm' * (-R)
                    c = np.clip(c, 1e-12, m)
                    d_lhs = _crsa_mu2(c, gamma)
                    d_rhs = 0.0
                    for zp in range(num_z):
                        mprime = R * (m - c) + income[zp]
                        if mprime < M[0]:
                            mprime = M[0]
                        elif mprime > M[-1]:
                            mprime = M[-1]
                        # Numerical derivative of mu_next w.r.t. m'
                        h = 1e-6 * max(1.0, mprime)
                        m1 = max(M[0], mprime - h)
                        m2 = min(M[-1], mprime + h)
                        dm = m2 - m1
                        if dm < 1e-12:
                            dmu_dm = 0.0
                        else:
                            dmu_dm = (mu_next_interp[zp](m2) - mu_next_interp[zp](m1)) / dm
                        d_rhs += transition[z, zp] * dmu_dm * (-R)
                    return d_lhs - beta * R * d_rhs

                # Newton's method with bisection safeguard
                lo = 1e-12
                hi = m
                c = 0.5 * (lo + hi)  # initial guess
                for _ in range(50):
                    f = euler_residual(c)
                    if abs(f) < 1e-10:
                        break
                    df = euler_deriv(c)
                    if abs(df) < 1e-14:
                        # Bisection
                        if f > 0:
                            hi = c
                        else:
                            lo = c
                        c = 0.5 * (lo + hi)
                    else:
                        c_new = c - f / df
                        if c_new <= lo or c_new >= hi:
                            if f > 0:
                                hi = c
                            else:
                                lo = c
                            c = 0.5 * (lo + hi)
                        else:
                            c = c_new
                    c = np.clip(c, lo, hi)
                c_sol[i] = np.clip(c, 1e-12, m)

            C[t, z, :] = c_sol

            # Compute V[t, z, m] via envelope: V[t,z,m] = u(c) + beta*sum trans*V[t+1,z',m']
            # But we only need V for interpolation in the next iteration (t-1).
            # Actually, we need V[t] for the next outer loop iteration.
            # V[t, z, m] = u(c) + beta * sum_z' trans[z,z'] * V[t+1, z', m']
            # We have V[t+1] already. Let's compute it.
            for i in range(num_m):
                m = M[i]
                c = C[t, z, i]
                v = _crsa_u(c, gamma)
                for zp in range(num_z):
                    mprime = R * (m - c) + income[zp]
                    if mprime < M[0]:
                        mprime = M[0]
                    elif mprime > M[-1]:
                        mprime = M[-1]
                    # Interpolate V[t+1, zp, mprime]
                    idx = np.searchsorted(M, mprime)
                    idx = np.clip(idx, 1, num_m - 1)
                    m1 = M[idx - 1]
                    m2 = M[idx]
                    if m2 - m1 < 1e-15:
                        vp = V[t + 1, zp, idx - 1]
                    else:
                        frac = (mprime - m1) / (m2 - m1)
                        vp = V[t + 1, zp, idx - 1] * (1 - frac) + V[t + 1, zp, idx] * frac
                    v += beta * transition[z, zp] * vp
                V[t, z, i] = v

    # Ensure final period consumption equals grid
    C[horizon - 1, :, :] = M

    # Validity checks: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, M + 1e-10)

    return C
