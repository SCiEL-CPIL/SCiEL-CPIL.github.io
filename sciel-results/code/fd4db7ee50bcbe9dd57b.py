"""Consumption-saving policy via Euler-FOC Newton with smooth log-space value representation."""
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 150.0 - 8.0  # safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    T = int(model['horizon'])
    Z = len(income)

    m = np.asarray(resource_grid, dtype=float)
    N = len(m)

    # ---- smooth log-space value representation: V_t(z, m) = sum_k a_k * exp(k * log m)
    # choose K so that exp(K*log m) spans the grid well
    K = 12

    # value coefficients per period/state: A[t][z] shape (K+1,)
    # V_t(z, m) = sum_{k=0..K} A[t][z][k] * m**k   (m**k = exp(k*log m))
    # This is a polynomial in m with K+1 coeffs -> smooth, easy derivatives.

    # ---- utility helpers (vectorized) ----
    def u(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u1(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        return c ** (-gamma)

    def u2(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / c**2
        return -gamma * c ** (-gamma - 1.0)

    # ---- value evaluation & derivatives at grid (vectorized over k) ----
    # m_pows[i, k] = m[i] ** k
    m_pows = np.empty((N, K + 1))
    m_pows[:, 0] = 1.0
    for k in range(1, K + 1):
        m_pows[:, k] = m_pows[:, k - 1] * m
    # derivatives: d/dm m^k = k * m^{k-1}
    m_pows_d1 = np.empty((N, K + 1))
    m_pows_d1[:, 0] = 0.0
    for k in range(1, K + 1):
        m_pows_d1[:, k] = k * m_pows[:, k - 1]
    m_pows_d2 = np.empty((N, K + 1))
    m_pows_d2[:, 0] = 0.0
    for k in range(2, K + 1):
        m_pows_d2[:, k] = k * (k - 1) * m_pows[:, k - 2]

    def evalV(Az, m_pows_):
        # Az shape (K+1,), m_pows_ shape (N, K+1) -> (N,)
        return m_pows_ @ Az

    # ---- solve one period/state: find c(i) on grid, then fit value coeffs ----
    def solve_period(Az_next, z):
        # Az_next[z'] for each next state
        # For each grid point i, maximize u(c) + beta * sum_{z'} trans[z,z'] * V_{t+1}(z', m')
        # where m' = R*(m_i - c) + income[z']
        # FOC: u1(c) = beta * sum_z' trans[z,z'] * V1_{t+1}(z', m')
        # Newton on c
        c = np.empty(N)
        # initial guess: proportional
        for i in range(N):
            mi = m[i]
            # start: consume a fraction
            c_i = 0.5 * mi
            c_i = min(max(c_i, 1e-12), mi)
            for it in range(40):
                if time.time() > DEADLINE:
                    break
                # compute m' for each z'
                c_safe = max(c_i, 1e-12)
                m_next = R * (mi - c_safe) + income  # shape (Z,)
                # V1 next for each z': sum_k A[k] * k * m'^{k-1}
                # build m'^{k-1} per z'
                # V1_zp = sum_k Az_next[zp][k] * k * m'^{k-1}
                V1 = np.zeros(Z)
                V1v = np.zeros(Z)
                for zp in range(Z):
                    Azp = Az_next[zp]
                    # m_next_zp = m_next[zp]
                    mzp = m_next[zp]
                    # compute powers
                    # V1 = sum_k Azp[k]*k*mzp^{k-1}
                    # V  = sum_k Azp[k]*mzp^{k}
                    p = 1.0
                    v = Azp[0]
                    d1 = 0.0
                    for k in range(1, K + 1):
                        p *= mzp
                        v += Azp[k] * p
                        d1 += Azp[k] * k * (p / mzp if mzp != 0 else 0.0)
                    V1v[zp] = v
                    V1[zp] = d1
                # FOC residual
                w = trans[z]  # shape (Z,)
                rhs = beta * np.dot(w, V1)
                f = u1(c_safe) - rhs
                # derivative of f w.r.t c: u2(c) - beta * sum w * V2 * dm'/dc
                # dm'/dc = -R
                # dV1/dm' = V2
                V2 = np.zeros(Z)
                for zp in range(Z):
                    Azp = Az_next[zp]
                    mzp = m_next[zp]
                    d2 = 0.0
                    p = 1.0
                    for k in range(2, K + 1):
                        p *= mzp
                        d2 += Azp[k] * k * (k - 1) * (p / (mzp * mzp) if mzp != 0 else 0.0)
                    V2[zp] = d2
                dfdc = u2(c_safe) - beta * np.dot(w, V2) * (-R)
                # Newton step
                step = f / dfdc if dfdc != 0 else 0.0
                c_new = c_i - step
                # project to (1e-12, mi)
                if c_new <= 1e-12:
                    c_new = 0.5 * (1e-12 + c_i)
                if c_new > mi:
                    c_new = 0.5 * (c_i + mi)
                if abs(c_new - c_i) < 1e-11 * (1.0 + abs(c_i)):
                    c_i = c_new
                    break
                c_i = c_new
            c[i] = c_i
        return c

    # ---- fit value coefficients A[t][z] from (m_i, V_i) where V_i = u(c_i) + beta*E[V_next] ----
    def fit_coeffs(V_vals):
        # V_vals shape (N,), fit V = sum_k a_k m^k  (least squares, K+1 coeffs)
        # solve normal equations: (M^T M) a = M^T V
        # M = m_pows (N x (K+1))
        # use lstsq for stability
        a, *_ = np.linalg.lstsq(m_pows, V_vals, rcond=None)
        return a

    # ---- backward induction ----
    # A[t][z]
    A = [None] * T
    # terminal: V_{T-1}(z, m) = u(m)  (consume all)
    # fit u(m) at grid for terminal
    V_term = u(m)
    A_term = fit_coeffs(V_term)
    A[T - 1] = np.tile(A_term, (Z, 1))
    c_policy = np.empty((T, Z, N))
    c_policy[T - 1] = np.tile(m, (Z, 1))

    for t in range(T - 2, -1, -1):
        if time.time() > DEADLINE:
            break
        A_t = np.empty((Z, K + 1))
        for z in range(Z):
            c_z = solve_period(A[t + 1], z)
            c_policy[t, z] = c_z
            # compute V at grid
            V_vals = np.empty(N)
            for i in range(N):
                mi = m[i]
                c_safe = max(c_z[i], 1e-12)
                base = u(c_safe)
                # expectation over next states
                m_next = R * (mi - c_safe) + income
                expV = 0.0
                for zp in range(Z):
                    Azp = A[t + 1][zp]
                    mzp = m_next[zp]
                    p = 1.0
                    v = Azp[0]
                    for k in range(1, K + 1):
                        p *= mzp
                        v += Azp[k] * p
                    expV += trans[z, zp] * v
                V_vals[i] = base + beta * expV
            A_t[z] = fit_coeffs(V_vals)
        A[t] = A_t

    # ---- final safety: enforce 0 < c <= m + 1e-10, finite; terminal = grid ----
    out = np.zeros((T, Z, N))
    for t in range(T):
        if t == T - 1:
            out[t] = np.tile(m, (Z, 1))
        else:
            for z in range(Z):
                cz = c_policy[t, z]
                cz = np.where(np.isfinite(cz), cz, 0.5 * m)
                cz = np.clip(cz, 1e-12, m + 1e-10)
                # ensure c <= m + 1e-10 and > 0
                out[t, z] = cz
    # ensure finite
    out = np.where(np.isfinite(out), out, 0.5 * m)
    return out
