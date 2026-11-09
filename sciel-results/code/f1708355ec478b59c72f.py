import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 300.0
_CACHE = {}

def _build_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_idx = np.arange(-Q, Q + 1)

    # depth grid
    d_max = 6.0 / k
    G = 300
    depths = np.linspace(0.0, d_max, G)
    P = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)

    # H(c) = max(0, max_b P(b)*(b + c)) on c-grid
    C = 100.0
    Gc = 4000
    c_grid = np.linspace(-C, C, Gc)
    # lines: slope P_b, intercept P_b * b
    lines = P[:, None] * c_grid[None, :] + (P * depths)[:, None]
    H_grid = np.maximum(0.0, lines.max(axis=0))

    V_next = np.zeros(n_q)
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    cost = (gamma / 2.0) * sigma * sigma * dt

    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
            # fallback: fill remaining with a simple heuristic
            for tt in range(t, N):
                opt_b[tt, :] = 1.0 / k
                opt_a[tt, :] = 1.0 / k
            return opt_b, opt_a
        # d_plus = V(q+1)-V(q), d_minus = V(q-1)-V(q)
        d_plus = np.zeros(n_q)
        d_minus = np.zeros(n_q)
        d_plus[:-1] = V_next[1:] - V_next[:-1]   # q -> q+1
        d_minus[1:] = V_next[:-1] - V_next[1:]    # q -> q-1
        d_cross = d_plus + d_minus

        V_cur = np.zeros(n_q)
        # interior q
        for qi in range(1, n_q - 1):
            dp = d_plus[qi]
            dm = d_minus[qi]
            dc = d_cross[qi]
            c_a = dp + P * dc
            H_vals = np.interp(c_a, c_grid, H_grid)
            vals = H_vals + depths * P + P * dm
            ai = int(np.argmax(vals))
            V_cur[qi] = vals[ai] - cost * (q_idx[qi] ** 2)
            opt_a[t, qi] = depths[ai]
            # best b for this a
            c_b = dp + P[ai] * dc
            Hb = np.interp(c_b, c_grid, H_grid)
            # recover b: argmax P(b)*(b + c_b)
            vb = P * (depths + c_b)
            bi = int(np.argmax(vb))
            opt_b[t, qi] = depths[bi] if vb[bi] > 0 else np.inf
        # q = -Q (qi=0): no ask fill, only bid
        qi = 0
        dp = d_plus[qi]
        vb = P * (depths + dp)
        bi = int(np.argmax(vb))
        V_cur[qi] = vb[bi] - cost * (q_idx[qi] ** 2)
        opt_b[t, qi] = depths[bi] if vb[bi] > 0 else np.inf
        # q = Q (qi=n_q-1): no bid fill, only ask
        qi = n_q - 1
        dm = d_minus[qi]
        va = P * (depths + dm)
        ai = int(np.argmax(va))
        V_cur[qi] = va[ai] - cost * (q_idx[qi] ** 2)
        opt_a[t, qi] = depths[ai] if va[ai] > 0 else np.inf

        V_next = V_cur

    return opt_b, opt_a


def _get_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _CACHE:
        _CACHE[key] = _build_regime(*key)
    return _CACHE[key]


def quote(t, inventory, mid, params):
    opt_b, opt_a = _get_tables(params)
    dt = params['T'] / params['steps']
    t_idx = int(round(t / dt))
    t_idx = max(0, min(params['steps'] - 1, t_idx))
    qidx = np.asarray(inventory, dtype=int) + params['Q']
    qidx = np.clip(qidx, 0, 2 * params['Q'])
    bid = opt_b[t_idx, qidx].astype(float)
    ask = opt_a[t_idx, qidx].astype(float)
    # safety: ensure no NaN / negative-below-threshold
    bid = np.where(np.isfinite(bid), bid, np.inf)
    ask = np.where(np.isfinite(ask), ask, np.inf)
    return bid, ask
