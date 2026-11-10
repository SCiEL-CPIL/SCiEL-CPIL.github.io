def solve_consumption(model, resource_grid, budget):
    import numpy as np
    from scipy.interpolate import CubicSpline

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    grid = np.asarray(resource_grid, dtype=float)
    N = len(grid)
    T = horizon

    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def uc(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        return np.power(c, -gamma)

    fit = np.geomspace(0.001, 40.0, 201)
    f0 = fit[0]
    ftop = fit[-1]
    eps = 1e-12
    cap = float(budget)

    def make_eval(vals):
        cs = CubicSpline(fit, vals, bc_type=(1, 0))
        def ev(m):
            m = np.asarray(m, dtype=float)
            out = cs(m)
            out = np.where(m <= f0, 0.0, out)
            out = np.where(m >= ftop, vals[-1], out)
            out = np.maximum(out, 0.0)
            return out
        return ev

    def solve_c(m, Vnext, z):
        m = np.asarray(m, dtype=float)
        out = np.empty_like(m)
        for i in range(2):
            mask = (z == i)
            if not np.any(mask):
                continue
            mm = m[mask]
            c = np.empty_like(mm)
            for k in range(mm.size):
                mk = mm[k]
                if mk <= eps:
                    c[k] = 0.0
                    continue
                lo = eps
                hi = min(mk, cap)
                if hi <= lo:
                    c[k] = hi if hi > eps else 0.0
                    continue
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    cont = R * (mk - mid) + income
                    Vn = Vnext(cont)
                    fmid = uc(mid) - beta * np.sum(trans[i] * R * uc(cont) * Vn)
                    if fmid > 0:
                        lo = mid
                    else:
                        hi = mid
                c[k] = 0.5 * (lo + hi)
                if c[k] > mk:
                    c[k] = mk
                if c[k] > cap:
                    c[k] = cap
            out[mask] = c
        out = np.clip(out, 0.0, None)
        return out

    def value_at(m, c, Vnext, z):
        cont = R * (m - c) + income
        Vn = Vnext(cont)
        return u(c) + beta * np.sum(trans[z] * Vn, axis=1)

    V = [None] * (T + 1)
    V[T] = (make_eval(np.zeros(N)), make_eval(np.zeros(N)))
    cons = np.zeros((T, 2, N))
    for t in range(T - 1, -1, -1):
        c = np.zeros((2, N))
        vals = np.zeros((2, N))
        for i in range(2):
            ci = solve_c(grid, V[t + 1][i], np.full(N, i))
            c[i] = ci
            vals[i] = value_at(grid, ci, V[t + 1][i], np.full(N, i))
        V[t] = (make_eval(vals[0]), make_eval(vals[1]))
        cons[t] = c

    cons = np.clip(cons, 0.0, None)
    for t in range(T):
        for i in range(2):
            cons[t, i] = np.minimum(cons[t, i], grid)
    cons[5] = np.tile(grid[None, :], (2, 1))

    return cons
