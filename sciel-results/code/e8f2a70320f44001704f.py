"""Combined program: structural peak-seeking + smooth L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
_START = time.time()
_DEADLINE = _START + 110

def _time_left():
    return _DEADLINE - time.time()

def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    denom = norm_1 * norm_inf
    if denom <= 0:
        return 0.0
    return l2 / denom

def _value_soft(h, tau):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    ag = np.abs(g)
    amax = np.max(ag)
    if amax <= 0:
        return 0.0
    norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
    denom = norm_1 * norm_inf_soft
    if denom <= 0:
        return 0.0
    return l2 / denom

def _value_grad(h, tau=0.0):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)

    if tau > 0:
        ag = np.abs(g)
        amax = np.max(ag)
        if amax <= 0:
            return 0.0, np.zeros_like(h)
        norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
        wsoft = np.exp((ag - amax) / tau)
        wsoft /= np.sum(wsoft)
        dnh_dg = wsoft * np.sign(g)
        denom = norm_1 * norm_inf_soft
        val = l2 / denom
        norm_inf = norm_inf_soft
    else:
        norm_inf = np.max(np.abs(g))
        if norm_inf <= 0:
            return 0.0, np.zeros_like(h)
        imax = int(np.argmax(np.abs(g)))
        dnh_dg = np.zeros(M)
        dnh_dg[imax] = np.sign(g[imax])
        denom = norm_1 * norm_inf
        val = l2 / denom

    a = W / 3.0
    dly = np.zeros(M + 2)
    dly[:-1] += a * (2 * y[:-1] + y[1:])
    dly[1:]  += a * (y[:-1] + 2 * y[1:])
    dlg = dly[1:-1]

    conv1 = np.convolve(dlg, h, mode='full')
    conv2 = np.convolve(dlg[::-1], h, mode='full')
    dgh = conv1[:N] + conv2[N-1:]

    dn1_dg = np.sign(g) / (M + 1)
    dnh1 = np.convolve(dn1_dg, h, mode='full')[:N] + np.convolve(dn1_dg[::-1], h, mode='full')[N-1:]
    dnhinf = np.convolve(dnh_dg, h, mode='full')[:N] + np.convolve(dnh_dg[::-1], h, mode='full')[N-1:]

    ddenom = dnh1 * norm_inf + norm_1 * dnhinf
    grad_val = dgh / denom - l2 * ddenom / (denom * denom)
    return val, grad_val

def _project(h):
    h = np.clip(h, 0.0, None)
    s = np.sum(h)
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _grad_ascent(h0, max_iter=150):
    h = _project(h0.copy())
    best_h = h.copy()
    best_v = _value(h)
    lr = 0.1
    for it in range(max_iter):
        if _time_left() < 5:
            break
        tau = max(1e-3, 1.0 / (1.0 + it))
        val, gv = _value_soft(h, tau)
        step = lr
        improved = False
        for _ in range(12):
            h_new = _project(h + step * gv)
            vn = _value_soft(h_new, tau)
            if vn > val:
                h = h_new
                val = vn
                improved = True
                break
            step *= 0.5
        if improved:
            v_exact = _value(h)
            if v_exact > best_v:
                best_v = v_exact
                best_h = h.copy()
            lr = min(step * 1.1, 1.0)
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    v_exact = _value(best_h)
    if v_exact > best_v:
        best_v = v_exact
    return best_h, best_v

def _lbfgs_polish(h0, max_iter=300):
    h = _project(h0.copy())
    tau = 1e-3
    def neg_obj(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return 1e6
        x = x / s
        return -_value_soft(x, tau)
    def neg_grad(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return np.zeros(N)
        x = x / s
        _, gv = _value_soft(x, tau)
        return -gv
    try:
        res = minimize(neg_obj, h, method='L-BFGS-B', jac=neg_grad,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 20})
        h2 = _project(res.x)
        v2 = _value(h2)
        if v2 > _value(h):
            return h2, v2
    except Exception:
        pass
    return h, _value(h)

def _optimize():
    rng = np.random.default_rng(0)
    # Diverse inits: concentrated peaks, ramps, sines, blocks
    inits = [
        np.ones(N),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        rng.uniform(0, 1, N),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 3.0**2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 5.0**2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 8.0**2)),
        np.sin(np.pi * np.arange(N) / N) + 0.1,
        np.concatenate([np.ones(N // 2), np.zeros(N - N // 2)]),
        np.concatenate([np.zeros(N // 3), np.ones(N - 2 * (N // 3)), np.zeros(N // 3)]),
        # Very concentrated: only center few nonzero
        (lambda x: (x.__setitem__(slice(20, 30), 1.0), x)[1])(np.zeros(N)),
        (lambda x: (x.__setitem__(slice(23, 27), 1.0), x)[1])(np.zeros(N)),
        (lambda x: (x.__setitem__(slice(24, 26), 1.0), x)[1])(np.zeros(N)),
        # Ramp up sharply
        np.concatenate([np.zeros(N//4), np.linspace(0, 1, N - N//4)]),
        # Triangle peak
        np.abs(np.linspace(-1, 1, N)),
    ]
    # Fix the broken inits
    inits = []
    inits.append(np.ones(N))
    inits.append(np.linspace(0, 1, N))
    inits.append(np.linspace(1, 0, N))
    inits.append(rng.uniform(0, 1, N))
    inits.append(np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 3.0**2)))
    inits.append(np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 5.0**2)))
    inits.append(np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 8.0**2)))
    inits.append(np.sin(np.pi * np.arange(N) / N) + 0.1)
    inits.append(np.concatenate([np.ones(N // 2), np.zeros(N - N // 2)]))
    inits.append(np.concatenate([np.zeros(N // 3), np.ones(N - 2 * (N // 3)), np.zeros(N // 3)]))
    h_tmp = np.zeros(N); h_tmp[20:30] = 1.0; inits.append(h_tmp)
    h_tmp = np.zeros(N); h_tmp[23:27] = 1.0; inits.append(h_tmp)
    h_tmp = np.zeros(N); h_tmp[24:26] = 1.0; inits.append(h_tmp)
    inits.append(np.concatenate([np.zeros(N//4), np.linspace(0, 1, N - N//4)]))
    inits.append(np.abs(np.linspace(-1, 1, N)))

    best_h = None
    best_v = -1.0
    for h0 in inits:
        if _time_left() < 8:
            break
        h0 = np.clip(np.asarray(h0, float), 0, None)
        if not np.all(np.isfinite(h0)):
            continue
        s = np.sum(h0)
        if s <= 0:
            continue
        h0 = h0 / s
        h1, v1 = _grad_ascent(h0, max_iter=120)
        if v1 > best_v:
            best_v = v1
            best_h = h1.copy()
        if _time_left() < 10:
            break
        h2, v2 = _lbfgs_polish(h1, max_iter=200)
        if v2 > best_v:
            best_v = v2
            best_h = h2.copy()
    if best_h is not None and _time_left() > 5:
        h3, v3 = _lbfgs_polish(best_h, max_iter=400)
        if v3 > best_v:
            best_v = v3
            best_h = h3.copy()
    if best_h is None:
        best_h = np.ones(N)
        best_v = _value(best_h)
    return best_h, best_v

def construct():
    try:
        h, v = _optimize()
        h = np.asarray(h, float)
        h = np.clip(h, 0, None)
        if not np.all(np.isfinite(h)):
            h = np.ones(N)
        return list(h)
    except Exception:
        return [1.0] * N
