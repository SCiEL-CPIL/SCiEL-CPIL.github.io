import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]


def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)


def _conv_max(h):
    return np.max(np.convolve(h, h, mode='full'))


def _project_simplex(v, z=1.0):
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0][-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

    def now_ok(margin=0.0):
        return time.time() < deadline - margin

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    x_idx = np.arange(N)
    np.random.seed(42)

    # --- Phase 1: diverse initializations ---
    inits = []
    inits.append(np.ones(N) / N)

    for trial in range(30):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    for c in np.linspace(3, N - 3, 8):
        for w in np.linspace(0.5, N / 3, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)

    for c in np.linspace(3, N - 3, 6):
        for w in np.linspace(2, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    for _ in range(15):
        h_r = np.random.rand(N)
        h_r = np.convolve(h_r, np.ones(5) / 5, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    for pos in range(N):
        h_spike = np.zeros(N)
        h_spike[pos] = 1.0
        inits.append(h_spike)

    for pos1 in range(0, N, 4):
        for pos2 in range(pos1 + 2, N, 4):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= np.sum(h_two)
            inits.append(h_two)

    scored_inits = []
    for h_init in inits:
        scored_inits.append((_score_raw(h_init), h_init))
    scored_inits.sort(key=lambda x: -x[0])

    # --- Phase 2: epigraph (SLSQP) minimizing max autoconvolution ---
    num_conv = 2 * N - 1

    def obj(x):
        return x[N]

    def obj_grad(x):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g

    def eq_fun(x):
        return np.sum(x[:N]) - 1.0

    def eq_jac(x):
        j = np.zeros(N + 1)
        j[:N] = 1.0
        return j

    def ineq_fun(x):
        h = x[:N]
        M = x[N]
        return M - np.convolve(h, h, mode='full')

    def ineq_jac(x):
        h = x[:N]
        j = np.zeros((num_conv, N + 1))
        j[:, N] = 1.0
        for jj in range(N):
            ks = jj
            ke = min(jj + N, num_conv)
            if ks < num_conv and ke > 0:
                ik = np.arange(ks, ke)
                ih = ik - jj
                j[ik, jj] = -2.0 * h[ih]
        return j

    constraints = [
        {'type': 'eq', 'fun': eq_fun, 'jac': eq_jac},
        {'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_jac},
    ]
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]

    def run_epigraph(h_init, maxiter=250, ftol=1e-12):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        M_init = _conv_max(h_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints, jac=obj_grad,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success:
                hc = np.clip(res.x[:N], 0, None)
                if np.sum(hc) > 1e-15:
                    return hc / np.sum(hc)
        except Exception:
            pass
        return None

    for s_init, h_init in scored_inits[:25]:
        if not now_ok(30.0):
            break
        hc = run_epigraph(h_init, maxiter=250, ftol=1e-12)
        if hc is not None:
            sc = _score_raw(hc)
            if sc > best_score:
                best_score = sc
                best_h = hc.copy()

    # --- Phase 3: smooth log-sum-exp max-constraint (L-BFGS-B) ---
    def run_lse(h_init, iters=60, beta=12.0):
        h = h_init / np.sum(h_init)
        for _ in range(iters):
            if not now_ok(20.0):
                break
            conv = np.convolve(h, h, mode='full')
            m = np.max(conv)
            logsum = m + np.log(np.sum(np.exp(conv - m)))
            lse = np.exp(logsum)
            # d/dh_i logsum = sum_k (conv_k/lse) * dconv_k/dh_i ; dconv_k/dh_i = 2 h_{k-i}
            contrib = np.zeros(N)
            for k in range(num_conv):
                w = np.exp(conv[k] - logsum)
                if w < 1e-15:
                    continue
                lo = max(0, k - (N - 1))
                hi = min(N - 1, k)
                idx = np.arange(lo, hi + 1)
                contrib[idx] += w * 2.0 * h[k - idx]
            grad = -beta * contrib
            # projected gradient ascent on l2_sq with simplex constraint
            g = 2.0 * h + grad
            h = _project_simplex(g, 1.0)
            if np.sum(h) < 1e-15:
                break
            h /= np.sum(h)
        return h

    for s_init, h_init in scored_inits[:12]:
        if not now_ok(25.0):
            break
        hc = run_lse(h_init, iters=60, beta=12.0)
        sc = _score_raw(hc)
        if sc > best_score:
            best_score = sc
            best_h = hc.copy()

    # --- Phase 4: block-coordinate descent on symmetric pairs ---
    def block_coordinate_refine(h_start, max_iter=30):
        h = np.clip(h_start, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        for _ in range(max_iter):
            if not now_ok(8.0):
                break
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            improved = False
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                hi = h[i]
                hj = h[j]
                t_min = -hi
                t_max = hj
                if t_max - t_min < 1e-15:
                    continue
                best_t = 0.0
                best_local = _score_raw(h)
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 10):
                    ht = h.copy()
                    ht[i] += t
                    ht[j] -= t
                    if np.any(ht < -1e-15):
                        continue
                    ht = np.clip(ht, 0, None)
                    if np.sum(ht) < 1e-15:
                        continue
                    ht /= np.sum(ht)
                    st = _score_raw(ht)
                    if st > best_local:
                        best_local = st
                        best_t = t
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) < 1e-15:
                        return h
                    h /= np.sum(h)
                    improved = True
            if not improved:
                break
        return h

    if now_ok(15.0):
        hr = block_coordinate_refine(best_h, max_iter=30)
        sr = _score_raw(hr)
        if sr > best_score:
            best_score = sr
            best_h = hr.copy()

    # --- Phase 5: local random perturbations ---
    if now_ok(10.0):
        for _ in range(25):
            if not now_ok(5.0):
                break
            hp = best_h.copy()
            mask = np.random.rand(N) < 0.3
            hp[mask] *= (1 + 0.2 * np.random.randn(int(np.sum(mask))))
            hp = np.clip(hp, 0, None)
            if np.sum(hp) < 1e-15:
                continue
            hp /= np.sum(hp)
            sp = _score_raw(hp)
            if sp > best_score:
                best_score = sp
                best_h = hp.copy()
                if now_ok(8.0):
                    hr = block_coordinate_refine(best_h, max_iter=20)
                    sr = _score_raw(hr)
                    if sr > best_score:
                        best_score = sr
                        best_h = hr.copy()

    # --- Phase 6: final high-precision polish ---
    if now_ok(5.0):
        hp = run_epigraph(best_h, maxiter=400, ftol=1e-14)
        if hp is not None:
            sp = _score_raw(hp)
            if sp > best_score:
                best_score = sp
                best_h = hp.copy()
        if now_ok(2.0):
            hf = block_coordinate_refine(best_h, max_iter=20)
            sf = _score_raw(hf)
            if sf > best_score:
                best_score = sf
                best_h = hf.copy()

    return list(best_h)
