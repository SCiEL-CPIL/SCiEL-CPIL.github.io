import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import sys

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


def _logsumexp_conv(h, beta=50.0):
    """Compute log-sum-exp of autoconvolution with temperature beta."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 1e-15:
        return 0.0, conv, m
    lse = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    return lse, conv, m


def _lse_grad(h, beta=50.0):
    """Gradient of log-sum-exp of autoconvolution w.r.t. h."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 1e-15:
        return np.zeros(N)
    weights = np.exp(beta * (conv - m))
    weights /= np.sum(weights)
    # d/dh_i LSE = sum_k weights_k * dconv_k/dh_i = sum_k weights_k * 2*h_{k-i}
    grad = np.zeros(N)
    for i in range(N):
        # conv_k = sum_j h_j * h_{k-j}, dconv_k/dh_i = sum_j h_{k-j} * delta_{j,i} + h_i * delta_{k-i,j}
        # Actually dconv_k/dh_i = h_{k-i} + h_{k-i} (if i and k-i are both valid)
        # More precisely: conv_k = sum_{j=0}^{N-1} h_j * h_{k-j} (with h out of range = 0)
        # dconv_k/dh_i = h_{k-i} (from first term, j=i) + h_i (from second term, k-j=i => j=k-i)
        # But when i == k-i, it's 2*h_i
        # Let me just compute it directly
        for k in range(2 * N - 1):
            j1 = i
            j2 = k - i
            val = 0.0
            if 0 <= j1 < N and 0 <= j2 < N:
                val += h[j2]
            if 0 <= j2 < N and 0 <= j1 < N:
                val += h[j1]
            # Wait, this double counts when j1==j2. Let me redo.
            # conv_k = sum_{j: 0<=j<N, 0<=k-j<N} h_j * h_{k-j}
            # dconv_k/dh_i:
            #   from h_j term: if j==i, contribution is h_{k-i} (if 0<=k-i<N)
            #   from h_{k-j} term: if k-j==i, i.e., j==k-i, contribution is h_i (if 0<=k-i<N)
            # So dconv_k/dh_i = h_{k-i}*1_{0<=k-i<N} + h_i*1_{0<=k-i<N}
            # = (h_{k-i} + h_i) * 1_{0<=k-i<N}
            # But when k-i == i (i.e., k=2i), this is 2*h_i, which is correct.
            if 0 <= k - i < N:
                val = h[k - i] + h[i]
            else:
                val = 0.0
            grad[i] += weights[k] * val
    return grad


def _lse_grad_fast(h, beta=50.0):
    """Faster gradient of log-sum-exp of autoconvolution using FFT."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 1e-15:
        return np.zeros(N)
    weights = np.exp(beta * (conv - m))
    weights /= np.sum(weights)
    # dconv_k/dh_i = (h_{k-i} + h_i) for 0<=k-i<N, else 0
    # grad_i = sum_k weights_k * (h_{k-i} + h_i) * 1_{0<=k-i<N}
    # = sum_k weights_k * h_{k-i} * 1_{0<=k-i<N} + h_i * sum_k weights_k * 1_{0<=k-i<N}
    # First term: for fixed i, sum over k where 0<=k-i<N, i.e., i<=k<i+N
    #   sum_{k=i}^{i+N-1} weights_k * h_{k-i} = sum_{j=0}^{N-1} weights_{i+j} * h_j
    # Second term: h_i * sum_{k=i}^{i+N-1} weights_k
    grad = np.zeros(N)
    for i in range(N):
        lo = i
        hi = min(i + N, 2 * N - 1)
        if lo > hi:
            continue
        w_slice = weights[lo:hi+1]
        h_slice = h[:len(w_slice)]
        grad[i] = np.dot(w_slice, h_slice) + h[i] * np.sum(w_slice)
    return grad


def _objective_lse(x, beta=50.0):
    """Minimize negative log-squared L2 norm + penalty on LSE of autoconvolution.
    x = [h_0, ..., h_{N-1}, M] where M is the LSE constraint variable.
    We minimize: -log(sum(h^2)) + lambda * (LSE(h) - M) subject to M >= 0, sum(h)=1, h>=0
    Actually, let's do: minimize M - log(sum(h^2)) subject to LSE(h) <= M, sum(h)=1, h>=0
    """
    h = x[:N]
    M = x[N]
    l2_sq = np.sum(h**2)
    if l2_sq <= 1e-15:
        return 1e10
    lse, conv, m = _logsumexp_conv(h, beta)
    # Penalty for violating LSE <= M
    penalty = 0.0
    if lse > M:
        penalty = 1000.0 * (lse - M)**2
    return M - np.log(l2_sq) + penalty


def _objective_lse_grad(x, beta=50.0):
    h = x[:N]
    M = x[N]
    l2_sq = np.sum(h**2)
    if l2_sq <= 1e-15:
        return np.concatenate([np.zeros(N), [0.0]])
    lse, conv, m = _logsumexp_conv(h, beta)
    grad_lse = _lse_grad_fast(h, beta)
    
    # d/dh_i [M - log(l2_sq)] = -2*h_i / l2_sq
    grad_h = -2.0 * h / l2_sq
    # d/dh_i penalty
    if lse > M:
        grad_h += 2000.0 * (lse - M) * grad_lse
    grad_M = 1.0
    if lse > M:
        grad_M -= 2000.0 * (lse - M)
    
    return np.concatenate([grad_h, [grad_M]])


def _objective_direct(x, beta=50.0):
    """Directly minimize: max_conv - log(l2_sq) using LSE approximation.
    x = h (N variables), with constraint sum(h)=1, h>=0.
    """
    h = x
    l2_sq = np.sum(h**2)
    if l2_sq <= 1e-15:
        return 1e10
    lse, conv, m = _logsumexp_conv(h, beta)
    return lse - np.log(l2_sq)


def _objective_direct_grad(x, beta=50.0):
    h = x
    l2_sq = np.sum(h**2)
    if l2_sq <= 1e-15:
        return np.zeros(N)
    lse, conv, m = _logsumexp_conv(h, beta)
    grad_lse = _lse_grad_fast(h, beta)
    grad = grad_lse - 2.0 * h / l2_sq
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit

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

    # Add alternating pattern inits
    for period in [2, 3, 4, 5]:
        h_alt = np.zeros(N)
        for i in range(N):
            if i % period < period // 2:
                h_alt[i] = 1.0
        if np.sum(h_alt) > 1e-10:
            h_alt /= np.sum(h_alt)
            inits.append(h_alt)

    scored_inits = []
    for h_init in inits:
        scored_inits.append((_score_raw(h_init), h_init))
    scored_inits.sort(key=lambda x: -x[0])

    # --- Phase 2: L-BFGS-B on log-sum-exp formulation ---
    # Minimize LSE(h) - log(sum(h^2)) subject to sum(h)=1, h>=0
    # Use equality constraint via penalty or reparameterization
    
    # Approach: use scipy minimize with constraints
    def run_lse_lbfgs(h_init, maxiter=500, beta=50.0):
        h_init = h_init / np.sum(h_init)
        
        # Use SLSQP with the smooth objective
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(_objective_direct, h_init.copy(), method='SLSQP',
                          jac=_objective_direct_grad, args=(beta,),
                          bounds=bounds, constraints=cons,
                          options={'maxiter': maxiter, 'ftol': 1e-14})
            if res.success:
                hc = np.clip(res.x, 0, None)
                if np.sum(hc) > 1e-15:
                    return hc / np.sum(hc)
        except Exception:
            pass
        
        # Try L-BFGS-B with barrier
        try:
            # Use log-barrier for simplex constraint
            def barrier_obj(x, mu=0.01):
                h = x
                if np.any(h <= 0):
                    return 1e10
                l2_sq = np.sum(h**2)
                if l2_sq <= 1e-15:
                    return 1e10
                lse, _, _ = _logsumexp_conv(h, beta)
                return lse - np.log(l2_sq) - mu * np.sum(np.log(h))
            
            def barrier_grad(x, mu=0.01):
                h = x
                if np.any(h <= 0):
                    return np.ones(N)
                l2_sq = np.sum(h**2)
                if l2_sq <= 1e-15:
                    return np.ones(N)
                grad_lse = _lse_grad_fast(h, beta)
                return grad_lse - 2.0 * h / l2_sq - mu / h
            
            res = minimize(barrier_obj, h_init.copy(), method='L-BFGS-B',
                          jac=lambda x: barrier_grad(x, mu=0.01),
                          options={'maxiter': maxiter, 'ftol': 1e-14})
            hc = np.clip(res.x, 0, None)
            if np.sum(hc) > 1e-15:
                return hc / np.sum(hc)
        except Exception:
            pass
        
        return None

    for s_init, h_init in scored_inits[:20]:
        if not now_ok(40.0):
            break
        hc = run_lse_lbfgs(h_init, maxiter=500, beta=50.0)
        if hc is not None:
            sc = _score_raw(hc)
            if sc > best_score:
                best_score = sc
                best_h = hc.copy()

    # --- Phase 3: Epigraph SLSQP (keep as backup) ---
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

    def run_epigraph(h_init, maxiter=300, ftol=1e-12):
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

    for s_init, h_init in scored_inits[:15]:
        if not now_ok(30.0):
            break
        hc = run_epigraph(h_init, maxiter=300, ftol=1e-12)
        if hc is not None:
            sc = _score_raw(hc)
            if sc > best_score:
                best_score = sc
                best_h = hc.copy()

    # --- Phase 4: Projected gradient ascent with LSE penalty ---
    def run_lse_projected(h_init, iters=100, beta=30.0):
        h = h_init / np.sum(h_init)
        for _ in range(iters):
            if not now_ok(10.0):
                break
            conv = np.convolve(h, h, mode='full')
            m = np.max(conv)
            if m <= 1e-15:
                break
            weights = np.exp(beta * (conv - m))
            weights /= np.sum(weights)
            # Gradient of LSE
            grad_lse = _lse_grad_fast(h, beta)
            # Gradient of log(l2_sq) = 2h / l2_sq
            l2_sq = np.sum(h**2)
            grad_obj = grad_lse - 2.0 * h / l2_sq
            # Projected gradient descent on LSE - log(l2_sq)
            # We want to minimize, so step in -grad direction
            # But we need to stay on simplex
            step_size = 0.01
            h_new = h - step_size * grad_obj
            h_new = _project_simplex(h_new, 1.0)
            h_new = np.clip(h_new, 0, None)
            if np.sum(h_new) < 1e-15:
                break
            h_new /= np.sum(h_new)
            
            # Check improvement
            if _score_raw(h_new) >= _score_raw(h):
                h = h_new
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        return h

    for s_init, h_init in scored_inits[:10]:
        if not now_ok(20.0):
            break
        hc = run_lse_projected(h_init, iters=100, beta=30.0)
        sc = _score_raw(hc)
        if sc > best_score:
            best_score = sc
            best_h = hc.copy()

    # --- Phase 5: Block-coordinate descent on symmetric pairs ---
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

    # --- Phase 6: Local random perturbations ---
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

    # --- Phase 7: Final high-precision polish ---
    if now_ok(5.0):
        hp = run_epigraph(best_h, maxiter=500, ftol=1e-14)
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
