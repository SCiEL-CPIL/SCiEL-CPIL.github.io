"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds margin


def _overlaps(h):
    """Compute overlap O(k) for all circular shifts k=0..N-1.
    O(k) = sum_i h[i] * (1 - h[(i+k) % N]) * dx, dx = 2/N."""
    dx = 2.0 / N
    # For each shift k, overlap = dx * sum_i h[i]*(1-h[i+k])
    # = dx * (sum h - sum h[i]*h[i+k]) = dx*(N/2 - corr[k]) where corr[k]=sum h[i]h[i+k]
    # Use FFT for all circular cross-correlations efficiently.
    h = np.asarray(h, dtype=np.float64)
    # circular autocorrelation via FFT
    F = np.fft.fft(h)
    corr = np.fft.ifft(F * np.conj(F)).real  # corr[k] = sum_i h[i]*h[(i+k)%N]
    dx = 2.0 / N
    O = dx * (0.5 * N - corr)  # since sum h = N/2
    return O


def _soft_max(O, tau):
    """Log-sum-exp soft max, scaled by tau."""
    m = O.max()
    return m + tau * np.log(np.sum(np.exp((O - m) / tau)))


def _objective(params, tau):
    """params: (a, f, phi) for h = 0.5 + a*sin(2pi*f*x + phi), clipped.
    Returns soft-max of overlaps (smooth surrogate for max overlap)."""
    a, f, phi = params
    x = np.linspace(0, 2.0, N, endpoint=False)
    h = 0.5 + a * np.sin(2.0 * np.pi * f * x + phi)
    h = np.clip(h, 0.0, 1.0)
    # enforce mean = 0.5 by subtracting deviation (then re-clip is approximate)
    h = h - (h.mean() - 0.5)
    h = np.clip(h, 0.0, 1.0)
    O = _overlaps(h)
    return _soft_max(O, tau), h


def _max_overlap(h):
    O = _overlaps(h)
    return O.max(), h


def construct():
    """Optimize a sinusoidal (and small higher-harmonic) pattern to minimize
    the max circular overlap of h with shifted 1-h, subject to h in [0,1]
    and sum(h) = N/2. Returns the step-height array."""
    global _START_TIME, _DEADLINE, _SAFETY

    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)[0]

    # Parameterization: h = 0.5 + sum_{m in modes} a_m * sin(2*pi*f_m*x + phi_m)
    # Use a few low-frequency modes for flexibility.
    modes = [1, 2, 3]
    # params: for each mode, (a, f, phi). f is fixed to mode (integer freq for circularity),
    # so params = [a1, phi1, a2, phi2, a3, phi3]
    n_params = 2 * len(modes)
    rng = np.random.default_rng(0)

    def make_h(params):
        x = np.linspace(0, 2.0, N, endpoint=False)
        h = np.full(N, 0.5)
        for i, m in enumerate(modes):
            a = params[2 * i]
            phi = params[2 * i + 1]
            h = h + a * np.sin(2.0 * np.pi * m * x + phi)
        h = np.clip(h, 0.0, 1.0)
        # enforce mean 0.5
        h = h - (h.mean() - 0.5)
        h = np.clip(h, 0.0, 1.0)
        return h

    def obj(params):
        h = make_h(params)
        O = _overlaps(h)
        sm = _soft_max(O, tau)
        # penalty for clipping violations / mean
        return sm, h

    # Adam optimizer (manual)
    def adam_step(params, grads, state, lr, t):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = state['m']
        v = state['v']
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * grads ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        state['m'] = m
        state['v'] = v
        return params

    # Multi-start with annealing tau
    best_tau = 0.05
    for start in range(8):
        if time.time() > _DEADLINE - _SAFETY:
            break
        # initialize params
        if start == 0:
            params = np.zeros(n_params)
        else:
            params = np.zeros(n_params)
            for i in range(len(modes)):
                params[2 * i] = rng.uniform(-0.25, 0.25)
                params[2 * i + 1] = rng.uniform(0, 2 * np.pi)

        state = {'m': np.zeros(n_params), 'v': np.zeros(n_params)}
        lr = 0.05
        t_iter = 0
        tau = 0.1
        n_iter = 400
        for it in range(n_iter):
            if time.time() > _DEADLINE - _SAFETY:
                break
            t_iter += 1
            # numerical gradient via central differences (n_params small)
            h = make_h(params)
            O = _overlaps(h)
            base = _soft_max(O, tau)
            grads = np.zeros(n_params)
            eps = 1e-4
            for j in range(n_params):
                pp = params.copy()
                pm = params.copy()
                pp[j] += eps
                pm[j] -= eps
                hp = make_h(pp)
                Op = _overlaps(hp)
                fp = _soft_max(Op, tau)
                hm = make_h(pm)
                Om = _overlaps(hm)
                fm = _soft_max(Om, tau)
                grads[j] = (fp - fm) / (2 * eps)
            params = adam_step(params, grads, state, lr, t_iter)
            # anneal tau and lr
            if it % 50 == 49:
                tau = max(0.02, tau * 0.8)
                lr = max(0.005, lr * 0.9)

        # evaluate final
        h = make_h(params)
        val, _ = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Final polish: local refinement with tiny steps on best_h directly
    if time.time() < _DEADLINE - _SAFETY:
        h = best_h.copy()
        for _ in range(200):
            if time.time() > _DEADLINE - _SAFETY:
                break
            O = _overlaps(h)
            # subgradient of max: weight on argmax shifts
            kmax = np.argmax(O)
            # derivative of O(k) w.r.t. h[i]: dx*(delta_{i,k} - h[i-k]) ... 
            # O(k) = dx*sum_i h[i](1-h[i+k])
            # dO(k)/dh[i] = dx*(1 - h[i+k]) - dx*h[i-k]  (since h[i] appears in term i of sum, and in term i-k as h[(i-k)+k]=h[i])
            dx = 2.0 / N
            grad = np.zeros(N)
            for i in range(N):
                term1 = (1 - h[(i + kmax) % N])  # from h[i] in O(kmax) term i
                term2 = h[(i - kmax) % N]        # from h[i] as h[(i-kmax)+kmax] in term (i-kmax)
                grad[i] = dx * (term1 - term2)
            # project: move h opposite to grad, keep in [0,1], fix mean
            step = 0.01
            h_new = h - step * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new = h_new - (h_new.mean() - 0.5)
            h_new = np.clip(h_new, 0.0, 1.0)
            val_new, _ = _max_overlap(h_new)
            if val_new < val:
                h = h_new
                val = val_new
            else:
                step *= 0.5
            if step < 1e-5:
                break
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Ensure exact sum = N/2 and in [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = best_h - (best_h.mean() - 0.5)
    best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
# EVOLVE-BLOCK-END
