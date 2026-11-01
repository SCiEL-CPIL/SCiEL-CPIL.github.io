"""Optimized signed step function via Gumbel-Softmax variational inference."""
import numpy as np
import time

N = 400
DT = 0.5 / N


def _autoconv(s):
    """Compute autoconvolution R[0..N-1] for signal s of length N."""
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(s, n)
    R = np.fft.irfft(F * np.conj(F), n)
    return R[:N] * DT


def _score(s):
    """Return max |R[k]| for k=1..N-1 (sidelobe peak)."""
    R = _autoconv(s)
    return np.max(np.abs(R[1:]))


def _gumbel_sample(logits, temp, rng):
    """Sample from Gumbel-Sigmoid: returns (hard_signs, soft_y, gumbel_noise)."""
    u = rng.random(N)
    u = np.clip(u, 1e-8, 1 - 1e-8)
    g = -np.log(-np.log(u))
    z = (logits + g) / temp
    y = 1.0 / (1.0 + np.exp(-z))
    hard = np.where(y > 0.5, 1.0, -1.0)
    return hard, y, g


def _grad_logits(logits, temp, rng, tau=0.005):
    """
    Compute stochastic gradient of soft-peak loss w.r.t. logits.
    Uses one Gumbel sample and backprop through soft sigmoid.
    """
    hard, y, g = _gumbel_sample(logits, temp, rng)
    s_soft = 2.0 * y - 1.0  # soft sign in [-1, 1]

    R = _autoconv(s_soft)
    side = R[1:]
    side_sq = side ** 2
    m = np.max(side_sq)
    if m < 1e-12:
        return np.zeros(N), 0.0

    # Soft-max surrogate for max(side_sq)
    exp_t = np.exp((side_sq - m) / tau)
    sum_exp = np.sum(exp_t)
    dL_dR_sq = exp_t / sum_exp
    dL_dR = 2.0 * side * dL_dR_sq

    # dR/ds_i = sum_k dL_dR[k] * (s_{i-k} + s_{i+k})  for k=1..N-1
    # Vectorized via correlation: for each i, sum over k of w_k * (s[i-k]+s[i+k])
    # where w_k = dL_dR[k] for k=1..N-1
    # This is a circular-like correlation but with boundary truncation.
    # Build weight vector w of length N: w[0]=0, w[k]=dL_dR[k] for k=1..N-1
    w = np.zeros(N)
    w[1:] = dL_dR

    # dL_ds[i] = sum_{k=1}^{N-1} w[k] * (s[i-k] + s[i+k]) with bounds
    # = (w * s)_circ[i] + (w * s_flip)_circ[i]  ... but not exactly circular.
    # Use FFT-based linear convolution to compute sum_k w[k]*s[i-k] and sum_k w[k]*s[i+k].
    # sum_k w[k]*s[i-k] = (w * s)[i]  (linear cross-correlation, but w is not symmetric)
    # Actually: (w * s)[i] = sum_j w[j] * s[i-j]. With j=k, this is sum_k w[k]*s[i-k]. Good.
    # sum_k w[k]*s[i+k] = sum_k w[k]*s[i+k]. Let j=-k, then = sum_j w[-j]*s[i-j].
    # Define w_neg[j] = w[-j] for j in -N+1..N-1. But easier: reverse w and correlate.
    # sum_k w[k]*s[i+k] = (w_rev * s)[N-1-i] where w_rev[k]=w[N-1-k]? Let's just do it via FFT.

    n_fft = 1
    while n_fft < 2 * N:
        n_fft <<= 1
    W = np.fft.rfft(w, n_fft)
    S = np.fft.rfft(s_soft, n_fft)
    # corr1[i] = sum_k w[k] * s[i-k]  = ifft( W * conj(S) )[i]  ... need to be careful.
    # Actually np.correlate(w, s, mode='full') gives sum_k w[k]*s[i+k] offset.
    # Let's use direct FFT approach:
    # corr_a[i] = sum_{k} w[k] * s[i-k]  for valid i
    # This is the convolution of w and s, evaluated at i.
    conv_ws = np.fft.irfft(W * S, n_fft)
    # conv_ws[i] = sum_j w[j] * s[i-j]  (linear convolution, zero-padded)
    # We want sum_{k=1}^{N-1} w[k]*s[i-k] = conv_ws[i] - w[0]*s[i] = conv_ws[i] since w[0]=0
    corr_a = conv_ws[:N]

    # corr_b[i] = sum_{k=1}^{N-1} w[k] * s[i+k]
    # = sum_k w[k]*s[i+k]. Let m = i+k, then = sum_m w[m-i]*s[m]. This is cross-correlation.
    # corr_b[i] = sum_m s[m] * w[m-i] = (s * w_rev)[i] where w_rev[k]=w[-k]?
    # Easier: reverse w, then convolve with s.
    w_rev = w[::-1]
    W_rev = np.fft.rfft(w_rev, n_fft)
    conv_s_wr = np.fft.irfft(S * W_rev, n_fft)
    # conv_s_wr[i] = sum_j s[j] * w_rev[i-j] = sum_j s[j] * w[N-1-(i-j)] = sum_j s[j]*w[N-1-i+j]
    # We want sum_k w[k]*s[i+k]. Let j=i+k, k=j-i. sum_j s[j]*w[j-i].
    # This is (s * w)[j] evaluated... let me just compute it directly with a small loop-free trick.
    # Alternative: corr_b[i] = sum_{k=1}^{N-1} w[k] * s[i+k]
    # Build an extended s: s_ext of length 2N, s_ext[0:N]=s, s_ext[N:2N]=s
    # Then corr_b[i] = sum_{k=1}^{N-1} w[k] * s_ext[i+k]
    # = (w * s_ext)[i] but w has length N, s_ext has length 2N.
    # Let's just use the fact that this is a shifted correlation.
    # For now, use a vectorized loop over k which is O(N^2) but N=400, so 160k ops.
    # But we want speed. Let me use FFT properly.
    # corr_b[i] = sum_k w[k] s[i+k]
    # Define s_shifted such that (s * w_neg)[i] where w_neg[k] = w[-k]
    # w_neg has w_neg[-k]=w[k]. In array form, w_neg is w reversed with offset.
    # Let's create w_full of length 2N-1: w_full[k] = w[k] for k=0..N-1, 0 otherwise.
    # w_neg_full[k] = w_full[N-1-k] for k=0..2N-2? No.
    # I'll just do the O(N^2) computation since N=400 and we're doing this maybe 100 times.
    # But 400*400=160k, times 100 steps = 16M ops, fine in NumPy if vectorized.
    # Vectorized: for each k, add w[k] * s[i+k] for valid i.
    # s[i+k] for i=0..N-1-k-1. So for each k, we add w[k] * s[k:N] to corr_b[0:N-k].
    corr_b = np.zeros(N)
    for k in range(1, N):
        corr_b[:N-k] += w[k] * s_soft[k:N]

    dL_ds = corr_a + corr_b

    # Backprop: dL/dlogits = dL_ds * ds/dy * dy/dz * dz/dlogits
    # s_soft = 2y-1 => ds/dy = 2
    # y = sigmoid(z) => dy/dz = y(1-y)
    # z = (logits+g)/temp => dz/dlogits = 1/temp
    dy_dz = y * (1.0 - y)
    dL_dlogits = dL_ds * 2.0 * dy_dz / temp

    loss_val = m + tau * np.log(sum_exp)
    return dL_dlogits, loss_val


def _local_search(signs, max_time=10.0):
    """Greedy 1-flip local search with FFT-based scoring."""
    start = time.time()
    cur = signs.copy()
    cur_score = _score(cur)
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for i in range(N):
            new = cur.copy()
            new[i] *= -1.0
            ns = _score(new)
            if ns < cur_score - 1e-12:
                cur = new
                cur_score = ns
                improved = True
    return cur, cur_score


def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    rng = np.random.default_rng(42)

    # Initialize logits to zero (uniform prior)
    logits = np.zeros(N)
    best_signs = np.ones(N)
    best_score = _score(best_signs)

    # Phase 1: Variational gradient descent with Gumbel-Softmax
    temp_start = 1.0
    temp_end = 0.1
    n_steps = 150
    lr = 0.5

    for step in range(n_steps):
        if time.time() > deadline - 30:
            break
        temp = temp_start * (temp_end / temp_start) ** (step / max(n_steps - 1, 1))
        lr_step = lr * (1.0 - 0.5 * step / n_steps)  # decay learning rate

        grad, loss = _grad_logits(logits, temp, rng)
        # Normalize gradient to avoid explosion
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-6:
            grad = grad / gnorm * min(gnorm, 1.0)
        logits -= lr_step * grad

        # Sample hard signs
        hard, y, g = _gumbel_sample(logits, temp, rng)
        sc = _score(hard)
        if sc < best_score:
            best_score = sc
            best_signs = hard.copy()

    # Phase 2: Local search refinement
    remaining = deadline - time.time() - 5
    if remaining > 2:
        best_signs, best_score = _local_search(best_signs, max_time=min(remaining, 60.0))

    # Phase 3: Multiple random restarts with local search if time remains
    remaining = deadline - time.time() - 5
    if remaining > 5:
        n_restarts = min(int(remaining / 10), 5)
        for _ in range(n_restarts):
            if time.time() > deadline - 5:
                break
            # Random restart
            rand_signs = rng.choice([-1.0, 1.0], size=N)
            refined, sc = _local_search(rand_signs, max_time=min(remaining, 15.0))
            if sc < best_score:
                best_score = sc
                best_signs = refined
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break

    return best_signs
