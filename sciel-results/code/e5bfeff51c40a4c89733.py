"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
DEADLINE = time.time() + 130  # 130s deadline (10s safety margin before 140s kill)

# EVOLVE-BLOCK-START
def _autoconv_interior(h):
    """Compute interior autoconvolution entries (lags 1..N-2)."""
    n = len(h)
    # Use FFT for speed
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv[1:n]  # lags 1..N-2


def _soft_max(x, beta):
    """Soft max using log-sum-exp trick."""
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    return x_max + np.log(np.sum(np.exp((x - x_max) * beta))) / beta


def construct():
    global DEADLINE
    deadline = DEADLINE

    # Start with interior-only pattern: zero first and last elements
    h = np.zeros(N)
    # Initialize interior with a smooth pattern
    x = np.linspace(0, 1, N - 2)
    h[1:-1] = np.cos(np.pi * x)  # positive bump

    best_h = h.copy()
    best_val = _objective(h)

    def _objective(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        conv = _autoconv_interior(h)
        # Also include boundary terms if h[0] or h[-1] are nonzero
        if abs(h[0]) > 1e-12 or abs(h[-1]) > 1e-12:
            boundary = np.array([np.dot(h, h), h[0] * h[-1]])
            conv = np.concatenate([boundary, conv])
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    best_val = _objective(h)

    # Gradient descent with soft-max smoothing
    beta = 50.0
    lr = 0.01
    patience = 0
    max_patience = 200

    while time.time() < deadline - 5:
        # Normalize to keep sum around 1 for numerical stability
        s = np.sum(h)
        if abs(s) < 1e-12:
            break
        h_norm = h / s  # sum = 1

        # Compute soft max of autoconvolution
        conv = _autoconv_interior(h_norm)
        # Include boundary if needed
        if abs(h_norm[0]) > 1e-12 or abs(h_norm[-1]) > 1e-12:
            boundary = np.array([np.dot(h_norm, h_norm), h_norm[0] * h_norm[-1]])
            conv_full = np.concatenate([boundary, conv])
        else:
            conv_full = conv

        # Soft max and its gradient
        x_max = np.max(conv_full)
        scaled = (conv_full - x_max) * beta
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        probs = exp_scaled / sum_exp  # softmax weights

        # Gradient of soft_max w.r.t. h: need d(conv)/dh
        # conv[k] = sum_i h[i] * h[i+k] for lag k
        # d(soft_max)/dh[j] = sum_k probs[k] * d(conv[k])/dh[j]
        # d(conv[k])/dh[j] = h[j-k] + h[j+k] (where valid indices)
        #
        # Let's compute gradient numerically for correctness, then switch to analytic
        # For speed, use analytic:
        grad = np.zeros(N)
        n = N
        # For each lag k, conv[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k]
        # d(conv[k])/dh[j] = h[j-k] (if j-k >= 0 and j-k <= n-1-k, i.e., i=j-k valid) 
        #                    + h[j+k] (if j+k < n-1 and j+k > k, i.e., i+k=j -> i=j-k... wait)
        # Actually conv_full[k] for lag k: sum_i h[i]*h[i+k]
        # d/dh[j]: term where i=j -> h[j]*h[j+k] contributes h[j+k] (if j+k < n)
        #           term where i+k=j -> i=j-k -> h[j-k]*h[j] contributes h[j-k] (if j-k >= 0)
        for k in range(len(conv_full)):
            w = probs[k]
            if w < 1e-15:
                continue
            # d(conv[k])/dh[j] for all j
            # conv[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k]
            # For j: contribution from i=j (if j <= n-1-k): h[j+k]
            #         contribution from i+k=j -> i=j-k (if j-k >= 0 and j-k <= n-1-k): h[j-k]
            if k == 0:
                # conv[0] = sum h[i]^2, d/dh[j] = 2*h[j]
                grad[j if False else slice(None)] += 0  # handled below
            # Let's just do it vectorized per k
            # indices where i=j: j from 0 to n-1-k
            j_range1 = np.arange(0, n - k)
            grad[j_range1] += w * h_norm[j_range1 + k]
            # indices where i+k=j: i=j-k, j from k to n-1
            j_range2 = np.arange(k, n)
            grad[j_range2] += w * h_norm[j_range2 - k]

        # Scale gradient back to original h (since we used h_norm)
        grad = grad / s  # chain rule: h_norm = h/s, d h_norm / dh = I/s - h/(s^2)... 
        # Actually let's just use h_norm for optimization and rescale at the end
        # Simpler: optimize h_norm directly, then rescale

        # Update h_norm
        h_norm = h_norm - lr * grad
        # Keep sum = 1
        s_new = np.sum(h_norm)
        if abs(s_new) < 1e-12:
            break
        h_norm = h_norm / s_new

        # Check value
        val = _objective(h_norm)
        if val < best_val:
            best_val = val
            best_h = h_norm.copy()
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            # Reduce learning rate and reset patience
            lr *= 0.5
            patience = 0
            beta = max(10.0, beta * 0.8)
            if lr < 1e-6:
                break

    # Final polish: try a few more iterations with smaller lr
    h = best_h.copy()
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s

    # Try local refinement with smaller steps
    lr_small = 0.001
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        conv = _autoconv_interior(h)
        if abs(h[0]) > 1e-12 or abs(h[-1]) > 1e-12:
            boundary = np.array([np.dot(h, h), h[0] * h[-1]])
            conv_full = np.concatenate([boundary, conv])
        else:
            conv_full = conv

        x_max = np.max(conv_full)
        scaled = (conv_full - x_max) * 100.0
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        probs = exp_scaled / sum_exp

        grad = np.zeros(N)
        for k in range(len(conv_full)):
            w = probs[k]
            if w < 1e-15:
                continue
            j_range1 = np.arange(0, N - k)
            grad[j_range1] += w * h[j_range1 + k]
            j_range2 = np.arange(k, N)
            grad[j_range2] += w * h[j_range2 - k]

        h = h - lr_small * grad
        s_new = np.sum(h)
        if abs(s_new) < 1e-12:
            break
        h = h / s_new

    val_final = _objective(h)
    if val_final < best_val:
        best_h = h.copy()

    return best_h
# EVOLVE-BLOCK-END
