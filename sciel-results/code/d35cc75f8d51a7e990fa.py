import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]


def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * DELTA * max_conv)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))


def _smooth_score(h, beta=50.0):
    """Smooth surrogate: L2^2 / (L1 * DELTA * soft_max_conv) where soft_max uses log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    h_conv = np.convolve(h, h, mode='full')
    # Soft max via log-sum-exp
    m = np.max(h_conv)
    if m <= 0:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (h_conv - m)))) / beta
    if soft_max <= 0:
        return 0.0
    return (l2_sq / (l1 * DELTA * soft_max))


def _project_onto_simplex(v, target_sum=1.0):
    """Project v onto the simplex {x >= 0, sum(x) = target_sum}."""
    n = len(v)
    if n == 0:
        return v.copy()
    v = np.asarray(v, dtype=np.float64)
    # Sort in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0][-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Initial best
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    x_idx = np.arange(N)

    # Generate diverse initializations
    inits = []

    # 1. Uniform
    inits.append(np.ones(N) / N)

    # 2. Gaussians with various centers and widths
    for c in np.linspace(3, N - 3, 15):
        for w in np.linspace(1, N / 2, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            s = np.sum(h_g)
            if s > 0:
                h_g /= s
                inits.append(h_g.copy())

    # 3. Triangles
    for c in np.linspace(3, N - 3, 15):
        for w in np.linspace(1, N, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            s = np.sum(h_t)
            if s > 0:
                h_t /= s
                inits.append(h_t.copy())

    # 4. Sparse peaks (Sidon-like)
    np.random.seed(42)
    for trial in range(50):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        # Random heights for peaks
        heights = np.random.rand(k) * 0.5 + 0.5
        h_s[idx] = heights
        s = np.sum(h_s)
        if s > 0:
            h_s /= s
            inits.append(h_s.copy())

    # 5. Random smooth functions
    for trial in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 7)) / 5
        h_r = np.convolve(h_r, kernel, mode='same')
        s = np.sum(h_r)
        if s > 0:
            h_r /= s
            inits.append(h_r.copy())

    # 6. Alternating patterns (to reduce autoconvolution overlap)
    for period in range(2, 8):
        h_alt = np.zeros(N)
        for i in range(N):
            if i % period < period // 2:
                h_alt[i] = 1.0
        s = np.sum(h_alt)
        if s > 0:
            h_alt /= s
            inits.append(h_alt.copy())

    # 7. Bimodal distributions
    for c1 in np.linspace(5, N / 2, 5):
        for c2 in np.linspace(N / 2, N - 5, 5):
            for w in np.linspace(1, N / 4, 5):
                h_b = np.exp(-0.5 * ((x_idx - c1) / w) ** 2) + np.exp(-0.5 * ((x_idx - c2) / w) ** 2)
                s = np.sum(h_b)
                if s > 0:
                    h_b /= s
                    inits.append(h_b.copy())

    # Evaluate all initializations
    for h_init in inits:
        if elapsed() > deadline - 10:
            break
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()

    # Optimization using projected gradient / coordinate descent on the smooth surrogate
    # Objective: maximize L2^2 / (L1 * DELTA * soft_max_conv)
    # With L1 = 1 constraint: maximize L2^2 / (DELTA * soft_max_conv)

    def _gradient_smooth(h, beta=50.0):
        """Compute gradient of the smooth score w.r.t. h (with L1=1 normalization implicit)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 == 0:
            return np.zeros_like(h)

        l2_sq = np.sum(h ** 2)
        h_conv = np.convolve(h, h, mode='full')

        m = np.max(h_conv)
        if m <= 0:
            return np.zeros_like(h)

        # Soft max
        exp_terms = np.exp(beta * (h_conv - m))
        sum_exp = np.sum(exp_terms)
        soft_max = m + np.log(sum_exp) / beta

        # d(soft_max)/d(h_conv) = exp_terms / sum_exp
        d_soft_max_d_conv = exp_terms / sum_exp

        # d(conv(h,h)_k)/d(h_i) = h_{k-i} for valid indices
        # conv(h,h)_k = sum_i h_i * h_{k-i}
        # So d(conv_k)/d(h_i) = h_{k-i} (with appropriate index handling)

        # Gradient of soft_max w.r.t. h:
        # d(soft_max)/d(h_i) = sum_k d_soft_max_d_conv[k] * d(conv_k)/d(h_i)
        # = sum_k d_soft_max_d_conv[k] * h_{k-i}

        # This is essentially a convolution of d_soft_max_d_conv with h
        # d(conv_k)/d(h_i) = h_{k-i} if 0 <= k-i < N, else 0
        # So d(soft_max)/d(h_i) = sum_k d_soft_max_d_conv[k] * h[k-i] (where valid)

        # This is cross-correlation / convolution
        # Let's compute it properly
        grad_soft_max = np.zeros(N)
        for i in range(N):
            # d(conv_k)/d(h_i) = h_{k-i} for k-i in [0, N-1]
            # i.e., k in [i, i+N-1]
            k_min = i
            k_max = min(i + N - 1, 2 * N - 2)
            if k_min <= k_max:
                valid_k = np.arange(k_min, k_max + 1)
                valid_j = valid_k - i  # j = k - i, so h_j
                grad_soft_max[i] = np.sum(d_soft_max_d_conv[valid_k] * h[valid_j])

        # Score = l2_sq / (DELTA * soft_max)  [with l1=1]
        # d(score)/d(h_i) = (2*h_i * soft_max - l2_sq * grad_soft_max[i]) / (DELTA * soft_max^2)
        # = (2*h_i / (DELTA * soft_max)) - (l2_sq * grad_soft_max[i]) / (DELTA * soft_max^2)

        score_grad = (2.0 * h / (DELTA * soft_max)) - (l2_sq * grad_soft_max) / (DELTA * soft_max ** 2)

        return score_grad

    def _optimize_from(h_init, max_iter=2000, lr_init=0.01):
        """Optimize using projected gradient with adaptive step size."""
        h = h_init.copy()
        h = _project_onto_simplex(h, 1.0)

        best_local = h.copy()
        best_local_score = _score_raw(h)

        lr = lr_init
        patience = 0
        max_patience = 50

        for it in range(max_iter):
            if elapsed() > deadline - 5:
                break

            # Compute gradient
            grad = _gradient_smooth(h, beta=50.0)

            # Adaptive step size
            h_new = h + lr * grad
            h_new = _project_onto_simplex(h_new, 1.0)

            s_new = _score_raw(h_new)
            if s_new > best_local_score:
                best_local_score = s_new
                best_local = h_new.copy()
                patience = 0
                # Increase LR slightly
                lr = min(lr * 1.05, 0.1)
            else:
                patience += 1
                lr *= 0.8

            h = h_new

            if patience > max_patience:
                break

            # Periodic restart with smaller LR
            if it % 200 == 0 and it > 0:
                lr = max(lr, 0.001)

        return best_local, best_local_score

    # Optimize from top initializations
    # First, rank initializations by score
    init_scores = []
    for h_init in inits:
        s = _score_raw(h_init)
        init_scores.append((s, h_init))
    init_scores.sort(key=lambda x: -x[0])

    # Take top 30 initializations
    top_inits = [h for _, h in init_scores[:30]]

    # Also include the current best
    if best_h not in top_inits:
        top_inits.insert(0, best_h.copy())

    for h_init in top_inits:
        if elapsed() > deadline - 15:
            break

        h_opt, s_opt = _optimize_from(h_init, max_iter=1500, lr_init=0.01)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Final polish: use a higher beta for the smooth surrogate and more iterations
    if elapsed() < deadline - 20:
        h_polish = best_h.copy()
        h_polish = _project_onto_simplex(h_polish, 1.0)

        # Coordinate descent polish
        for coord in range(N * 2):
            if elapsed() > deadline - 5:
                break
            for i in range(N):
                if elapsed() > deadline - 5:
                    break
                # Try perturbing h[i]
                for delta in [0.001, 0.005, 0.01, 0.02, -0.001, -0.005, -0.01, -0.02]:
                    h_try = h_polish.copy()
                    h_try[i] += delta
                    h_try = _project_onto_simplex(h_try, 1.0)
                    s_try = _score_raw(h_try)
                    if s_try > _score_raw(h_polish):
                        h_polish = h_try
                        break

        s_polish = _score_raw(h_polish)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Ensure valid output
    best_h = np.maximum(best_h, 0.0)
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h / s_final

    return list(best_h)
