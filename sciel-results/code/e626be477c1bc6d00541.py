import numpy as np
import time

N = 600
DEADLINE = 142  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    # Build Legendre basis: h(n) = sum_j a_j * P_j(x_n), x_n in [-1, 1]
    # We use first M Legendre polynomials. h must be nonnegative.
    M = 40  # number of Legendre modes
    nn = np.arange(N)
    # Map interval indices to [-1, 1]
    x = 2.0 * nn / (N - 1) - 1.0

    # Compute Legendre polynomials P_0..P_{M-1} at x
    # Using recurrence: P_0=1, P_1=x, P_{k+1} = ((2k+1)x P_k - k P_{k-1})/(k+1)
    P = np.zeros((M, N))
    P[0, :] = 1.0
    if M > 1:
        P[1, :] = x
    for k in range(1, M - 1):
        P[k+1, :] = ((2*k + 1) * x * P[k, :] - k * P[k-1, :]) / (k + 1)

    # h = P.T @ a, where a is coefficient vector of length M
    # We need h >= 0, so we'll use a soft penalty for negative values.

    # Initial guess: constant function (a_0 = 1, rest = 0)
    a = np.zeros(M)
    a[0] = 1.0

    def h_from_a(a):
        return P.T @ a

    def metric_and_grad(a):
        """Compute metric value and gradient w.r.t. a."""
        h = h_from_a(a)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(M)
        
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        max_c = np.max(c)
        val = max_c / (s * s)
        
        # Gradient of val = max_c / s^2 w.r.t. a
        # d(val)/da = (dc/da * s^2 - max_c * 2s * ds/da) / s^4
        #           = (dc/da) / s^2 - 2 * max_c * ds/da / s^3
        # dc/da: derivative of max(c) w.r.t. a. Use subgradient: c_j where j = argmax(c)
        # ds/da: derivative of sum(h) w.r.t. a = sum of P_j over n for each j = P[:, :].sum(axis=1)
        
        j_max = np.argmax(c)
        # c = autoconvolution of h. c[j] = sum_n h[n] * h[n+j] (for j >= 0)
        # dc[j]/da_m = sum_n (dh[n]/da_m * h[n+j] + h[n] * dh[n+j]/da_m)
        # dh[n]/da_m = P[m, n]
        # So dc[j]/da_m = sum_n P[m,n]*h[n+j] + sum_n h[n]*P[m,n+j]
        #              = (P[:, j:] @ h[:N-j]) + (P[:, :N-j] @ h[j:])
        # where we handle boundaries carefully.
        
        # More carefully: c[j] = sum_{n=0}^{N-1-j} h[n] * h[n+j] for j >= 0
        # dc[j]/da_m = sum_{n=0}^{N-1-j} [P[m,n]*h[n+j] + h[n]*P[m,n+j]]
        
        # Vectorized: for a fixed j, 
        # term1 = P[:, :N-j] @ h[j:]  (but P has shape (M, N), so P[:, :N-j] is (M, N-j))
        # term2 = P[:, j:] @ h[:N-j]
        
        if j_max == 0:
            # c[0] = sum h[n]^2
            # dc[0]/da_m = 2 * sum h[n] * P[m,n] = 2 * (h @ P[m, :])
            dc_da = 2.0 * P @ h
        else:
            # c[j] = sum_{n=0}^{N-1-j} h[n]*h[n+j]
            # dc[j]/da_m = sum_{n=0}^{N-1-j} P[m,n]*h[n+j] + h[n]*P[m,n+j]
            # = P[:, :N-j] @ h[j:] + P[:, j:] @ h[:N-j]
            dc_da = P[:, :N-j_max] @ h[j_max:] + P[:, j_max:] @ h[:N-j_max]
        
        ds_da = P.sum(axis=1)  # sum over n of P[m, n]
        
        grad = dc_da / (s * s) - 2.0 * max_c * ds_da / (s * s * s)
        return val, grad

    # Soft penalty for non-negativity: penalty = beta * sum(max(0, -h)^2)
    # This encourages h >= 0

    def penalized_objective(a, beta):
        h = h_from_a(a)
        neg = np.maximum(0.0, -h)
        penalty = beta * np.sum(neg * neg)
        val, grad_base = metric_and_grad(a)
        # Gradient of penalty w.r.t. a: beta * sum(2 * (-h) * (-dh/da) for h < 0)
        # = beta * 2 * sum(max(0, -h) * (-P[:, :].T) for negative h)
        # d(penalty)/da_m = beta * 2 * sum_n max(0, -h[n]) * (-P[m, n])
        neg_mask = np.maximum(0.0, -h)
        penalty_grad = -2.0 * beta * P @ neg_mask
        return val + penalty, grad_base + penalty_grad

    # Start with constant
    a = np.zeros(M)
    a[0] = 1.0
    h = h_from_a(a)
    best_val = fast_metric(h)
    best_a = a.copy()

    # Projected gradient descent with adaptive step size
    beta = 10.0  # penalty weight for non-negativity
    lr = 0.01
    n_iter = 5000
    
    for it in range(n_iter):
        if elapsed() > DEADLINE - 30:
            break
        
        val, grad = penalized_objective(a, beta)
        
        # Gradient step
        a_new = a - lr * grad
        
        # Soft projection: keep a in a reasonable range
        # No hard constraint on a, but h = P.T @ a should be nonnegative
        # We can add a small regularization to keep a bounded
        a_new = np.clip(a_new, -10, 10)
        
        h_new = h_from_a(a_new)
        if np.any(h_new < -0.1):  # too many negatives, shrink step
            lr *= 0.5
            continue
        
        v_new = fast_metric(np.maximum(h_new, 0))
        if v_new < best_val:
            best_val = v_new
            best_a = a_new.copy()
        
        # Adaptive step size
        if v_new < val:
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
        
        a = a_new
        
        # Periodically increase beta to enforce non-negativity more strongly
        if it % 500 == 499 and beta < 1000:
            beta *= 2.0

    # Reconstruct h from best_a and clip to nonnegative
    h_best = h_from_a(best_a)
    h_best = np.maximum(h_best, 0)
    
    if np.sum(h_best) < 1e-15:
        h_best = np.ones(N)
    else:
        # Scale to convenient magnitude
        h_best = h_best / np.sum(h_best) * N

    # Phase 2: Coordinate descent polish
    current_h = h_best.copy()
    current_val = fast_metric(current_h)
    np.random.seed(42)

    no_improve = 0
    max_no_improve = 5000
    mean_h = np.mean(current_h)

    while elapsed() < DEADLINE - 15 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        mean_h = np.mean(current_h)
        step_factors = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig
        for cv in candidates_val:
            if abs(cv - orig) < 1e-13:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-13:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                h_best = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 400 == 0:
            # Perturbation to escape local minima
            pert = current_h.copy()
            pert += 0.03 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        h_best = current_h.copy()

    # Phase 3: Fine polish
    if elapsed() < DEADLINE - 5:
        step_frac = 0.001
        for _ in range(2000):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(h_best)
            new_h = h_best.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    h_best = new_h

    h_best = np.maximum(h_best, 0)
    if np.sum(h_best) < 1e-15:
        h_best = np.ones(N)

    return h_best.tolist()
