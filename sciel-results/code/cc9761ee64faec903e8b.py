import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Strategy: Frank-Wolfe on the simplex to maximize f(h) = ||h||_2^2 / max_conv(h)
    # Subject to h >= 0, sum(h) = 1.
    
    def frank_wolfe(initial_h, max_iter=5000, tol=1e-8):
        h = np.asarray(initial_h, dtype=np.float64)
        h = h / np.sum(h)
        
        best_f = compute_ratio(h)
        best_h_fw = h.copy()
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            f = compute_ratio(h)
            if f > best_f:
                best_f = f
                best_h_fw = h.copy()
                
            # Compute gradient of f(h)
            # f(h) = S2 / (S1 * M). With S1=1, f = S2 / M.
            # grad S2 = 2h
            # grad M = e_k where k is argmax of conv(h,h).
            # grad f = (1/M) * grad S2 - (S2/M^2) * grad M
            #        = (1/M) * (2h - (S2/M) * e_k)
            
            conv_h = np.convolve(h, h, mode='full')
            M = np.max(conv_h)
            if M < 1e-15:
                break
            
            k_max = np.argmax(conv_h)
            
            S2 = np.sum(h * h)
            
            # Gradient
            grad = (1.0 / M) * (2.0 * h - (S2 / M) * np.zeros(2*N - 1))
            # Wait, the gradient is with respect to h.
            # f = S2 / M
            # df/dh_i = (2 h_i * M - S2 * dM/dh_i) / M^2
            # dM/dh_i is 0 unless i is part of the max convolution pair.
            # If conv_max is at lag k, then conv_k = sum_j h_j h_{j+k} (with indices adjusted).
            # Actually, it's easier to just compute the gradient numerically or analytically.
            # Analytical:
            # M = max_l conv_l. Let L be the set of lags where conv_l = M.
            # If we assume a unique max at k, then dM/dh_i = h_{i-k} + h_{i+k} (roughly, depending on definition).
            # Let's use a subgradient or just the gradient assuming unique max.
            
            # Let's compute the gradient properly.
            # conv(h,h) at lag d is sum_i h_i h_{i+d}.
            # d/dd conv_d = h_{i+d} + h_{i-d} (summed over valid i).
            # So grad M is a vector where (grad M)_i = h_{i-k} + h_{i+k} if k is the unique max lag.
            # Note: h is defined on 0..N-1.
            
            grad_M = np.zeros(N)
            # If k is the max lag, then conv_k = sum_{i} h_i h_{i+k} where i+k < N.
            # d conv_k / d h_i = h_{i+k} (if i+k < N) + h_{i-k} (if i-k >= 0).
            
            for i in range(N):
                if i + k_max < N:
                    grad_M[i] += h[i + k_max]
                if i - k_max >= 0:
                    grad_M[i] += h[i - k_max]
            
            # grad f = (M * 2h - S2 * grad_M) / M^2
            grad_f = (M * 2.0 * h - S2 * grad_M) / (M * M)
            
            # Frank-Wolfe step:
            # v = argmax_{v in simplex} <grad_f, v> = e_j where j = argmax grad_f
            j = np.argmax(grad_f)
            v = np.zeros(N)
            v[j] = 1.0
            
            # Line search:
            # min_{gamma in [0,1]} f(h + gamma (v - h))
            # We can do a few steps of line search.
            
            def line_search(h_cur, v, gamma):
                h_new = h_cur + gamma * (v - h_cur)
                # Ensure non-negativity (should be fine for gamma in [0,1])
                if np.min(h_new) < 0:
                    return 0.0, h_cur
                return compute_ratio(h_new), h_new
            
            # Golden section or simple ternary search
            a, b = 0.0, 1.0
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            f_c, h_c = line_search(h, v, c)
            f_d, h_d = line_search(h, v, d)
            
            for _ in range(20):
                if abs(b - a) < 1e-4:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    f_c, h_c = line_search(h, v, c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    f_d, h_d = line_search(h, v, d)
            
            gamma_best = (a + b) / 2
            f_best, h_new = line_search(h, v, gamma_best)
            
            if f_best > f + 1e-10:
                h = h_new
            else:
                break
                
        return best_h_fw, best_f

    # Multiple random restarts with different sparsities
    rng = np.random.RandomState(42)
    
    for K in [10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random support
            support = rng.choice(N, size=K, replace=False)
            h_init = np.zeros(N)
            h_init[support] = 1.0 / K
            
            h_fw, score_fw = frank_wolfe(h_init)
            
            if score_fw > best_score:
                best_score = score_fw
                best_h = h_fw
    
    # Final refinement of the best found
    if time.time() < deadline - 5:
        h_fw, score_fw = frank_wolfe(best_h, max_iter=10000)
        if score_fw > best_score:
            best_score = score_fw
            best_h = h_fw

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
