# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def compute_P(S, W):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. X's from discrete measure (S, W).
        S: 1-D array of support points (sorted).
        W: 1-D array of weights (sum to 1).
        Uses O(n^4) vectorized approach for small n.
        """
        n = len(S)
        if n == 0:
            return 0.0
        
        if n <= 15:
            # O(n^4) vectorized approach
            TripleSums = S[:, None, None] + S[None, :, None] + S[None, None, :]
            WeightsTriple = W[:, None, None] * W[None, :, None] * W[None, None, :]
            Thresholds = 2.0 * S[:, None, None, None]
            Indicators = (TripleSums[None, :, :, :] < Thresholds).astype(np.float64)
            P_l = np.tensordot(Indicators, WeightsTriple, axes=([1, 2, 3], [0, 1, 2]))
            P = np.dot(W, P_l)
            return P
        else:
            # O(n^3) approach
            P_total = 0.0
            for l in range(n):
                t_l = 2.0 * S[l]
                C_l = 0.0
                for i in range(n):
                    for j in range(n):
                        x = t_l - S[i] - S[j]
                        if x <= 0:
                            continue
                        idx = np.searchsorted(S, x, side='left')
                        if idx > 0:
                            F_x = np.sum(W[:idx])
                            C_l += W[i] * W[j] * F_x
                P_total += W[l] * C_l
            return P_total

    def P_3point(a, b, c):
        """
        Compute P for 3-point distribution with support {0, 1, M} where M = a,
        weights {b, c, 1-b-c}.
        """
        S = np.array([0.0, 1.0, a])
        S = np.sort(S)
        W = np.array([b, c, 1.0 - b - c])
        # Ensure weights are non-negative
        if W.min() < -1e-12:
            return 0.0
        W = np.clip(W, 0, None)
        W_sum = W.sum()
        if W_sum < 1e-12:
            return 0.0
        W /= W_sum
        return compute_P(S, W)

    # Analytic 2-point solution
    p2 = (15.0 - np.sqrt(33.0)) / 16.0
    S2 = np.array([0.0, 1.0])
    W2 = np.array([p2, 1.0 - p2])
    P2 = compute_P(S2, W2)
    
    global_best_P = P2
    global_best_S = S2.copy()
    global_best_W = W2.copy()
    
    # 3-point grid search over (a, b) where a = M (support point), b = weight at 0
    # c = weight at 1, and weight at M is 1-b-c
    # We need b >= 0, c >= 0, 1-b-c >= 0, so b + c <= 1
    # a > 0 (and typically a > 1 for interesting behavior, or 0 < a < 1)
    
    # Let's search over a in (0, 3) and b in [0, 1]
    # For each (a, b), we can optimize over c in [0, 1-b]
    
    # Coarse grid first
    a_vals = np.linspace(0.1, 3.0, 60)
    b_vals = np.linspace(0.01, 0.95, 50)
    
    best_P_3 = global_best_P
    best_a = 1.0
    best_b = p2
    best_c = 1.0 - p2
    
    for a in a_vals:
        if time.time() > deadline - 30.0:
            break
        for b in b_vals:
            if time.time() > deadline - 10.0:
                break
            # For fixed a, b, optimize over c in [0, 1-b]
            # P is a function of c. Let's sample c.
            c_max = 1.0 - b
            if c_max <= 0:
                continue
            # Sample c values
            c_samples = np.linspace(0.0, c_max, 20)
            for c in c_samples:
                P_val = P_3point(a, b, c)
                if P_val > best_P_3:
                    best_P_3 = P_val
                    best_a = a
                    best_b = b
                    best_c = c
    
    if best_P_3 > global_best_P:
        global_best_P = best_P_3
        S3 = np.array([0.0, 1.0, best_a])
        S3 = np.sort(S3)
        W3 = np.array([best_b, best_c, 1.0 - best_b - best_c])
        W3 = np.clip(W3, 0, None)
        W3 /= W3.sum()
        global_best_S = S3
        global_best_W = W3
    
    # Fine grid around the best 3-point solution
    if time.time() < deadline - 20.0:
        a_lo = max(0.01, best_a - 0.5)
        a_hi = min(3.5, best_a + 0.5)
        b_lo = max(0.0, best_b - 0.2)
        b_hi = min(1.0, best_b + 0.2)
        
        a_fine = np.linspace(a_lo, a_hi, 40)
        b_fine = np.linspace(b_lo, b_hi, 30)
        
        for a in a_fine:
            if time.time() > deadline - 10.0:
                break
            for b in b_fine:
                if time.time() > deadline - 5.0:
                    break
                c_max = 1.0 - b
                if c_max <= 0:
                    continue
                c_samples = np.linspace(0.0, c_max, 25)
                for c in c_samples:
                    P_val = P_3point(a, b, c)
                    if P_val > global_best_P:
                        global_best_P = P_val
                        S3 = np.array([0.0, 1.0, a])
                        S3 = np.sort(S3)
                        W3 = np.array([b, c, 1.0 - b - c])
                        W3 = np.clip(W3, 0, None)
                        W3 /= W3.sum()
                        global_best_S = S3
                        global_best_W = W3
    
    # Local optimization from best 3-point solution
    if time.time() < deadline - 15.0:
        S = global_best_S.copy()
        W = global_best_W.copy()
        n = len(S)
        
        step_s = 0.01
        step_w = 0.01
        
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
            
            S_new = S.copy()
            W_new = W.copy()
            
            if n > 1:
                noise_s = np.random.randn(n - 1) * step_s
                S_new[0:n-1] += noise_s
                S_new[0:n-1] = np.clip(S_new[0:n-1], 0.0, 3.0)
                S_new[0:n-1] = np.sort(S_new[0:n-1])
            noise_w = np.random.randn(n) * step_w
            W_new += noise_w
            W_new = np.clip(W_new, 0.0, None)
            W_sum = np.sum(W_new)
            if W_sum < 1e-9:
                break
            W_new /= W_sum
            
            P_new = compute_P(S_new, W_new)
            
            if P_new > global_best_P:
                global_best_P = P_new
                S = S_new
                W = W_new
                global_best_S = S.copy()
                global_best_W = W.copy()
            else:
                step_s *= 0.95
                step_w *= 0.95
            
            if step_s < 1e-5:
                break
    
    # Try 4-point distributions
    if time.time() < deadline - 30.0:
        # 4-point: {0, 1, a, b} with a < b
        # Weights: w0, w1, wa, wb
        # Coarse search
        a_vals4 = np.linspace(0.1, 2.0, 15)
        b_vals4 = np.linspace(1.5, 3.0, 15)
        
        for a in a_vals4:
            if time.time() > deadline - 10.0:
                break
            for b in b_vals4:
                if b <= a:
                    continue
                # Random weights
                for _ in range(5):
                    W4 = np.random.dirichlet(np.ones(4))
                    S4 = np.array([0.0, 1.0, a, b])
                    S4 = np.sort(S4)
                    # Reorder weights to match sorted support
                    # Original order: 0, 1, a, b
                    # If a < 1, sort order changes
                    order = np.argsort(S4)
                    S4_sorted = S4[order]
                    W4_sorted = W4[order]
                    P4 = compute_P(S4_sorted, W4_sorted)
                    if P4 > global_best_P:
                        global_best_P = P4
                        global_best_S = S4_sorted
                        global_best_W = W4_sorted
    
    # Final validity check and normalization
    global_best_S = np.sort(global_best_S)
    global_best_W = np.clip(global_best_W, 0.0, None)
    w_sum = np.sum(global_best_W)
    if w_sum < 1e-9:
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([p2, 1.0 - p2])
    else:
        global_best_W /= w_sum
        global_best_S = np.clip(global_best_S, 0.0, None)
        
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
