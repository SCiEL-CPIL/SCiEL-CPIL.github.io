# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def evaluate(S, W):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for discrete measure (S, W).
        S: 1-D array of support points (sorted).
        W: 1-D array of weights (sum to 1).
        Uses O(n^4) vectorized computation.
        """
        n = len(S)
        if n == 0:
            return 0.0
        
        # Vectorized computation
        # S_i + S_j + S_k < 2 S_l
        TripleSums = S[:, None, None] + S[None, :, None] + S[None, None, :]
        WeightsTriple = W[:, None, None] * W[None, :, None] * W[None, None, :]
        Thresholds = 2.0 * S[:, None, None, None]
        Indicators = (TripleSums[None, :, :, :] < Thresholds).astype(np.float64)
        P_l = np.tensordot(Indicators, WeightsTriple, axes=([1, 2, 3], [0, 1, 2]))
        P = np.dot(W, P_l)
        return P

    def evaluate_fast_2pt(a, M):
        """
        Fast evaluation for 2-point measure {0, M} with weight a at 0.
        X_i is 0 with prob a, M with prob (1-a).
        P[X1+X2+X3 < 2*X4]
        
        Let p = 1-a (prob of M).
        X4 = 0 (prob a): need X1+X2+X3 < 0 -> impossible (all non-negative)
        X4 = M (prob p): need X1+X2+X3 < 2M
        
        X1+X2+X3 can be: 0, M, 2M, 3M
        Prob(X1+X2+X3 = 0) = a^3
        Prob(X1+X2+X3 = M) = 3*a^2*p
        Prob(X1+X2+X3 = 2M) = 3*a*p^2
        Prob(X1+X2+X3 = 3M) = p^3
        
        Need sum < 2M, so: 0 and M qualify.
        P = p * (a^3 + 3*a^2*p)
        """
        p = 1.0 - a
        return p * (a**3 + 3.0 * a**2 * p)

    def evaluate_fast_3pt(a, b, c, M):
        """
        Fast evaluation for 3-point measure {0, 1, M} with weights {a, b, c}.
        Scale invariance: we can fix the middle point at 1 and vary M.
        X_i in {0, 1, M} with probs {a, b, c}.
        Need P[X1+X2+X3 < 2*X4].
        
        This is more complex, so we use the general evaluator but with precomputed sums.
        """
        S = np.array([0.0, 1.0, M])
        W = np.array([a, b, c])
        return evaluate(S, W)

    # Global best tracking
    global_best_P = 0.0
    global_best_S = np.array([0.0, 1.0])
    global_best_W = np.array([0.5, 0.5])

    # === Phase 1: Analytical 2-point optimization ===
    # For 2-point {0, M} with weight a at 0:
    # P = (1-a) * (a^3 + 3*a^2*(1-a))
    # = (1-a) * a^2 * (a + 3 - 3a)
    # = (1-a) * a^2 * (3 - 2a)
    # Let f(a) = (1-a) * a^2 * (3-2a) = a^2(1-a)(3-2a)
    # Expand: a^2(3 - 2a - 3a + 2a^2) = a^2(3 - 5a + 2a^2) = 3a^2 - 5a^3 + 2a^4
    # f'(a) = 6a - 15a^2 + 8a^3
    # f'(a) = a(6 - 15a + 8a^2)
    # 8a^2 - 15a + 6 = 0
    # a = (15 +- sqrt(225 - 192)) / 16 = (15 +- sqrt(33)) / 16
    # sqrt(33) ≈ 5.7446
    # a1 = (15 - 5.7446)/16 ≈ 0.5785
    # a2 = (15 + 5.7446)/16 ≈ 1.309 (invalid, > 1)
    
    a_opt_2pt = (15.0 - np.sqrt(33.0)) / 16.0
    P_2pt = evaluate_fast_2pt(a_opt_2pt, 1.0)
    
    if P_2pt > global_best_P:
        global_best_P = P_2pt
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([a_opt_2pt, 1.0 - a_opt_2pt])

    # === Phase 2: 3-point analytical search ===
    # Support {0, 1, M}, weights {a, b, c} with a+b+c=1
    # For fixed M and a, we need to optimize over b (c = 1-a-b)
    # P is piecewise polynomial in b for fixed M, a
    
    def P_3pt(M, a, b):
        """Compute P for 3-point {0,1,M} with weights {a,b,1-a-b}"""
        c = 1.0 - a - b
        if c < -1e-12 or b < -1e-12:
            return 0.0
        c = max(c, 0.0)
        b = max(b, 0.0)
        S = np.array([0.0, 1.0, M])
        W = np.array([a, b, c])
        return evaluate(S, W)

    def P_3pt_analytic(M, a, b):
        """
        Analytic computation of P[X1+X2+X3 < 2*X4] for {0,1,M} with weights {a,b,c}.
        X4 values: 0 (prob a), 1 (prob b), M (prob c)
        
        Case X4=0 (prob a): need X1+X2+X3 < 0 -> impossible
        Case X4=1 (prob b): need X1+X2+X3 < 2
        Case X4=M (prob c): need X1+X2+X3 < 2M
        
        X1+X2+X3 values (each Xi in {0,1,M}):
        Possible sums: 0, 1, 2, 3, M, M+1, M+2, M+3, 2M, 2M+1, 2M+2, 3M
        (assuming M > 2, so all these are distinct)
        
        For each sum value, compute its probability, then check if < threshold.
        """
        c = 1.0 - a - b
        if c < 0 or b < 0:
            return 0.0
        
        # Enumerate all (i,j,k) combinations of {0,1,M}
        vals = [0.0, 1.0, M]
        probs = [a, b, c]
        
        # Build all triple sums and their probabilities
        triple_sums = {}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = vals[i] + vals[j] + vals[k]
                    w = probs[i] * probs[j] * probs[k]
                    triple_sums[s] = triple_sums.get(s, 0.0) + w
        
        P = 0.0
        # Case X4 = 1: threshold = 2
        for s, w in triple_sums.items():
            if s < 2.0:
                P += b * w
        # Case X4 = M: threshold = 2M
        for s, w in triple_sums.items():
            if s < 2.0 * M:
                P += c * w
        
        return P

    # Grid search over M and a, then optimize b
    if time.time() < deadline - 60.0:
        # Coarse grid
        M_vals = np.logspace(0.3, 1.5, 40)  # M from ~2 to ~30
        a_vals = np.linspace(0.0, 1.0, 30)
        
        best_3P = global_best_P
        best_3M = 1.0
        best_3a = a_opt_2pt
        best_3b = 1.0 - a_opt_2pt
        
        for M in M_vals:
            if time.time() > deadline - 30.0:
                break
            for a in a_vals:
                if time.time() > deadline - 30.0:
                    break
                # Optimize b via ternary search or fine grid
                # P as function of b is a polynomial, so we can find max analytically
                # or use a fine grid
                b_grid = np.linspace(0.0, 1.0 - a, 20)
                best_b = 0.0
                best_bP = 0.0
                for bb in b_grid:
                    P_val = P_3pt_analytic(M, a, bb)
                    if P_val > best_bP:
                        best_bP = P_val
                        best_b = bb
                
                if best_bP > best_3P:
                    best_3P = best_bP
                    best_3M = M
                    best_3a = a
                    best_3b = best_b
        
        if best_3P > global_best_P:
            global_best_P = best_3P
            global_best_S = np.array([0.0, 1.0, best_3M])
            global_best_W = np.array([best_3a, best_3b, 1.0 - best_3a - best_3b])

    # === Phase 3: Refine 3-point with finer grid around best ===
    if time.time() < deadline - 30.0:
        # Refine around the best 3-point solution
        S_ref = global_best_S.copy()
        W_ref = global_best_W.copy()
        n_ref = len(S_ref)
        
        if n_ref == 3:
            M_best = S_ref[2]
            a_best = W_ref[0]
            b_best = W_ref[1]
            
            # Fine grid refinement
            M_fine = np.linspace(M_best * 0.9, M_best * 1.1, 30)
            a_fine = np.linspace(max(0, a_best - 0.05), min(1, a_best + 0.05), 20)
            
            best_ref_P = global_best_P
            best_ref_M = M_best
            best_ref_a = a_best
            best_ref_b = b_best
            
            for M in M_fine:
                if time.time() > deadline - 15.0:
                    break
                for a in a_fine:
                    if time.time() > deadline - 15.0:
                        break
                    b_grid = np.linspace(0.0, max(0, 1.0 - a), 30)
                    for bb in b_grid:
                        P_val = P_3pt_analytic(M, a, bb)
                        if P_val > best_ref_P:
                            best_ref_P = P_val
                            best_ref_M = M
                            best_ref_a = a
                            best_ref_b = bb
            
            if best_ref_P > global_best_P:
                global_best_P = best_ref_P
                global_best_S = np.array([0.0, 1.0, best_ref_M])
                global_best_W = np.array([best_ref_a, best_ref_b, 1.0 - best_ref_a - best_ref_b])

    # === Phase 4: Local search on best solution ===
    def optimize_local(S_init, W_init, max_iters=500):
        S = np.sort(S_init.copy())
        W = W_init.copy()
        W = np.clip(W, 0.0, None)
        W_sum = np.sum(W)
        if W_sum == 0:
            W = np.ones_like(W)
        else:
            W /= W_sum
            
        if len(S) > 1:
            S = S / S[-1]
            S[-1] = 1.0
            
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        step_s = 0.05
        step_w = 0.05
        
        for i in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            S_new = S.copy()
            W_new = W.copy()
            
            if n > 1:
                noise_s = np.random.randn(n - 1) * step_s
                S_new[0:n-1] += noise_s
                S_new[0:n-1] = np.clip(S_new[0:n-1], 0.0, 1.0)
                S_new[0:n-1] = np.sort(S_new[0:n-1])
            
            noise_w = np.random.randn(n) * step_w
            W_new += noise_w
            W_new = np.clip(W_new, 0.0, None)
            W_sum = np.sum(W_new)
            if W_sum < 1e-9:
                continue
            W_new /= W_sum
            
            P_new = evaluate(S_new, W_new)
            
            if P_new > current_P:
                current_P = P_new
                S = S_new
                W = W_new
                if P_new > best_P:
                    best_P = P_new
                    best_S = S_new.copy()
                    best_W = W_new.copy()
            
            step_s *= 0.99
            step_w *= 0.99
            
        return best_P, best_S, best_W

    if time.time() < deadline - 20.0:
        P_loc, S_loc, W_loc = optimize_local(global_best_S, global_best_W, max_iters=300)
        if P_loc > global_best_P:
            global_best_P = P_loc
            global_best_S = S_loc
            global_best_W = W_loc

    # === Phase 5: Try 4-point measures ===
    if time.time() < deadline - 15.0:
        # Try a few 4-point configurations
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            
            if trial == 0:
                S_init = np.array([0.0, 0.5, 1.0, 2.0])
                W_init = np.array([0.3, 0.2, 0.2, 0.3])
            elif trial == 1:
                S_init = np.array([0.0, 0.3, 0.7, 1.0])
                W_init = np.array([0.35, 0.2, 0.2, 0.25])
            elif trial == 2:
                S_init = np.array([0.0, 0.4, 0.8, 1.0])
                W_init = np.array([0.3, 0.25, 0.2, 0.25])
            elif trial == 3:
                S_init = np.array([0.0, 0.5, 1.0, 1.5])
                W_init = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                S_init = np.sort(np.random.uniform(0, 1, 3))
                S_init = np.append(S_init, 1.0)
                W_init = np.random.dirichlet(np.ones(4))
            
            P_4, S_4, W_4 = optimize_local(S_init, W_init, max_iters=200)
            if P_4 > global_best_P:
                global_best_P = P_4
                global_best_S = S_4
                global_best_W = W_4

    # === Final validity check and normalization ===
    global_best_S = np.sort(global_best_S)
    global_best_W = np.clip(global_best_W, 0.0, None)
    w_sum = np.sum(global_best_W)
    if w_sum < 1e-9:
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([0.5, 0.5])
    else:
        global_best_W /= w_sum
        global_best_S = np.clip(global_best_S, 0.0, None)
        
    # Ensure no NaN or inf
    if not np.all(np.isfinite(global_best_S)) or not np.all(np.isfinite(global_best_W)):
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([a_opt_2pt, 1.0 - a_opt_2pt])
    
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
