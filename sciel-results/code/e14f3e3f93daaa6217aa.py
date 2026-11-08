import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4
    
    def try_snap_validate(U, V, W):
        # Try half-integers first (most likely for Strassen-based)
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def newton_polish(U, V, W, max_iter=50, tol=1e-14):
        """High-precision Newton-style polish for near-exact solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break
            
            # Compute gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adaptive step size based on gradient magnitude
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.002 * min(1.0, 1e-6 / grad_norm)
            
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
        
        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.999, l1_weight=0.0):
        """Adam optimizer with adaptive restarts and optional L1 sparsity penalty."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            # L1 penalty to encourage sparsity
            l1_loss = l1_weight * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
            total_loss = loss + l1_loss
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Stagnation detection with restart
            if total_loss > prev_loss * 1.005 and no_improve > 15:
                restart_count += 1
                if restart_count > 5:
                    break
                # Perturb to escape local minimum
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = total_loss
                continue
            else:
                no_improve += 1
                prev_loss = total_loss
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            if l1_weight > 0:
                gradU += l1_weight * np.sign(U)
                gradV += l1_weight * np.sign(V)
                gradW += l1_weight * np.sign(W)
            
            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            current_lr *= lr_decay
            
            # Decay L1 weight as we get closer to solution to allow fine-tuning
            if l1_weight > 0:
                l1_weight = max(0.0, l1_weight * 0.99)
        
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s
        
        # Final polish: try tighter tolerance with Newton
        if best_loss < 1e-3:
            U2, V2, W2 = newton_polish(best_U, best_V, best_W, max_iter=60, tol=1e-14)
            valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
            if valid:
                return U_s, V_s, W_s
        
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    
    np.random.seed(42)
    
    # Score columns by "removability" - columns that share more support are more redundant
    col_nonzeros_U = np.count_nonzero(U46, axis=0)
    col_nonzeros_V = np.count_nonzero(V46, axis=0)
    col_nonzeros_W = np.count_nonzero(W46, axis=0)
    
    # Create support sets for each column
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    # Score: lower = more likely to be removable (smaller support, more overlap with others)
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        # Count how many other columns share significant support
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 3:
                overlap_count += 1
        score = support_size + overlap_count * 0.5
        col_scores.append(score)
    
    col_order = np.argsort(col_scores)
    
    # Strategy 1: Try removing single columns (rank 45)
    max_attempts = min(25, len(col_order))
    for attempt in range(max_attempts):
        if time.time() > deadline - 40:
            break
            
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Try multiple noise scales and seeds
        found = False
        for seed in [42, 123, 999, 777, 555]:
            if time.time() > deadline - 35:
                break
            np.random.seed(seed + attempt * 100)
            for noise_scale in [0.005, 0.02, 0.08, 0.15]:
                if time.time() > deadline - 33:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=150, lr=0.04)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: If rank 45 failed, try removing pairs of columns (rank 44) with sparsity penalty
    if best_result is None and time.time() < deadline - 50:
        # Try pairs of columns with high overlap
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2])
                pair_candidates.append((overlap, c1, c2))
        pair_candidates.sort(reverse=True)
        
        for idx in range(min(30, len(pair_candidates))):
            if time.time() > deadline - 45:
                break
            _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123, 999, 777]:
                if time.time() > deadline - 42:
                    break
                np.random.seed(seed + idx * 10)
                for noise_scale in [0.01, 0.05, 0.12]:
                    if time.time() > deadline - 40:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    # Use sparsity penalty for rank 44 search
                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 44, max_iter=150, lr=0.035, l1_weight=0.1)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    # Strategy 3: Try rank 43 with structured perturbation
    if best_result is None and time.time() < deadline - 55:
        # Remove 3 columns with highest mutual overlap
        triple_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                for k in range(j+1, len(col_order)):
                    c1, c2, c3 = col_order[i], col_order[j], col_order[k]
                    overlap = (len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2]) +
                              len(support_U[c1] & support_U[c3]) + len(support_V[c1] & support_V[c3]) + len(support_W[c1] & support_W[c3]) +
                              len(support_U[c2] & support_U[c3]) + len(support_V[c2] & support_V[c3]) + len(support_W[c2] & support_W[c3]))
                    triple_candidates.append((overlap, c1, c2, c3))
        triple_candidates.sort(reverse=True)
        
        for idx in range(min(10, len(triple_candidates))):
            if time.time() > deadline - 50:
                break
            _, c1, c2, c3 = triple_candidates[idx]
            mask = [i for i in range(46) if i not in (c1, c2, c3)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123]:
                if time.time() > deadline - 48:
                    break
                np.random.seed(seed + idx * 20)
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * 0.05
                V_try += np.random.randn(*V_try.shape) * 0.05
                W_try += np.random.randn(*W_try.shape) * 0.05
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 43, max_iter=100, lr=0.03)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result is not None:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        # Verify the result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
