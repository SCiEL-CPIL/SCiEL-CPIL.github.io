import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
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
    
    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
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
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W
    
    def optimize_to_rank(U_init, V_init, W_init, target_rank, max_iter=500, lr=0.008, beta1=0.9, beta2=0.999, eps=1e-8):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        # Adam optimizer state
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
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam update
            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    break
            
            if loss < 1e-10:
                break
            
            if it % 100 == 0:
                valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
                if valid:
                    return U_s, V_s, W_s
        
        # Final polish with tighter tolerance
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s
        
        # Also try snapping the current U, V, W
        valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
        if valid:
            return U_s, V_s, W_s
        
        return None, None, None
    
    def polish_solution(U, V, W, max_iter=200, lr=0.005):
        """Final high-precision polish pass"""
        R = U.shape[1]
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()
        
        mU = np.zeros_like(U_p)
        vU = np.zeros_like(U_p)
        mV = np.zeros_like(V_p)
        vV = np.zeros_like(V_p)
        mW = np.zeros_like(W_p)
        vW = np.zeros_like(W_p)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prod = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V_p, W_p, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U_p, W_p, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U_p, V_p, optimize=True)
            
            t = it + 1
            mU = 0.9 * mU + 0.1 * gradU
            vU = 0.999 * vU + 0.001 * (gradU ** 2)
            mU_hat = mU / (1 - 0.9 ** t)
            vU_hat = vU / (1 - 0.999 ** t)
            U_p -= lr * mU_hat / (np.sqrt(vU_hat) + 1e-8)
            
            mV = 0.9 * mV + 0.1 * gradV
            vV = 0.999 * vV + 0.001 * (gradV ** 2)
            mV_hat = mV / (1 - 0.9 ** t)
            vV_hat = vV / (1 - 0.999 ** t)
            V_p -= lr * mV_hat / (np.sqrt(vV_hat) + 1e-8)
            
            mW = 0.9 * mW + 0.1 * gradW
            vW = 0.999 * vW + 0.001 * (gradW ** 2)
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W_p -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            
            loss = np.sum(diff * diff)
            if loss < 1e-12:
                break
        
        valid, U_s, V_s, W_s = try_snap_validate(U_p, V_p, W_p)
        if valid:
            return U_s, V_s, W_s
        return None, None, None
    
    # First, verify rank-46 works
    U46, V46, W46 = build_rank46()
    if validate(U46, V46, W46):
        # Try to find rank-45 solution
        best_result = None
        
        for restart in range(8):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(100 + restart)
            
            # Try different strategies for reducing rank
            strategies = [
                'zero_single',
                'zero_multiple',
                'perturb_all',
            ]
            
            for strategy in strategies:
                if time.time() > deadline - 20:
                    break
                
                if strategy == 'zero_single':
                    # Try zeroing different single columns
                    for col_to_zero in range(46):
                        if time.time() > deadline - 18:
                            break
                        
                        mask = [i for i in range(46) if i != col_to_zero]
                        U_test = U46[:, mask].copy()
                        V_test = V46[:, mask].copy()
                        W_test = W46[:, mask].copy()
                        
                        # Add small noise
                        noise_scale = 0.005
                        U_test += np.random.randn(*U_test.shape) * noise_scale
                        V_test += np.random.randn(*V_test.shape) * noise_scale
                        W_test += np.random.randn(*W_test.shape) * noise_scale
                        
                        U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 45, max_iter=300, lr=0.008)
                        
                        if U_opt is not None:
                            # Polish
                            U_pol, V_pol, W_pol = polish_solution(U_opt, V_opt, W_opt, max_iter=100, lr=0.003)
                            if U_pol is not None:
                                return {
                                    "U": [[float(x) for x in row] for row in U_pol],
                                    "V": [[float(x) for x in row] for row in V_pol],
                                    "W": [[float(x) for x in row] for row in W_pol]
                                }
                            else:
                                return {
                                    "U": [[float(x) for x in row] for row in U_opt],
                                    "V": [[float(x) for x in row] for row in V_opt],
                                    "W": [[float(x) for x in row] for row in W_opt]
                                }
                
                elif strategy == 'zero_multiple':
                    # Try zeroing 2 columns and re-optimizing
                    import itertools
                    col_pairs = list(itertools.combinations(range(46), 2))
                    # Limit number of pairs to try
                    np.random.shuffle(col_pairs)
                    for pair in col_pairs[:20]:
                        if time.time() > deadline - 18:
                            break
                        
                        mask = [i for i in range(46) if i not in pair]
                        U_test = U46[:, mask].copy()
                        V_test = V46[:, mask].copy()
                        W_test = W46[:, mask].copy()
                        
                        noise_scale = 0.01
                        U_test += np.random.randn(*U_test.shape) * noise_scale
                        V_test += np.random.randn(*V_test.shape) * noise_scale
                        W_test += np.random.randn(*W_test.shape) * noise_scale
                        
                        U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 44, max_iter=300, lr=0.008)
                        
                        if U_opt is not None:
                            U_pol, V_pol, W_pol = polish_solution(U_opt, V_opt, W_opt, max_iter=100, lr=0.003)
                            if U_pol is not None:
                                return {
                                    "U": [[float(x) for x in row] for row in U_pol],
                                    "V": [[float(x) for x in row] for row in V_pol],
                                    "W": [[float(x) for x in row] for row in W_pol]
                                }
                            else:
                                return {
                                    "U": [[float(x) for x in row] for row in U_opt],
                                    "V": [[float(x) for x in row] for row in V_opt],
                                    "W": [[float(x) for x in row] for row in W_opt]
                                }
                
                elif strategy == 'perturb_all':
                    # Perturb all columns and try to optimize to rank 45
                    U_test = U46.copy()
                    V_test = V46.copy()
                    W_test = W46.copy()
                    
                    noise_scale = 0.02
                    U_test += np.random.randn(*U_test.shape) * noise_scale
                    V_test += np.random.randn(*V_test.shape) * noise_scale
                    W_test += np.random.randn(*W_test.shape) * noise_scale
                    
                    # Select 45 columns (drop the one with smallest norm)
                    col_norms = np.linalg.norm(U_test, axis=0)
                    drop_col = np.argmin(col_norms)
                    mask = [i for i in range(46) if i != drop_col]
                    
                    U_test = U_test[:, mask].copy()
                    V_test = V_test[:, mask].copy()
                    W_test = W_test[:, mask].copy()
                    
                    U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 45, max_iter=400, lr=0.008)
                    
                    if U_opt is not None:
                        U_pol, V_pol, W_pol = polish_solution(U_opt, V_opt, W_opt, max_iter=150, lr=0.003)
                        if U_pol is not None:
                            return {
                                "U": [[float(x) for x in row] for row in U_pol],
                                "V": [[float(x) for x in row] for row in V_pol],
                                "W": [[float(x) for x in row] for row in W_pol]
                            }
                        else:
                            return {
                                "U": [[float(x) for x in row] for row in U_opt],
                                "V": [[float(x) for x in row] for row in V_opt],
                                "W": [[float(x) for x in row] for row in W_opt]
                            }
        
        # Return rank-46 solution
        return {
            "U": [[float(x) for x in row] for row in U46],
            "V": [[float(x) for x in row] for row in V46],
            "W": [[float(x) for x in row] for row in W46]
        }
    else:
        # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
