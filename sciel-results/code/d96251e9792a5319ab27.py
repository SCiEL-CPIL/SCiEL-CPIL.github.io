import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
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
        for mult in [1, 2, 4, 8, 16]:
            U_s = np.round(U * mult) / mult
            V_s = np.round(V * mult) / mult
            W_s = np.round(W * mult) / mult
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
    
    def newton_polish(U, V, W, max_iter=50, tol=1e-10):
        """Newton's method to polish near-integer solutions."""
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break
            
            # Compute Jacobian-like quantities for Newton step
            # dL/dU = 2 * diff * V * W^T (einsum)
            gradU = 2 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = 2 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = 2 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Simple damped Newton step
            lr = 0.1
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
            lr *= 0.95
        
        return U, V, W
    
    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=400):
        """Faster optimization with adaptive learning rate and early stopping."""
        U = U_init[:, :target_rank].astype(float).copy()
        V = V_init[:, :target_rank].astype(float).copy()
        W = W_init[:, :target_rank].astype(float).copy()
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.02
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
                
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if loss < 1e-14:
                break
                
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            vU = momentum * vU + (1 - momentum) * gradU
            vV = momentum * vV + (1 - momentum) * gradV
            vW = momentum * vW + (1 - momentum) * gradW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            if it % 50 == 49:
                lr *= 0.95
        
        # Newton polish
        best_U, best_V, best_W = newton_polish(best_U, best_V, best_W, max_iter=30)
        
        valid, Us, Vs, Ws = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return Us, Vs, Ws
        return None, None, None
    
    # Build rank-46 baseline
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        rank = 50
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
    
    # Try rank 45: remove one column and optimize
    for c_remove in range(46):
        if time.time() > deadline - 60:
            break
        mask = [i for i in range(46) if i != c_remove]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Small perturbation
        U_test += np.random.randn(*U_test.shape) * 0.005
        V_test += np.random.randn(*V_test.shape) * 0.005
        W_test += np.random.randn(*W_test.shape) * 0.005
        
        U_opt, V_opt, W_opt = fast_optimize(U_test, V_test, W_test, 45, max_iter=300)
        if U_opt is not None:
            best_result = (U_opt, V_opt, W_opt)
            break
    
    # Try rank 44
    if best_result is None and time.time() < deadline - 80:
        # Try all pairs, but limit attempts
        attempts = 0
        for c1 in range(46):
            for c2 in range(c1+1, 46):
                if time.time() > deadline - 60 or attempts >= 20:
                    break
                mask = [i for i in range(46) if i != c1 and i != c2]
                U_test = U46[:, mask].copy()
                V_test = V46[:, mask].copy()
                W_test = W46[:, mask].copy()
                
                U_test += np.random.randn(*U_test.shape) * 0.01
                V_test += np.random.randn(*V_test.shape) * 0.01
                W_test += np.random.randn(*W_test.shape) * 0.01
                
                U_opt, V_opt, W_opt = fast_optimize(U_test, V_test, W_test, 44, max_iter=250)
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
                attempts += 1
            if best_result is not None:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
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
