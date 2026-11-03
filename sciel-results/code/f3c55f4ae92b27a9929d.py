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
    
    def optimize_to_rank(U_init, V_init, W_init, target_rank, max_iter=300, lr=0.05, momentum=0.9):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_loss = float('inf')
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            loss = np.sum(diff * diff)
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    break
            
            if loss < 1e-8:
                break
            
            if it % 50 == 0:
                valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
                if valid:
                    return U_s, V_s, W_s
        
        valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
        if valid:
            return U_s, V_s, W_s
        return None, None, None
    
    # First, verify rank-46 works
    U46, V46, W46 = build_rank46()
    if validate(U46, V46, W46):
        # Try to find rank-45 solution
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + restart)
            
            # Add small perturbation to encourage different solution
            U_pert = U46.copy()
            V_pert = V46.copy()
            W_pert = W46.copy()
            
            # Try zeroing out different columns and re-optimizing
            for col_to_zero in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]:
                if time.time() > deadline - 15:
                    break
                
                U_test = U46[:, :col_to_zero].copy()
                V_test = V46[:, :col_to_zero].copy()
                W_test = W46[:, :col_to_zero].copy()
                
                if U_test.shape[1] < 45:
                    continue
                
                # Add small noise to escape local minima
                U_test += np.random.randn(*U_test.shape) * 0.01
                V_test += np.random.randn(*V_test.shape) * 0.01
                W_test += np.random.randn(*W_test.shape) * 0.01
                
                U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 45, max_iter=200, lr=0.02, momentum=0.9)
                
                if U_opt is not None:
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
