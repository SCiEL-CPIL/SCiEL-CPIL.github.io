import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # Target rank: 56 (7 * 8)
    # Fallback: 96 (schoolbook)
    
    # Step 1: Find decomposition for 2x2x2 (Strassen, Rank 7)
    # U (4, 7), V (4, 7), W (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U222 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0]
    ])
    V222 = np.array([
        [0, 0, 0, 0, 0, 0, 0], # j=0, k=0 -> V[0, r]
        [0, 1, 0, 0, 0, 0, 0], # j=0, k=1 -> V[1, r]
        [0, 0, 0, 0, 1, 0, 0], # j=1, k=0 -> V[2, r]
        [0, 0, 0, 0, 0, 0, 0]  # j=1, k=1 -> V[3, r]
    ])
    # Wait, standard Strassen matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Let's use a known integer Strassen decomposition.
    # U (A factors), V (B factors), W (C factors)
    # A is 2x2. Indices 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # B is 2x2. Indices 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # C is 2x2. Indices 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    
    U_222 = np.array([
        [1, 1, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 1, 0, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 0, 0, 0, 1]  # A11
    ])
    V_222 = np.array([
        [1, 0, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1], # B01
        [0, 1, 0, 1, 0, 0, 0], # B10
        [1, 1, -1, -1, 1, 0, 0] # B11
    ])
    W_222 = np.array([
        [0, 0, 0, 0, 0, 0, 0], # C00 ? No, W[k*n+i]
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        # Indices: 0:00, 1:01, 2:10, 3:11
        # W[0] (C00)
        [1, 0, 0, -1, 0, 1, -1],
        # W[1] (C01)
        [0, 0, 1, 0, 1, 0, 0],
        # W[2] (C10)
        [0, 1, 0, 1, 0, 0, 0],
        # W[3] (C11)
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    # Verify 2x2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # Check sum_r ... == delta
    
    # Step 2: Find decomposition for 2x2x3 (Rank 8)
    # A: 2x2 (4 entries)
    # B: 2x3 (6 entries)
    # C: 2x3 (6 entries)
    # U (4, 8), V (6, 8), W (6, 8)
    
    # Use optimization to find U, V, W
    def find_223_decomposition():
        # Random restarts
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(100):
            if time.time() > deadline - 10: break
            
            U = np.random.randn(4, 8)
            V = np.random.randn(6, 8)
            W = np.random.randn(6, 8)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Define objective function
            def objective(params):
                U_flat, V_flat, W_flat = params[:32], params[32:80], params[80:128]
                U_t = U_flat.reshape(4, 8)
                V_t = V_flat.reshape(6, 8)
                W_t = W_flat.reshape(6, 8)
                
                # Compute tensor T[i,j,k] = sum_r U[i*2+j, r] V[j*3+k, r] W[k*2+i, r]
                # i in 0..1, j in 0..1, k in 0..2
                errors = []
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            # Sum over r
                            u_idx = i * 2 + j
                            v_idx = j * 3 + k
                            w_idx = k * 2 + i
                            val = np.dot(U_t[u_idx, :], V_t[v_idx, :] * W_t[w_idx, :])
                            errors.append(val - 1.0)
                
                return np.sum(np.square(errors))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            from scipy.optimize import minimize
            try:
                res = minimize(objective, params, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-8})
                if res.fun < best_error:
                    best_error = res.fun
                    if best_error < 1e-8:
                        break
                    best_U = res.x[:32].reshape(4, 8)
                    best_V = res.x[32:80].reshape(6, 8)
                    best_W = res.x[80:128].reshape(6, 8)
            except Exception:
                pass
        
        if best_U is None:
            return None
            
        # Check if we can snap to half-integers
        # Multiply by 2 to check integers
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        
        # Verify snapped solution
        def check_solution(U_t, V_t, W_t):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        u_idx = i * 2 + j
                        v_idx = j * 3 + k
                        w_idx = k * 2 + i
                        val = np.dot(U_t[u_idx, :], V_t[v_idx, :] * W_t[w_idx, :])
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
            
        if check_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
        elif check_solution(best_U, best_V, best_W):
            return best_U, best_V, best_W
        else:
            return None

    res_223 = find_223_decomposition()
    
    if res_223 is not None:
        U223, V223, W223 = res_223
        # Combine using Kronecker product
        # U_total[i1*4+i2, j1*4+j2, r1*8+r2] = U222[i1*2+j1, r1] * U223[i2*2+j2, r2]
        # Wait, indices.
        # A is 4x4. Partitioned into 2x2 blocks of 2x2 matrices.
        # A[i, j] where i = i1*2 + i2, j = j1*2 + j2.
        # U222 acts on (i1, j1). U223 acts on (i2, j2).
        # So U_total[(i1*2+i2)*4 + (j1*2+j2), r1*8+r2] = U222[i1*2+j1, r1] * U223[i2*2+j2, r2]
        
        rank_total = 7 * 8
        U_total = np.zeros((16, rank_total))
        V_total = np.zeros((24, rank_total)) # B is 4x6. m=4, p=6.
        # B[j, k] where j = j1*2 + j2, k = k1*3 + k2?
        # No, B is 4x6.
        # In 2x2x3, B is 2x3.
        # So B[j, k] with j in 0..1, k in 0..2.
        # In 4x6, j in 0..3, k in 0..5.
        # j = j1*2 + j2. k = k1*3 + k2.
        # V222 acts on (j1, k1). V223 acts on (j2, k2).
        # V_total[(j1*2+j2)*6 + (k1*3+k2), r1*8+r2] = V222[j1*2+k1, r1] * V223[j2*3+k2, r2]
        
        # C is 4x6. C[i, k].
        # C[i, k] with i = i1*2 + i2, k = k1*3 + k2.
        # W222 acts on (k1, i1). W223 acts on (k2, i2).
        # W_total[(k1*3+k2)*4 + (i1*2+i2), r1*8+r2] = W222[k1*2+i1, r1] * W223[k2*2+i2, r2]
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for r1 in range(7):
                            for r2 in range(8):
                                r = r1 * 8 + r2
                                u_val = U222[i1*2+j1, r1] * U223[i2*2+j2, r2]
                                U_total[(i1*2+i2)*4 + (j1*2+j2), r] = u_val
                                
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(3):
                        for r1 in range(7):
                            for r2 in range(8):
                                r = r1 * 8 + r2
                                v_val = V222[j1*2+k1, r1] * V223[j2*3+k2, r2]
                                V_total[(j1*2+j2)*6 + (k1*3+k2), r] = v_val
                                
        for k1 in range(2):
            for k2 in range(3):
                for i1 in range(2):
                    for i2 in range(2):
                        for r1 in range(7):
                            for r2 in range(8):
                                r = r1 * 8 + r2
                                w_val = W222[k1*2+i1, r1] * W223[k2*2+i2, r2]
                                W_total[(k1*3+k2)*4 + (i1*2+i2), r] = w_val
                                
        # Convert to lists
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
