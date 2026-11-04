import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor T indices
    # T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    # Convert to numpy arrays for fast indexing
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    
    # Precompute the full tensor for verification
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for a, b, w in indices:
        T[a, b, w] = 1.0

    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        
        # Check that all target entries are 1
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        # Check that non-target entries are 0 (approximately)
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9

    def get_fallback():
        """Rank 42 solution using Strassen blocks"""
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            # Strassen rank 7 for 2x2x2
            # M0 = (A00+A11)(B00+B11)
            # M1 = (A10+A11)B00
            # M2 = A00(B10-B11)
            # M3 = A11(B01-B10)
            # M4 = (A00+A01)B11
            # M5 = (A10-A00)(B00+B01)
            # M6 = (A01-A11)(B10+B11)
            
            # a_idx: [A00, A01, A10, A11]
            # b_idx: [B00, B01, B10, B11]
            # w_idx: [C00, C01, C10, C11]
            
            # M0
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[3]][start_col + 0] += 1

            # M1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            V[b_idx[0]][start_col + 1] += 1
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[3]][start_col + 1] -= 1

            # M2
            U[a_idx[0]][start_col + 2] += 1
            V[b_idx[2]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[3]][start_col + 2] += 1

            # M3
            U[a_idx[3]][start_col + 3] += 1
            V[b_idx[1]][start_col + 3] += 1
            V[b_idx[2]][start_col + 3] -= 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[2]][start_col + 3] += 1

            # M4
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            V[b_idx[3]][start_col + 4] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[1]][start_col + 4] += 1

            # M5
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            W[w_idx[3]][start_col + 5] += 1

            # M6
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            W[w_idx[0]][start_col + 6] += 1

        # We need to decompose 2x4 * 4x6.
        # Split A (2x4) into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3)
        # Split B (4x6) into three 2x2 blocks? No, 6 is not divisible by 2.
        # The current fallback uses a specific decomposition.
        # Let's stick to the known working fallback from the prompt.
        
        a_L = [0, 1, 4, 5] # A00, A01, A10, A11 for left block
        a_R = [2, 3, 6, 7] # A00, A01, A10, A11 for right block
        
        # Block 1: k=0 (b cols 0-5), i=0 (w rows 0,3,6,9) -> w indices 0, 2, 1, 3
        # Wait, the indices in the prompt's fallback were specific.
        # Let's just copy the working fallback from the prompt to ensure correctness.
        
        def add_strassen_block_orig(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block_orig(a_L, b_1_T, w_1, 0)
        add_strassen_block_orig(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block_orig(a_L, b_2_T, w_2, 14)
        add_strassen_block_orig(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block_orig(a_L, b_3_T, w_3, 28)
        add_strassen_block_orig(a_R, b_3_B, w_3, 35)
        
        return U, V, W

    U_fb, V_fb, W_fb = get_fallback()
    
    # Verify fallback
    if not verify_solution(U_fb, V_fb, W_fb):
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Attempt to improve using WGA / Kronecker structure
    # Strategy: 4 = 2x2, 6 = 2x3.
    # We can try to find a rank 28 solution (7 * 4) or similar.
    # However, the tensor structure might not allow a simple Kronecker product 
    # due to the specific index mapping w = k*n + i.
    # Let's try a numerical search for rank 28-41 with a time limit.
    
    start_time = time.time()
    deadline = 110.0
    best_rank = 42
    best_solution = (U_fb, V_fb, W_fb)
    
    # We will try to optimize for ranks 28 to 41.
    # Using scipy.optimize.least_squares
    
    try:
        import scipy.optimize as so
    except ImportError:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Precompute indices for fast residual calculation
    # The residual is only defined on the 48 non-zero entries.
    # hat_T[a,b,w] = sum_r U[a,r] V[b,r] W[w,r]
    # We want hat_T[idx_a, idx_b, idx_w] = 1
    
    for r in range(28, 42):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(r * 100 + attempt)
            U0 = np.random.randn(dim_A, r) * 0.1
            V0 = np.random.randn(dim_B, r) * 0.1
            W0 = np.random.randn(dim_C, r) * 0.1
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def res_func(t):
                U = t[:dim_A * r].reshape(dim_A, r)
                V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
                hat_T = np.einsum('ar,br,wr->abw', U, V, W)
                vals = hat_T[idx_a, idx_b, idx_w]
                return vals - 1.0

            try:
                res = so.least_squares(res_func, theta0, method='lm', xtol=1e-12, ftol=1e-12)
                if res.success and res.cost < 1e-10:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    if verify_solution(U_int, V_int, W_int):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    else:
                        # If rounding failed, try to keep the float solution 
                        # if it's very close, but the task requires integer/half-integer.
                        # So we must discard if rounding fails.
                        pass
            except Exception:
                pass

    return {"U": U_fb, "V": V_fb, "W": W_fb}
