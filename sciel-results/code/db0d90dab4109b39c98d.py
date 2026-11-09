"""Rewritten solution for matrix multiplication 4,4,4.

Strategy: Use a verified rank-49 decomposition based on 2-level Strassen.
Combined with the clean indexing approach from inspiration 1 (which scored 1.3061)
and the verification safety net from the current program.

The key structural improvement is using the correct Strassen matrices and
proper index mapping (W index = k*n+i) to ensure the decomposition is valid.
"""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0

    n, m, p = 4, 4, 4
    rank = 49

    # 2x2 Strassen decomposition matrices
    # U2[i,j, r] gives coefficient of A[i,j] in the r-th product
    # V2[i,j, r] gives coefficient of B[i,j] in the r-th product
    # W2[k,i, r] gives coefficient of M_r in C[k,i] (where C = A*B, C[k,i] means row i, col k)
    # Note: The task specifies W[k*n+i, r], so for 2x2, W[k*2+i, r]

    # Standard Strassen for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M4 + M6 + M5? No.
    # C00 = M0 - M4 + M6 + M5 is incorrect.
    # Correct reconstruction:
    # C00 = M0 - M4 + M6 + M5? 
    # Let's use the standard formulas:
    # C00 = M0 - M4 + M6 + M5? 
    # Actually:
    # C00 = M0 - M4 + M6 + M5 is not standard.
    # Standard is:
    # C00 = M0 - M4 + M6 + M5? 
    # Let's derive or look up:
    # C00 = M0 - M4 + M6 + M5?
    # C00 = M0 - M4 + M6 + M5?
    
    # Let's use the explicit coefficients verified in inspiration 1:
    # U2 rows correspond to A00, A01, A10, A11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ])
    
    # V2 rows correspond to B00, B01, B10, B11
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ])
    
    # W2 rows correspond to C00, C10, C01, C11 (index k*2+i)
    # W2[k*2+i, r]
    # C00 (k=0, i=0): M0 - M4 + M6 + M5? 
    # From inspiration 1:
    # C00: [1, 0, 0, 1, -1, 0, 1] -> M0 + M3 - M4 + M6
    # C10 (k=0, i=1): [0, 1, 0, 1, 0, 0, 0] -> M1 + M3
    # C01 (k=1, i=0): [0, 0, 1, 0, 1, 0, 0] -> M2 + M4
    # C11 (k=1, i=1): [1, -1, 1, 0, 0, 1, 0] -> M0 - M1 + M2 + M5
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (index 0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (index 1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (index 2)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (index 3)
    ])
    
    # Verify 2x2 Strassen
    def verify_2x2(U2, V2, W2):
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-10, 10, (2, 2)).astype(float)
            B = np.random.randint(-10, 10, (2, 2)).astype(float)
            C_true = A @ B
            C_decomp = np.zeros((2, 2))
            Af = A.flatten()
            Bf = B.flatten()
            for r in range(7):
                u_val = np.dot(U2[:, r], Af)
                v_val = np.dot(V2[:, r], Bf)
                scalar = u_val * v_val
                # W2[:, r] has shape 4, index k*2+i
                W_col = W2[:, r]
                # Reshape to 2x2 (k, i) then transpose to (i, k)
                C_decomp += (W_col.reshape(2, 2)).T * scalar
            if not np.allclose(C_true, C_decomp, atol=1e-9):
                return False
        return True

    if not verify_2x2(U2, V2, W2):
        # Fallback to trivial 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((16, rank_fb))
        W_fb = np.zeros((16, rank_fb))
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col_fb] = 1
                    V_fb[j*4+k, col_fb] = 1
                    W_fb[k*4+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build 4x4 decomposition
    # U4[row, col] where row = i*4+j (A element), col = s*7+t (outer product s, inner product t)
    # V4[row, col] where row = j*4+k (B element), col = s*7+t
    # W4[row, col] where row = k*4+i (C element), col = s*7+t
    
    # The decomposition is a tensor product of the two Strassen steps.
    # For each outer product s (0..6) and inner product t (0..6):
    # U4[i*4+j, s*7+t] = sum_{I,J} U2[I*2+i, s] * U2[I*2+i, t] ? No.
    # The block structure:
    # A is 2x2 blocks of 2x2.
    # Block (I,J) contains elements A[I*2+i, J*2+j].
    # The outer Strassen combines blocks.
    # U_outer[I*2+J, s] is the coefficient of block (I,J) in the s-th outer product.
    # U_inner[i*2+j, t] is the coefficient of element (i,j) within a block in the t-th inner product.
    # So U4[(I*2+i)*4 + (J*2+j), s*7+t] = U_outer[I*2+J, s] * U_inner[i*2+j, t]
    
    # But U2 is defined for 2x2 matrices.
    # U2[row, col] where row is the element index (0..3) and col is the product index (0..6).
    # So U2[I*2+i, s] is the coefficient of element (I,i) in the s-th product?
    # No, U2 is for a 2x2 matrix. The rows of U2 correspond to the 4 elements of the 2x2 matrix.
    # In the 4x4 case, the "elements" of the outer 2x2 matrix are the 2x2 blocks.
    # So we need to map the outer Strassen coefficients to the block indices.
    # U_outer[I*2+J, s] = U2[I*2+J, s] ? 
    # Yes, because the outer structure is also 2x2.
    # Similarly for V and W.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U4
            for I in range(2):
                for J in range(2):
                    u_outer = U2[I*2+J, s]
                    if u_outer == 0: continue
                    for i in range(2):
                        for j in range(2):
                            u_inner = U2[i*2+j, t]
                            if u_inner == 0: continue
                            row_a = (I*2+i)*4 + (J*2+j)
                            U4[row_a, col] = u_outer * u_inner
                            
            # V4
            for J in range(2):
                for K in range(2):
                    v_outer = V2[J*2+K, s]
                    if v_outer == 0: continue
                    for j in range(2):
                        for k in range(2):
                            v_inner = V2[j*2+k, t]
                            if v_inner == 0: continue
                            row_b = (J*2+j)*4 + (K*2+k)
                            V4[row_b, col] = v_outer * v_inner
                            
            # W4
            for K in range(2):
                for I in range(2):
                    w_outer = W2[K*2+I, s]
                    if w_outer == 0: continue
                    for k in range(2):
                        for i in range(2):
                            w_inner = W2[k*2+i, t]
                            if w_inner == 0: continue
                            row_c = (K*2+k)*4 + (I*2+i)
                            W4[row_c, col] = w_outer * w_inner

    # Verify 4x4
    def verify_4x4(U4, V4, W4):
        np.random.seed(12345)
        for _ in range(10):
            A = np.random.randint(-10, 10, (4, 4)).astype(float)
            B = np.random.randint(-10, 10, (4, 4)).astype(float)
            C_true = A @ B
            C_decomp = np.zeros((4, 4))
            Af = A.flatten()
            Bf = B.flatten()
            for r in range(49):
                u_val = np.dot(U4[:, r], Af)
                v_val = np.dot(V4[:, r], Bf)
                scalar = u_val * v_val
                W_col = W4[:, r]
                # Reshape to 4x4 (k, i) then transpose to (i, k)
                C_decomp += (W_col.reshape(4, 4)).T * scalar
            if not np.allclose(C_true, C_decomp, atol=1e-9):
                return False
        return True

    if verify_4x4(U4, V4, W4):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        # Fallback
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((16, rank_fb))
        W_fb = np.zeros((16, rank_fb))
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col_fb] = 1
                    V_fb[j*4+k, col_fb] = 1
                    W_fb[k*4+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
