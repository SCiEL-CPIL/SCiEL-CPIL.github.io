import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130.0
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Structured Block Decomposition
    # Split the p=9 dimension into three blocks of size 3.
    # Each block is a 2x2x3 multiplication.
    # We hypothesize that the rank of 2x2x3 is 10.
    # Total rank R = 3 * 10 = 30.
    
    # We will try to find a rank 30 decomposition.
    # If that fails, we fall back to the known rank 32 (4x Strassen(2x2x2) + 1x Schoolbook(2x2x1)).
    
    def get_residuals(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U_flat = params[:R * size_u].reshape(R, size_u)
        V_flat = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
        W_flat = params[R * (size_u + size_v) :].reshape(R, size_w)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.sum(U_flat[:, ui] * V_flat[:, vj] * W_flat[:, wk])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def verify_decomposition(U, V, W):
        # U, V, W are (size, R)
        R = U.shape[1]
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.sum(U[ui, :] * V[vj, :] * W[wk, :])
                    err = max(err, abs(val - 1.0))
        return err

    # Helper to construct a 2x2x2 Strassen block
    # Indices in U: i*m+j (i in 0..1, j in 0..1) -> 0,1,2,3
    # Indices in V: j*p+k (j in 0..1, k in 0..1) -> 0,1,2,3
    # Indices in W: k*n+i (k in 0..1, i in 0..1) -> 0,1,2,3
    # Strassen terms for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # We will build a rank 30 candidate.
    # 3 blocks of 2x2x3.
    # For each 2x2x3 block, we use 10 terms.
    # The first 7 terms are Strassen for the first 2 columns (k=0,1).
    # The last 3 terms are for the 3rd column (k=2).
    # Wait, schoolbook for 2x2x1 is rank 4.
    # So 7 + 4 = 11 terms per block. Total 33.
    # To get 10, we need to share terms or use a better 2x2x3 algo.
    # Let's just try to optimize a rank 30 decomposition with a random init but constrained to be "close" to the block structure?
    # No, let's try rank 30 with random init first, but with a better solver setup.
    
    # Actually, let's try the known rank 32 construction first as a baseline, 
    # then try to "prune" it to rank 30 or 31.
    
    def build_rank_32():
        # 4 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Schoolbook, rank 4)
        # p=9. Blocks: k=0,1; k=2,3; k=4,5; k=6,7; k=8.
        U = np.zeros((n * m, 32))
        V = np.zeros((m * p, 32))
        W = np.zeros((p * n, 32))
        
        col = 0
        # Strassen 2x2x2 template
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        strassen_u = [
            [1, 0, 1, 0, 1, 0, 0], # A11
            [0, 0, 0, 0, 0, -1, 1], # A12
            [0, 1, 0, 0, 0, 1, 0],  # A21
            [1, 1, 0, 1, 0, 0, 0]   # A22
        ]
        strassen_v = [
            [1, 0, 0, 0, 0, 1, 0], # B11
            [0, 0, 1, 0, 0, 1, 0], # B12
            [1, 1, -1, -1, 0, 0, 1], # B21
            [1, 0, -1, 1, 1, 0, 1]  # B22
        ]
        strassen_w = [
            [1, 0, 0, 0, 0, 0, 0], # C11
            [0, 0, 1, 0, 1, 0, 0], # C12
            [0, 1, 0, 1, 0, 1, 1], # C21
            [0, 0, 0, 0, 0, 1, 1]  # C22
        ]
        # Note: The Strassen matrices above are for A (2x2), B (2x2), C (2x2).
        # We need to map these to the tensor indices.
        
        for block_idx in range(4):
            k_offset = block_idx * 2
            for r in range(7):
                # U part: depends on i, j
                for i in range(2):
                    for j in range(2):
                        u_val = strassen_u[i][j] if r < 7 else 0 # This mapping is wrong, strassen_u is 2x2 for A, but we have 7 terms.
                        # Let's use explicit tuples for Strassen terms
                        pass
                
        # Let's use a simpler, verified Strassen implementation for 2x2x2
        # Terms (u, v, w) where u, v, w are 4-dim vectors (flattened 2x2)
        # u index: i*m+j
        # v index: j*p+k (local k)
        # w index: k*n+i (local k)
        
        s_terms = [
            # (U_vec, V_vec, W_vec)
            # M1 = (A11+A22)(B11+B22)
            (np.array([1,0,0,1]), np.array([1,0,1,0]), np.array([1,0,0,0]) + np.array([0,0,0,1]) + np.array([0,1,0,0]) - np.array([0,0,1,0]) - np.array([1,0,0,0])), # This is getting messy.
        ]
        
        # Let's just use the known rank 32 from the "inspiration" logic if it's easy, or just return rank 36 if we can't do better.
        # But we want to beat 1.0 (which is 36). Rank 32 gives score 36/32 = 1.125.
        # The current program returns 36.
        # So if we just implement Rank 32 correctly, we win.
        
        # Let's construct Rank 32 explicitly.
        # 4 Strassen blocks (k=0..7) + 1 Schoolbook block (k=8).
        
        U = np.zeros((4, 32))
        V = np.zeros((18, 32))
        W = np.zeros((18, 32))
        
        # Strassen 2x2x2 terms
        # U (A): 
        # 1: [1,0,0,1] (A11+A22)
        # 2: [0,0,1,1] (A21+A22)
        # 3: [1,0,0,0] (A11)
        # 4: [0,0,0,1] (A22)
        # 5: [1,1,0,0] (A11+A12)
        # 6: [-1,0,1,0] (A21-A11)
        # 7: [0,1,0,-1] (A12-A22)
        
        # V (B):
        # 1: [1,0,1,0] (B11+B22)
        # 2: [1,0,0,0] (B11)
        # 3: [0,1,0,-1] (B12-B22)
        # 4: [0,-1,1,0] (B21-B11)
        # 5: [0,0,0,1] (B22)
        # 6: [1,1,0,0] (B11+B12)
        # 7: [0,0,1,1] (B21+B22)
        
        # W (C):
        # 1: [1,0,0,1] (C11+C22) -- Wait, C11 gets M1-M2+M4-M6, C12 gets M3+M6, C21 gets M2+M5, C22 gets M1-M3+M4+M7
        # Let's just use the standard Strassen coefficients.
        # C11 = M1 - M2 + M4 - M6
        # C12 = M3 + M6
        # C21 = M2 + M5
        # C22 = M1 - M3 + M4 + M7
        
        w_vecs = [
            np.array([1,0,0,0]), # C11 coeff for M1
            np.array([0,0,1,0]), # C21 coeff for M1? No.
        ]
        # Let's define W vectors for each term r (0..6) and each output index (0..3)
        # Output index 0: C11 (i=0, k=0) -> W[0*n+0] = W[0]
        # Output index 1: C12 (i=0, k=1) -> W[1*n+0] = W[1]
        # Output index 2: C21 (i=1, k=0) -> W[0*n+1] = W[2]
        # Output index 3: C22 (i=1, k=1) -> W[1*n+1] = W[3]
        
        # M1: C11+=1, C22+=1
        # M2: C11-=1, C21+=1
        # M3: C12+=1, C22-=1
        # M4: C11+=1, C22+=1
        # M5: C21+=1
        # M6: C11-=1, C12+=1
        # M7: C22+=1
        
        w_coeffs = [
            np.array([1, 0, 0, 1]), # M1
            np.array([-1, 0, 1, 0]), # M2
            np.array([0, 1, 0, -1]), # M3
            np.array([1, 0, 0, 1]), # M4
            np.array([0, 0, 1, 0]), # M5
            np.array([-1, 1, 0, 0]), # M6
            np.array([0, 0, 0, 1])  # M7
        ]
        
        u_vecs = [
            np.array([1,0,0,1]), # M1: A11+A22
            np.array([0,0,1,1]), # M2: A21+A22
            np.array([1,0,0,0]), # M3: A11
            np.array([0,0,0,1]), # M4: A22
            np.array([1,1,0,0]), # M5: A11+A12
            np.array([-1,0,1,0]), # M6: A21-A11
            np.array([0,1,0,-1])  # M7: A12-A22
        ]
        
        v_vecs = [
            np.array([1,0,1,0]), # M1: B11+B22
            np.array([1,0,0,0]), # M2: B11
            np.array([0,1,0,-1]), # M3: B12-B22
            np.array([0,-1,1,0]), # M4: B21-B11
            np.array([0,0,0,1]), # M5: B22
            np.array([1,1,0,0]), # M6: B11+B12
            np.array([0,0,1,1])  # M7: B21+B22
        ]
        
        col = 0
        for block in range(4):
            k_off = block * 2
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        U[i*m+j, col] = u_vecs[r][i*m+j]
                
                # V
                for j in range(2):
                    for k in range(2):
                        V[j*p + (k_off+k), col] = v_vecs[r][j*2+k]
                
                # W
                for k in range(2):
                    for i in range(2):
                        W[(k_off+k)*n + i, col] = w_coeffs[r][k*2+i]
                        
                col += 1
        
        # Last block: k=8 (2x2x1)
        # Schoolbook 4 terms
        # A11*B11 -> C11
        # A12*B11 -> C12 (No, B is 2x1, so B11, B21)
        # A is 2x2, B is 2x1.
        # C11 = A11*B11 + A12*B21
        # C21 = A21*B11 + A22*B21
        
        # Term 1: A11 * B11 -> C11
        U[0, col] = 1; V[0*p+8, col] = 1; W[8*n+0, col] = 1; col+=1
        # Term 2: A12 * B21 -> C11
        U[1, col] = 1; V[1*p+8, col] = 1; W[8*n+0, col] = 1; col+=1
        # Term 3: A21 * B11 -> C21
        U[2, col] = 1; V[0*p+8, col] = 1; W[8*n+1, col] = 1; col+=1
        # Term 4: A22 * B21 -> C21
        U[3, col] = 1; V[1*p+8, col] = 1; W[8*n+1, col] = 1; col+=1
        
        # Verify
        err = verify_decomposition(U, V, W)
        if err < 1e-8:
            return U, V, W, 32
        
        return None

    U32, V32, W32, R32 = build_rank_32()
    
    if U32 is not None:
        # Try to optimize to lower rank? 
        # For now, return rank 32.
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    # Fallback to schoolbook
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
