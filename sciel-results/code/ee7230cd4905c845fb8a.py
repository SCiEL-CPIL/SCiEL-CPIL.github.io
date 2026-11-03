import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 3, 4, 8
    
    # Schoolbook fallback
    def schoolbook():
        rank_sb = n * m * p  # 96
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
            return False
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        rng = np.random.default_rng(12345)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            M = U_3d * A[:, :, np.newaxis]
            N = V_3d * B[:, :, np.newaxis]
            S = np.einsum('ijc,jkc->ikc', M, N)
            S_T = np.transpose(S, (1, 0, 2))
            C_dec = np.sum(S_T * W_3d, axis=2)
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def to_clean_list(mat_np):
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(float(v))
            result.append(new_row)
        return result

    # Strategy: Deterministic Block-Recursive Strassen Composition
    # We decompose the 3x4x8 multiplication into:
    # 1. A 2x4x8 block (rows 0,1 of A) handled by Strassen-like recursion
    # 2. A 1x4x8 block (row 2 of A) handled by schoolbook
    
    # The 2x4x8 block can be split into two 2x2x4 blocks (cols 0,1 and 2,3 of B)
    # Each 2x2x4 block can be split into two 2x2x2 blocks (cols 0,1 and 2,3 of B)
    # Each 2x2x2 block uses Strassen (rank 7)
    
    # Total rank for 2x4x8: 2 (cols split) * 2 (cols split) * 7 (Strassen) = 28
    # Total rank for 1x4x8: 1*4*8 = 32 (schoolbook)
    # Total rank: 28 + 32 = 60
    
    # Wait, let's re-verify the Strassen composition.
    # For 2x2x2 Strassen, we have 7 terms.
    # For 2x2x4, we split B into two 2x2 blocks. 
    # C_left = A * B_left (2x2 * 2x2) -> 7 terms
    # C_right = A * B_right (2x2 * 2x2) -> 7 terms
    # Total 14 terms for 2x2x4.
    # For 2x4x4, we split A into two 1x2 rows? No, Strassen splits both A and B.
    # Actually, the standard recursive Strassen for 2x2x2x2x2x2...
    
    # Let's be more precise.
    # We want to compute C (3x8) = A (3x4) * B (4x8).
    
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4) and B2 (4x4).
    
    # C1 = A1 * B1 (2x4 * 4x4 = 2x4)
    # C2 = A1 * B2 (2x4 * 4x4 = 2x4)
    # C3 = A2 * B1 (1x4 * 4x4 = 1x4)
    # C4 = A2 * B2 (1x4 * 4x4 = 1x4)
    
    # C = [C1, C2; C3, C4] (where rows are stacked, cols are concatenated)
    
    # Now, how to compute C1 = A1 * B1 efficiently?
    # A1 is 2x4, B1 is 4x4.
    # Split A1 into A1a (2x2) and A1b (2x2).
    # Split B1 into B1a (4x2) and B1b (4x2).
    # C1a = A1 * B1a (2x4 * 4x2 = 2x2)
    # C1b = A1 * B1b (2x4 * 4x2 = 2x2)
    
    # C1a = A1 * B1a.
    # A1 is 2x4. B1a is 4x2.
    # Split A1 into A1_top (1x4) and A1_bot (1x4)? No, Strassen works on square blocks.
    # Let's use the fact that 2x4x2 can be done by splitting the 4 into 2+2.
    # A1 (2x4) = [A1a | A1b] where A1a, A1b are 2x2.
    # B1a (4x2) = [B1a_top; B1a_bot] where B1a_top, B1a_bot are 2x2.
    # C1a = A1a * B1a_top + A1b * B1a_bot (2x2 + 2x2)
    # Each 2x2x2 multiplication can be done with Strassen (rank 7).
    # So C1a takes 7 + 7 = 14 terms.
    # Similarly, C1b takes 14 terms.
    # Total for C1 (2x4x4) is 28 terms.
    
    # Similarly, C2 (2x4x4) takes 28 terms.
    # C3 (1x4x4) and C4 (1x4x4) can be done with schoolbook: 1*4*4 = 16 terms each.
    # Total for C3, C4 is 32 terms.
    
    # Total rank = 28 (C1) + 28 (C2) + 16 (C3) + 16 (C4) = 88.
    
    # This matches the "inspiration" finding of rank 88.
    # Let's implement this deterministic construction.
    
    def strassen_2x2x2(U_base, V_base, W_base, offset):
        # Adds 7 Strassen terms to U, V, W starting at column offset
        # U_base: (2, 2, r) slice? No, let's use flat indices.
        # This helper is complex. Let's just build the full matrices directly.
        pass

    # Direct construction of rank 88 decomposition
    
    U = np.zeros((n * m, 88))
    V = np.zeros((m * p, 88))
    W = np.zeros((p * n, 88))
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We will define a helper to add a Strassen block for specific (i1,i2,j1,j2,k1,k2)
    def add_strassen(i1, i2, j1, j2, k1, k2, offset):
        nonlocal col
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1
        U[i1*m + j1, offset] += 1
        U[i2*m + j2, offset] += 1
        V[j1*p + k1, offset] += 1
        V[j2*p + k2, offset] += 1
        W[k1*n + i1, offset] += 1
        W[k2*n + i2, offset] += 1
        # M2
        U[i2*m + j1, offset+1] += 1
        U[i2*m + j2, offset+1] += 1
        V[j1*p + k1, offset+1] += 1
        W[k1*n + i2, offset+1] += 1
        # M3
        U[i1*m + j1, offset+2] += 1
        V[j2*p + k1, offset+2] += 1
        V[j2*p + k2, offset+2] -= 1
        W[k1*n + i1, offset+2] += 1
        W[k2*n + i1, offset+2] += 1
        # M4
        U[i2*m + j2, offset+3] += 1
        V[j1*p + k2, offset+3] += 1
        V[j1*p + k1, offset+3] -= 1
        W[k2*n + i2, offset+3] += 1
        W[k1*n + i1, offset+3] += 1
        # M5
        U[i1*m + j1, offset+4] += 1
        U[i1*m + j2, offset+4] += 1
        V[j2*p + k2, offset+4] += 1
        W[k2*n + i1, offset+4] -= 1
        W[k2*n + i2, offset+4] += 1
        # M6
        U[i2*m + j1, offset+5] += 1
        U[i1*m + j1, offset+5] -= 1
        V[j1*p + k1, offset+5] += 1
        V[j1*p + k2, offset+5] += 1
        W[k1*n + i2, offset+5] += 1
        W[k2*n + i2, offset+5] += 1
        # M7
        U[i1*m + j2, offset+6] += 1
        U[i2*m + j2, offset+6] -= 1
        V[j2*p + k1, offset+6] += 1
        V[j2*p + k2, offset+6] += 1
        W[k1*n + i1, offset+6] += 1
        W[k2*n + i1, offset+6] += 1
        
        col = offset + 7

    # 1. C1 = A1 * B1 (i=0,1; j=0,1,2,3; k=0,1,2,3)
    # Split into C1a (k=0,1) and C1b (k=2,3)
    # C1a = A1 * B1a
    # A1 = [A1a | A1b] (j=0,1 and j=2,3)
    # B1a = [B1a_top; B1a_bot] (j=0,1 and j=2,3)
    # C1a = A1a * B1a_top + A1b * B1a_bot
    # A1a (i=0,1; j=0,1) * B1a_top (j=0,1; k=0,1) -> Strassen (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 1, 0, 1, 0, 1, col)
    # A1b (i=0,1; j=2,3) * B1a_bot (j=2,3; k=0,1) -> Strassen (i=0,1; j=2,3; k=0,1)
    add_strassen(0, 1, 2, 3, 0, 1, col)
    
    # C1b = A1 * B1b
    # A1a (i=0,1; j=0,1) * B1b_top (j=0,1; k=2,3) -> Strassen (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 1, 0, 1, 2, 3, col)
    # A1b (i=0,1; j=2,3) * B1b_bot (j=2,3; k=2,3) -> Strassen (i=0,1; j=2,3; k=2,3)
    add_strassen(0, 1, 2, 3, 2, 3, col)
    
    # 2. C2 = A1 * B2 (i=0,1; j=0,1,2,3; k=4,5,6,7)
    # Split into C2a (k=4,5) and C2b (k=6,7)
    # C2a = A1a * B2a_top + A1b * B2a_bot
    add_strassen(0, 1, 0, 1, 4, 5, col)
    add_strassen(0, 1, 2, 3, 4, 5, col)
    
    # C2b = A1a * B2b_top + A1b * B2b_bot
    add_strassen(0, 1, 0, 1, 6, 7, col)
    add_strassen(0, 1, 2, 3, 6, 7, col)
    
    # 3. C3 = A2 * B1 (i=2; j=0,1,2,3; k=0,1,2,3)
    # Schoolbook: 1*4*4 = 16 terms
    for j in range(4):
        for k in range(4):
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
            
    # 4. C4 = A2 * B2 (i=2; j=0,1,2,3; k=4,5,6,7)
    # Schoolbook: 1*4*4 = 16 terms
    for j in range(4):
        for k in range(4, 8):
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
            
    if col != 88:
        # Fallback if construction failed
        return best
    
    # Verify the constructed decomposition
    U_list = to_clean_list(U)
    V_list = to_clean_list(V)
    W_list = to_clean_list(W)
    
    if verify(U_list, V_list, W_list, 88):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification failed, try random search for rank 88 or lower
    # But first, return schoolbook as fallback
    return best
