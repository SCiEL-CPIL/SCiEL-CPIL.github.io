import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Tensor T[u,v,w] = sum_a U[u,a] V[v,a] W[w,a]
    # u = i*m+j, v = j*p+k, w = k*n+i
    # For fixed (i,j,k), we need T[i*m+j, j*p+k, k*n+i] == 1 and all other entries 0.
    # Let's compute the tensor as a 3D array.
    # T_flat[u, v, w]
    # This is equivalent to: for each a, add outer(U[:,a], outer(V[:,a], W[:,a]))
    # Or more efficiently:
    # T[u,v,w] = sum_a U[u,a] * V[v,a] * W[w,a]
    # We can check this by computing the tensor for all u, v, w?
    # Size: (nm, mp, pn) = (6, 24, 16) = 2304 entries.
    # This is fast.
    T = np.zeros((n * m, m * p, p * n))
    for a in range(rank):
        T += np.einsum('u,v,w->uvw', U[:, a], V[:, a], W[:, a])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > 1e-9:
                    return False
    
    # Check that all other entries are 0
    # Create a mask of the expected non-zero positions
    mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                mask[u, v, w] = True
    
    # Entries not in mask should be 0
    if np.any(np.abs(T[~mask]) > 1e-9):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 48
    
    # Schoolbook
    U_school = [[0] * full_rank for _ in range(n * m)]
    V_school = [[0] * full_rank for _ in range(m * p)]
    W_school = [[0] * full_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strassen-based construction for rank 44
    # We treat the 2x3 * 3x8 product as follows:
    # Split the 3 columns of A and 3 rows of B into:
    # Left 2 columns/rows (indices 0,1) and Right 1 column/row (index 2).
    # A = [A_L, A_R] where A_L is 2x2, A_R is 2x1.
    # B = [B_L; B_R] where B_L is 2x8, B_R is 1x8.
    # C = A_L * B_L + A_R * B_R.
    
    # A_R * B_R is 2x1 * 1x8 = 2x8. This can be done with rank 2 (schoolbook for 2x1x8? No, 2*1*8=16? No.
    # Wait. A_R is 2x1. B_R is 1x8.
    # C_R = A_R * B_R.
    # C_R[i, k] = A_R[i, 0] * B_R[0, k].
    # This is rank 1? No.
    # Tensor for 2x1 * 1x8:
    # T[i, k] = A[i] B[k].
    # This is an outer product. Rank 1?
    # Let's check the tensor definition.
    # U[i, a], V[k, a], W[i, k, a]? No, W is (p*n, rank).
    # For 2x1 * 1x8:
    # u = i (0..1), v = k (0..7), w = k*n + i (0..15).
    # T[i, k, k*2+i] = A[i] B[k].
    # This tensor has rank 1?
    # T[i, k, w] = A[i] B[k] delta_{w, 2k+i}.
    # Yes, this can be represented with rank 1?
    # Let U[i, 0] = A[i], V[k, 0] = B[k], W[w, 0] = 1 if w matches?
    # No, W depends on w which depends on i,k.
    # Actually, the "schoolbook" rank for 2x1*1x8 is 2*1*8 = 16.
    # But we can do better?
    # Let's just use schoolbook for the right part: 16 terms.
    # And Strassen for the left part (2x2 * 2x8):
    # 2x2 * 2x8.
    # Split 2x8 into two 2x4 blocks?
    # Or just treat 2x2 * 2x8 as 4 instances of 2x2 * 2x1? No.
    # 2x2 * 2x8 is rank 16 schoolbook.
    # Strassen for 2x2 * 2x2 is rank 7.
    # 2x2 * 2x8 can be seen as 4 independent 2x2 * 2x1? No, the 2x8 is contiguous.
    # Actually, 2x2 * 2x8 is just 4 columns of (2x2 * 2x2).
    # So we can use Strassen for each 2x2 * 2x2 block.
    # 2x8 has 4 blocks of width 2.
    # So 4 * 7 = 28 terms for the left part.
    # Plus 16 terms for the right part.
    # Total = 28 + 16 = 44.
    
    rank_strassen = 44
    U_new = [[0.0] * rank_strassen for _ in range(n * m)]
    V_new = [[0.0] * rank_strassen for _ in range(m * p)]
    W_new = [[0.0] * rank_strassen for _ in range(p * n)]
    
    # Part 1: Right part (A_R * B_R)
    # A_R is column 2 of A (index j=2).
    # B_R is row 2 of B (index j=2).
    # i in 0..1, k in 0..7.
    # u = i*3 + 2.
    # v = 2*8 + k = 16 + k.
    # w = k*2 + i.
    # Term for (i, k):
    col_offset = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U_new[u_idx][col_offset] = 1
            V_new[v_idx][col_offset] = 1
            W_new[w_idx][col_offset] = 1
            col_offset += 1
    # col_offset is now 16.
    
    # Part 2: Left part (A_L * B_L)
    # A_L is 2x2 (cols 0,1). B_L is 2x8 (rows 0,1).
    # We split B_L into 4 blocks of 2x2: B_L[:, 0:2], B_L[:, 2:4], B_L[:, 4:6], B_L[:, 6:8].
    # For each block b (0..3), we compute A_L * B_block using Strassen (rank 7).
    
    # Strassen matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A_L:
    # A00 -> u = 0*3+0 = 0
    # A01 -> u = 0*3+1 = 1
    # A10 -> u = 1*3+0 = 3
    # A11 -> u = 1*3+1 = 4
    
    # Indices for B_L block b:
    # B00 -> v = 0*8 + (2b+0) = 2b
    # B01 -> v = 0*8 + (2b+1) = 2b+1
    # B10 -> v = 1*8 + (2b+0) = 8+2b
    # B11 -> v = 1*8 + (2b+1) = 8+2b+1
    
    # Indices for C_L block b (which contributes to C[:, 2b:2b+2]):
    # C00 (i=0, k=2b) -> w = (2b)*2 + 0 = 4b
    # C01 (i=0, k=2b+1) -> w = (2b+1)*2 + 0 = 4b+2
    # C10 (i=1, k=2b) -> w = (2b)*2 + 1 = 4b+1
    # C11 (i=1, k=2b+1) -> w = (2b+1)*2 + 1 = 4b+3
    
    for b in range(4):
        base = col_offset
        # M1
        U_new[0][base] = 1
        U_new[4][base] = 1
        V_new[2*b][base] = 1
        V_new[8+2*b][base] = 1
        W_new[4*b][base] = 1
        W_new[4*b+3][base] = 1
        # M2
        U_new[3][base+1] = 1
        U_new[4][base+1] = 1
        V_new[2*b][base+1] = 1
        W_new[4*b+1][base+1] = 1
        W_new[4*b+3][base+1] = -1
        # M3
        U_new[0][base+2] = 1
        V_new[2*b+1][base+2] = 1
        V_new[8+2*b+1][base+2] = -1
        W_new[4*b+2][base+2] = 1
        W_new[4*b+3][base+2] = 1
        # M4
        U_new[4][base+3] = 1
        V_new[8+2*b][base+3] = 1
        V_new[2*b][base+3] = -1
        W_new[4*b][base+3] = 1
        W_new[4*b+1][base+3] = 1
        # M5
        U_new[0][base+4] = 1
        U_new[1][base+4] = 1
        V_new[8+2*b+1][base+4] = 1
        W_new[4*b][base+4] = -1
        W_new[4*b+2][base+4] = 1
        # M6
        U_new[3][base+5] = 1
        U_new[0][base+5] = -1
        V_new[2*b][base+5] = 1
        V_new[2*b+1][base+5] = 1
        W_new[4*b+3][base+5] = 1
        # M7
        U_new[1][base+6] = 1
        U_new[4][base+6] = -1
        V_new[8+2*b][base+6] = 1
        V_new[8+2*b+1][base+6] = 1
        W_new[4*b][base+6] = 1
        
        col_offset += 7
    
    # col_offset should be 16 + 4*7 = 44.
    
    # Verify
    if _verify(U_new, V_new, W_new, n, m, p):
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        # Fallback to schoolbook
        return best_result
