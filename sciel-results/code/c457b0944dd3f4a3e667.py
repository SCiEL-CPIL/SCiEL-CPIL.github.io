import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    n, m, p = 2, 4, 4
    target_rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    # Helper to get global indices
    # U index: (i, j) where i is row of A, j is col of A (0..3) -> i*4+j
    # V index: (i, j) where i is row of B, j is col of B (0..3) -> i*4+j
    # W index: (i, j) where i is row of C (0..1), j is col of C (0..3)
    # Task says: W[k*n+i, r]. Here n=2.
    # Standard C is 2x4.
    # Index in W matrix corresponds to output element C[i, j].
    # The tensor T_{i, j, k} = sum_l A_{i, l} B_{l, j}.
    # U corresponds to A_{i, l}. V corresponds to B_{l, j}. W corresponds to C_{i, j}.
    # Wait, the task description says:
    # U: (n*m, rank) -> (8, rank)
    # V: (m*p, rank) -> (16, rank)
    # W: (p*n, rank) -> (8, rank)
    # And "output factor follows the official trace convention W[k*n+i,r]"
    # This usually implies W is indexed by (k, i) where k is col of C (0..p-1) and i is row of C (0..n-1)?
    # Or is it C[i, k]?
    # Let's look at the seed's W indexing:
    # def get_w_idx(i, j): return j * n + i
    # Here i is row of C (0,1), j is col of C (0,1,2,3).
    # So W[j*n + i, r] corresponds to C[i, j].
    # This matches "W[k*n+i]" if k=j (col) and i=i (row).
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(i, j):
        return i * p + j
        
    def w_idx(i, j):
        # i: row of C (0,1)
        # j: col of C (0,1,2,3)
        return j * n + i

    # Define Strassen 2x2 terms
    # Each term is a list of (u_idx, v_idx, w_idx, coeff)
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00     -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)     -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)     -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11     -> C01+=M5, C00-=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    # We define local indices 0,1 for rows/cols in the 2x2 block.
    # Term structure: [ (u_local, v_local, w_local, coeff), ... ]
    # We need to map local (0,1) to global indices based on the block position.
    
    strassen_terms = [
        # M1
        [((0,0), (0,0), (0,0), 1), ((1,1), (0,0), (0,0), 1),
         ((0,0), (1,1), (0,0), 1), ((1,1), (1,1), (0,0), 1),
         ((0,0), (0,0), (1,1), 1), ((1,1), (0,0), (1,1), 1),
         ((0,0), (1,1), (1,1), 1), ((1,1), (1,1), (1,1), 1)],
        # M2
        [((1,0), (0,0), (1,0), 1), ((1,1), (0,0), (1,0), 1),
         ((1,0), (0,0), (1,1), -1), ((1,1), (0,0), (1,1), -1)],
        # M3
        [((0,0), (0,1), (0,1), 1), ((0,0), (1,1), (0,1), -1),
         ((0,0), (0,1), (1,1), 1), ((0,0), (1,1), (1,1), 1)],
        # M4
        [((1,1), (1,0), (0,0), 1), ((1,1), (0,0), (0,0), -1),
         ((1,1), (1,0), (1,0), 1), ((1,1), (0,0), (1,0), 1)],
        # M5
        [((0,0), (1,1), (0,1), 1), ((0,1), (1,1), (0,1), 1),
         ((0,0), (1,1), (0,0), -1), ((0,1), (1,1), (0,0), -1)],
        # M6
        [((1,0), (0,0), (1,1), 1), ((0,0), (0,0), (1,1), -1),
         ((1,0), (0,1), (1,1), 1), ((0,0), (0,1), (1,1), -1)],
        # M7
        [((0,1), (1,0), (0,0), 1), ((1,1), (1,0), (0,0), -1),
         ((0,1), (1,1), (0,0), 1), ((1,1), (1,1), (0,0), -1)]
    ]
    
    # Define the 4 blocks
    # Block (a_r, a_c, b_r, b_c, c_c)
    # A rows a_r..a_r+1, cols a_c..a_c+1
    # B rows b_r..b_r+1, cols b_c..b_c+1
    # C rows 0..1 (since A has 2 rows), cols c_c..c_c+1
    blocks = [
        (0, 0, 0, 0, 0), # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0), # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 2), # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 2), # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]
    
    rank = 0
    for block in blocks:
        if time.time() > deadline:
            break
            
        a_r, a_c, b_r, b_c, c_c = block
        
        # Map local indices (0,1) to global offsets
        a_row_offset = a_r
        a_col_offset = a_c
        b_row_offset = b_r
        b_col_offset = b_c
        c_row_offset = 0 # A always has 2 rows, so C rows are 0,1
        c_col_offset = c_c
        
        for term in strassen_terms:
            if rank >= target_rank:
                break
                
            for (u_l, v_l, w_l, coeff) in term:
                # Map local u_l=(i,j) to global U index
                g_u_i = a_row_offset + u_l[0]
                g_u_j = a_col_offset + u_l[1]
                idx_u = u_idx(g_u_i, g_u_j)
                
                # Map local v_l=(i,j) to global V index
                g_v_i = b_row_offset + v_l[0]
                g_v_j = b_col_offset + v_l[1]
                idx_v = v_idx(g_v_i, g_v_j)
                
                # Map local w_l=(i,j) to global W index
                g_w_i = c_row_offset + w_l[0]
                g_w_j = c_col_offset + w_l[1]
                idx_w = w_idx(g_w_i, g_w_j)
                
                U[idx_u, rank] += coeff
                V[idx_v, rank] += coeff
                W[idx_w, rank] += coeff
                
            rank += 1
            
    # Ensure we didn't run out of rank
    if rank < target_rank:
        # Pad with zeros if necessary (shouldn't happen)
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
    else:
        U = U[:, :target_rank]
        V = V[:, :target_rank]
        W = W[:, :target_rank]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
