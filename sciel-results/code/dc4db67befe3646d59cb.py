import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # Order: A: a00,a01,a10,a11 ; B: b00,b01,b10,b11 ; C: c00,c01,c10,c11
    # Products:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # Reconstruction:
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6

    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0],    # b01
        [0, 0, 0, 1, 0, 0, 1],    # b10
        [1, 0, -1, 0, 1, 0, 1]    # b11
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0]    # c11
    ], dtype=np.float64)

    # Build decomposition:
    # For i in {0,1}: 2x2x8 using 4 Strassen blocks (2x2x2 each) -> rank 28
    # For i = 2: 1x2x8 schoolbook -> rank 16
    # Total rank = 2*28 + 16 = 72? No, wait.
    # The current program had rank 88. Let's re-read the current program logic.
    
    # Current program logic:
    # for j1 in range(2): # m split into 2 blocks of 2
    #   Top (i=0,1): 2x2x8. Split p=8 into 4 blocks of 2.
    #     Each 2x2x2 uses Strassen (rank 7).
    #     Total Top Rank = 4 * 7 = 28.
    #   Bot (i=2): 1x2x8. Schoolbook rank = 1*2*8 = 16.
    #   Total per m-block = 28 + 16 = 44.
    # Total Rank = 2 * 44 = 88.
    
    # This is the structure I must maintain/improve.
    
    rank = 88
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    for j1 in range(2):
        # Top part: i in {0,1}, j in {2*j1, 2*j1+1}
        # Shape 2x2x8. Split k into 4 blocks of 2.
        for k_block in range(4):
            k_start = 2 * k_block
            for c in range(7):
                # Strassen term c
                # U contributions (A part)
                # A indices: (0, 2*j1), (0, 2*j1+1), (1, 2*j1), (1, 2*j1+1)
                # Local indices 0,1,2,3 map to:
                # 0: (0,0) -> global (0, 2*j1)
                # 1: (0,1) -> global (0, 2*j1+1)
                # 2: (1,0) -> global (1, 2*j1)
                # 3: (1,1) -> global (1, 2*j1+1)
                
                # U2 rows: 0,1,2,3 correspond to local A indices
                # U2[0] is coeffs for a00
                # U2[1] is coeffs for a01
                # U2[2] is coeffs for a10
                # U2[3] is coeffs for a11
                
                # For each local A index li, add U2[li, c] to U[global_i, global_j, col]
                # li=0: i=0, j=2*j1
                U[0 * m + (2 * j1)][col] += U2[0, c]
                # li=1: i=0, j=2*j1+1
                U[0 * m + (2 * j1 + 1)][col] += U2[1, c]
                # li=2: i=1, j=2*j1
                U[1 * m + (2 * j1)][col] += U2[2, c]
                # li=3: i=1, j=2*j1+1
                U[1 * m + (2 * j1 + 1)][col] += U2[3, c]
                
                # V contributions (B part)
                # B indices: (2*j1, k_start), (2*j1, k_start+1), (2*j1+1, k_start), (2*j1+1, k_start+1)
                # Local indices 0,1,2,3 map to:
                # 0: (0,0) -> global (2*j1, k_start)
                # 1: (0,1) -> global (2*j1, k_start+1)
                # 2: (1,0) -> global (2*j1+1, k_start)
                # 3: (1,1) -> global (2*j1+1, k_start+1)
                
                # V2 rows: 0,1,2,3 correspond to local B indices
                V[(2 * j1) * p + k_start][col] += V2[0, c]
                V[(2 * j1) * p + (k_start + 1)][col] += V2[1, c]
                V[(2 * j1 + 1) * p + k_start][col] += V2[2, c]
                V[(2 * j1 + 1) * p + (k_start + 1)][col] += V2[3, c]
                
                # W contributions (C part)
                # C indices: (k_start, 0), (k_start, 1), (k_start+1, 0), (k_start+1, 1)
                # Local indices 0,1,2,3 map to:
                # 0: (0,0) -> global (k_start, 0)
                # 1: (0,1) -> global (k_start, 1)
                # 2: (1,0) -> global (k_start+1, 0)
                # 3: (1,1) -> global (k_start+1, 1)
                
                # W2 rows: 0,1,2,3 correspond to local C indices
                # W index convention: W[k*n + i, r]
                W[(k_start) * n + 0][col] += W2[0, c]
                W[(k_start) * n + 1][col] += W2[1, c]
                W[(k_start + 1) * n + 0][col] += W2[2, c]
                W[(k_start + 1) * n + 1][col] += W2[3, c]
                
                col += 1
        
        # Bot part: i=2, j in {2*j1, 2*j1+1}
        # Shape 1x2x8. Schoolbook.
        # C[2, k] = A[2, j] * B[j, k] for j in {2*j1, 2*j1+1}, summed?
        # No, it's just the product for each (j,k).
        # Rank = 1 * 2 * 8 = 16.
        for j2 in range(2):
            j_global = 2 * j1 + j2
            for k in range(8):
                # Term: A[2, j_global] * B[j_global, k] * C[2, k]
                U[2 * m + j_global][col] = 1.0
                V[j_global * p + k][col] = 1.0
                W[k * n + 2][col] = 1.0
                col += 1

    # Verify initial decomposition
    def verify(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should be 1 for all i,j,k
        
        # Efficient check
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_mat.reshape(n, m, rank)
        V_r = V_mat.reshape(m, p, rank)
        W_r = W_mat.reshape(p, n, rank)
        
        # We need to sum over r: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction.
        # For each i,j,k:
        # val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
        
        # Vectorized approach:
        # Create arrays of indices
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_vals = U_r[i_idx, j_idx, :]  # (n, m, p, rank)
        v_vals = V_r[j_idx, k_idx, :]  # (n, m, p, rank)
        w_vals = W_r[k_idx, i_idx, :]  # (n, m, p, rank)
        
        result = np.sum(u_vals * v_vals * w_vals, axis=3)
        return np.allclose(result, 1.0, atol=1e-9)

    if not verify(U, V, W):
        # Fallback to schoolbook if something went wrong (shouldn't happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Rank Reduction Strategy
    # We have 88 terms.
    # The structure is:
    # 8 Strassen blocks (each rank 7) for i=0,1.
    # 16 Schoolbook terms for i=2.
    
    # Can we reduce rank?
    # 1. Greedy removal: Try removing columns and check validity.
    # 2. Linear dependencies: If columns are linearly dependent in a way that allows cancellation?
    #    The tensor is fixed. If we have two terms that are identical, we can merge?
    #    But Strassen terms are distinct.
    #    Schoolbook terms for i=2 are distinct (different j,k).
    
    # Let's try greedy removal first. It's fast and might find obvious redundancies.
    # Then, we can try a more sophisticated local search or integer programming relaxation if time permits.
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=30.0):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        keep = np.ones(rank, dtype=bool)
        start_time = time.time()
        
        # Precompute for fast verification
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        
        def check_valid(keep_mask):
            idx = np.where(keep_mask)[0]
            if len(idx) == 0:
                return False
            U_r = U_mat[:, idx]
            V_r = V_mat[:, idx]
            W_r = W_mat[:, idx]
            
            U_rs = U_r.reshape(n, m, -1)
            V_rs = V_r.reshape(m, p, -1)
            W_rs = W_r.reshape(p, n, -1)
            
            u_vals = U_rs[i_idx, j_idx, :]
            v_vals = V_rs[j_idx, k_idx, :]
            w_vals = W_rs[k_idx, i_idx, :]
            
            result = np.sum(u_vals * v_vals * w_vals, axis=3)
            return np.allclose(result, 1.0, atol=1e-9)
        
        # Multiple passes
        for _pass in range(10):
            if time.time() - start_time > max_time:
                break
            removed_any = False
            # Shuffle order to avoid bias? Or just iterate.
            # Iterating is fine.
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                keep[r] = False
                if check_valid(keep):
                    removed_any = True
                else:
                    keep[r] = True
            if not removed_any:
                break
        
        idx = np.where(keep)[0]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat

    remaining_time = TIME_LIMIT - (time.time() - START_TIME) - 5.0
    if remaining_time > 5.0:
        U_opt, V_opt, W_opt = try_reduce_rank(U, V, W, max_time=remaining_time)
    else:
        U_opt, V_opt, W_opt = U, V, W

    # Final check
    if not verify(U_opt, V_opt, W_opt):
        # If reduction failed, return original
        U_opt, V_opt, W_opt = U, V, W

    # Convert to lists, ensuring integer/half-integer representation if possible
    def to_list(mat):
        # Round to nearest 0.5
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()

    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
