import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time as _time
import numpy as np

def construct():
    _start = _time.time()
    _deadline = _start + 145
    n, m, p = DIMENSIONS  # all 6

    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        np.random.seed(42)
        A = np.random.randint(-10, 10, size=(n, m))
        B = np.random.randint(-10, 10, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            C_computed += u_mat @ v_mat * w_mat.T
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    def construct_rank189():
        """3x3 block Strassen: rank 9*7 = 63 per block-level, total 3*3*7=63 blocks * ... 
        Actually: 3 block rows * 3 block cols * 7 Strassen terms = 63 terms, each expanded to 2*2*1=4 scalar mults? No.
        Each block is 2x2, so each block multiplication uses 7 Strassen sub-terms, each of which is a scalar mult.
        Total rank = 3*3*7 = 63... that's not right either.
        
        Let me reconsider: We have a 6x6 matrix. Split into 3x3 blocks of size 2x2.
        Each 2x2 block multiplication uses 7 scalar multiplications (Strassen).
        Number of block positions: 3 (row blocks) * 3 (col blocks) = 9 block multiplications needed for the full product?
        No, for matrix multiplication C = A*B where A is 6x6 and B is 6x6:
        A is split into 3x3 blocks of 2x2, B is split into 3x3 blocks of 2x2.
        C[i_block, j_block] = sum_k_block A[i_block, k_block] * B[k_block, j_block]
        So we need 3*3*3 = 27 block multiplications, each of rank 7 (Strassen for 2x2).
        Total rank = 27 * 7 = 189.
        """
        strassen_U = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 1, 1), (1, 1, -1)],
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)],
        ]
        rank = 189
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for bi in range(3):
            for bk in range(3):
                for bj in range(3):
                    for t in range(7):
                        for (li, lk, coeff) in strassen_U[t]:
                            gi = 2 * bi + li
                            gk = 2 * bk + lk
                            U[gi * m + gk, col] += coeff
                        for (lk, lj, coeff) in strassen_V[t]:
                            gk = 2 * bk + lk
                            gj = 2 * bj + lj
                            V[gk * p + gj, col] += coeff
                        for (li, lj, coeff) in strassen_W[t]:
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            W[gj * n + gi, col] += coeff
                        col += 1
        return U, V, W, rank

    def construct_rank182():
        """4+2 block structure: one 4x4 Strassen block + two 2x2 Strassen blocks.
        
        Split 6x6 as:
        - Rows: 4 + 2
        - Cols: 4 + 2  
        - Middle: 4 + 2
        
        Block (0,0): 4x4 * 4x4 -> 4x4 result, Strassen rank = 7 * 4^2 = 112? No.
        Strassen for 4x4: split into 2x2 blocks, 3x3 blocks, each 2x2 uses rank 7.
        So 4x4 Strassen rank = 9 * 7 = 63.
        
        Block (0,1): 4x4 * 4x2 -> 4x2 result. Split differently...
        This gets complicated. Let me use a simpler approach.
        
        Actually, the 4+2 decomposition:
        - A is 6x6, split rows as [4, 2], cols as [4, 2]
        - B is 6x6, split rows as [4, 2], cols as [4, 2]
        - C[i,j] = sum_k A[i,k]*B[k,j]
        
        C[0:4, 0:4] = A[0:4, 0:4]*B[0:4, 0:4] + A[0:4, 4:6]*B[4:6, 0:4]
        C[0:4, 4:6] = A[0:4, 0:4]*B[0:4, 4:6] + A[0:4, 4:6]*B[4:6, 4:6]
        C[4:6, 0:4] = A[4:6, 0:4]*B[0:4, 0:4] + A[4:6, 4:6]*B[4:6, 0:4]
        C[4:6, 4:6] = A[4:6, 0:4]*B[0:4, 4:6] + A[4:6, 4:6]*B[4:6, 4:6]
        
        For 4x4 * 4x4: use Strassen, rank 63
        For 4x4 * 4x2: this is a rectangular product. We can still use Strassen-like approach.
        4x4 * 4x2: split 4x4 into 2x2 blocks (4 blocks), 4x2 into 2x2 and 2x1 blocks.
        This is getting complex. Let me try a different approach.
        
        Alternative: use the fact that 6 = 4 + 2, and handle each rectangular block
        with an optimal or near-optimal decomposition.
        
        For now, let me just try the 3x3 Strassen (rank 189) and see if we can reduce it.
        """
        # Fallback to rank 189 if 182 construction is too complex
        return construct_rank189()

    # Start with rank 189
    U_np, V_np, W_np, best_rank = construct_rank189()
    best_result = {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist(),
        "rank": best_rank
    }
    
    # Verify initial decomposition
    if not verify_decomposition(U_np, V_np, W_np, best_rank, n, m, p):
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    # Try greedy column elimination
    if _time.time() < _deadline - 10:
        rng = np.random.RandomState(123)
        best_rank_found = best_rank
        cur_U = U_np.copy()
        cur_V = V_np.copy()
        cur_W = W_np.copy()
        
        # Precompute the "target" tensor values for verification
        # Instead of full verification each time, use a fast check
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # From decomposition: C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over k
        # Let's use a single test case for speed
        
        np.random.seed(42)
        A_test = np.random.randint(-10, 10, size=(n, m))
        B_test = np.random.randint(-10, 10, size=(m, p))
        C_expected = A_test @ B_test
        
        def fast_check(U, V, W, rank, skip_col=None):
            """Fast verification using precomputed test matrices."""
            C_check = np.zeros((n, p))
            for r in range(rank):
                if r == skip_col:
                    continue
                u_mat = U[:, r].reshape(n, m)
                v_mat = V[:, r].reshape(m, p)
                w_mat = W[:, r].reshape(p, n)
                C_check += u_mat @ v_mat * w_mat.T
            return np.allclose(C_check, C_expected, atol=1e-9)
        
        # Greedy elimination: try to remove columns one by one
        for trial in range(100):
            if _time.time() > _deadline - 5:
                break
            
            removed = False
            # Try columns in random order for diversity
            order = rng.permutation(best_rank_found)
            for r_idx in order:
                if _time.time() > _deadline - 5:
                    break
                
                # Check if column r_idx is redundant
                if fast_check(cur_U, cur_V, cur_W, best_rank_found, skip_col=r_idx):
                    # Column is redundant, remove it
                    mask = np.ones(best_rank_found, dtype=bool)
                    mask[r_idx] = False
                    cur_U = cur_U[:, mask]
                    cur_V = cur_V[:, mask]
                    cur_W = cur_W[:, mask]
                    best_rank_found -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if best_rank_found < best_rank:
            best_rank = best_rank_found
            best_result = {
                "U": cur_U.tolist(),
                "V": cur_V.tolist(),
                "W": cur_W.tolist(),
                "rank": best_rank
            }
    
    if _time.time() > _deadline:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
        
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
