"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Use a 2x2x3 block decomposition with rank 6, repeated 4 times for rank 24.
    If that fails verification, fall back to Strassen rank-42.
    
    For 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}
    We need 6 output terms. Try to find 6 rank-1 terms.
    
    The decomposition satisfies:
    For all i in [0,2), j in [0,2), k in [0,12):
      sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    t_start = time.time()
    deadline = t_start + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-24 approach: 4 blocks of 2x2x3, each with rank 6
    # For 2x2x3, the 6 outputs are C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    # Each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # So we need 6 rank-1 terms, one per output, but we can combine them.
    # Actually the minimum rank for 2x2x3 is 6 (schoolbook), since each output 
    # involves a distinct pair (i,k) and the bilinear form doesn't factor further.
    # Wait - can we do better than 6? The tensor has 6 non-zero entries in the 
    # appropriate slicing. Let's just use schoolbook rank 6 for 2x2x3.
    
    # Rank 24 approach: 4 blocks of 3 columns each
    rank_24 = 24
    num_blocks = 4
    cols_per_block = 3
    
    U = np.zeros((n * m, rank_24), dtype=np.float64)
    V = np.zeros((m * p, rank_24), dtype=np.float64)
    W = np.zeros((p * n, rank_24), dtype=np.float64)
    
    for blk in range(num_blocks):
        base = blk * cols_per_block
        k_start = blk * cols_per_block
        
        for k_local in range(cols_per_block):
            c = base + k_local
            k_global = k_start + k_local
            
            # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
            # Term for A[0,0]*B[0,k]:
            U[0, c] = 1  # A[0,0] -> index 0*2+0 = 0
            V[0 * p + k_global, c] = 1  # B[0,k]
            W[k_global * n + 0, c] = 1  # C[0,k] -> index k*2+0
            
            # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
            # We need separate terms for these
            # Actually, each output needs its own rank-1 term in schoolbook.
            # C[0,k] needs: U[0,c1]=1, V[0*p+k,c1]=1, W[k*n+0,c1]=1
            #              and U[1,c2]=1, V[1*p+k,c2]=1, W[k*n+0,c2]=1
            # That's 2 terms per output, 6 outputs = 12 terms per block of 3 cols
            # That gives rank 48, worse than Strassen!
            # 
            # Let me reconsider. For 2x2x3, schoolbook is rank 6? No.
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] -> 2 terms
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] -> 2 terms
            # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] -> 2 terms
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] -> 2 terms
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] -> 2 terms
            # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] -> 2 terms
            # Total: 12 terms for 2x2x3 schoolbook. That's worse.
            #
            # So Strassen at rank 42 is likely near-optimal. Let me try a hybrid:
            # Use Strassen for pairs but try to share across 3 columns.
            #
            # Alternative: Try rank 28 by using a 2x2x4 block with rank 7 (if possible)
            # repeated 3 times. But 2x2x4 minimum rank is at least 8 (since 8 outputs).
            # Actually no - the rank is about the tensor, not the number of outputs.
            # For 2x2x4, can we do rank 7? Probably not easily.
            #
            # Let me try a different approach: use the fact that we can group 4 columns
            # and use a rank-9 decomposition for 2x2x4, then repeat 3 times = rank 27.
            #
            # For 2x2x4: 8 outputs. Schoolbook = 16 terms. 
            # Can we do better? Let's try Strassen on 2x2x2 twice = rank 14.
            # Or find a rank-9 or lower decomposition.
            #
            # Actually, let me just stick with Strassen rank 42 but verify it's correct.
            # The current approach is sound. Let me verify and return it.
            pass
    
    # The above analysis shows rank 24 via schoolbook is actually rank 48 (worse).
    # Strassen rank 42 is the best known simple approach.
    # Let me construct the Strassen rank-42 decomposition properly.
    
    rank = 42
    num_pairs = 6
    cols_per_pair = 2
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each term, specify:
    # u: list of (A_index, coeff) where A_index is 0..3 for A[0,0],A[0,1],A[1,0],A[1,1]
    # v: list of (B_j, B_k_local, coeff) 
    # w: list of (C_k_local, C_i, coeff)
    
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        # C00 += M1, C11 += M1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A10+A11)*B00
        # C10 += M2, C11 -= M2
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 1, 1), (1, 1, -1)]
        },
        # M3 = A00*(B01-B11)
        # C01 += M3, C11 += M3
        {
            'u': [(0, 1)],
            'v': [(0, 1, 1), (1, 1, -1)],
            'w': [(1, 0, 1), (1, 1, 1)]
        },
        # M4 = A11*(B10-B00)
        # C00 += M4, C10 += M4
        {
            'u': [(3, 1)],
            'v': [(1, 0, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (0, 1, 1)]
        },
        # M5 = (A00+A01)*B11
        # C00 -= M5, C01 += M5
        {
            'u': [(0, 1), (1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 0, -1), (1, 0, 1)]
        },
        # M6 = (A10-A00)*(B00+B01)
        # C11 += M6
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)]
        },
        # M7 = (A01-A11)*(B10+B11)
        # C00 += M7
        {
            'u': [(1, 1), (3, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)]
        }
    ]
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t
        
        for s in range(7):
            c = base + s
            term = strassen_terms[s]
            
            for u_idx, val in term['u']:
                U[u_idx, c] += val
                
            for j, k_local, val in term['v']:
                global_k = k0 + k_local
                V[j * p + global_k, c] += val
                
            for k_local, i, val in term['w']:
                global_k = k0 + k_local
                W[global_k * n + i, c] += val

    # Verify the decomposition
    if time.time() < deadline - 5:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err > 1e-10:
            # Fallback to a simpler but guaranteed correct approach
            # Use direct schoolbook: rank = n*m*p = 48 (worse but correct)
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
            V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
            W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, idx] = 1.0
                        V_fb[j*p+k, idx] = 1.0
                        W_fb[k*n+i, idx] = 1.0
                        idx += 1
            
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
