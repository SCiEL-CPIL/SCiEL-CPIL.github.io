import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Strategy: Decompose 2x4 x 4x8 into two 2x2 x 2x8 blocks along j-dimension.
    # Block 0: j in {0,1}, Block 1: j in {2,3}
    # For each block, use a rank-14 decomposition of the 2x2x8 tensor.
    # The 2x2x8 tensor: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # Key idea: Use Strassen on 2x2x2 sub-blocks but share A-side combinations.
    # For 4 k-pairs, use the same 7 Strassen A-combinations, but the B-side
    # and W-side are per-k-pair. This gives 7*4 = 28 per block.
    # 
    # Wait - that's 28 per block, total 56. Let me think if we can do better.
    #
    # Alternative: The 2x2x8 tensor has rank at most 16 (naive). 
    # Can we achieve rank 14? Let's try a direct construction.
    #
    # For 2x2x8: C[i,k] = a0*b0k + a1*b1k for i=0,1; k=0..7
    # where a0=A[i,0], a1=A[i,1], b0k=B[0,k], b1k=B[1,k]
    #
    # Naive: 16 terms (a_i * b_jk for each i,j,k)
    # 
    # Try to save 2 terms using Strassen-like sharing:
    # Group k into pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair, use Strassen 2x2x2: 7 terms per pair = 28. Worse.
    #
    # Alternative sharing: Use 2 A-combos and 7 k-groupings?
    # A-combos: (a0+a1), (a0-a1) -> 2 combos
    # For each k, we need: a0*b0k + a1*b1k
    # = 0.5*((a0+a1)*(b0k+b1k) + (a0-a1)*(b0k-b1k))
    # So: term_k1 = (a0+a1)*(b0k+b1k)*0.5, term_k2 = (a0-a1)*(b0k-b1k)*0.5
    # That's 2 terms per k, 8 k's = 16 terms. Same as naive.
    #
    # What if we group k's? For k-pair (2s, 2s+1):
    # C[i,2s] = a0*b0(2s) + a1*b1(2s)
    # C[i,2s+1] = a0*b0(2s+1) + a1*b1(2s+1)
    # Using Strassen on this 2x2x2: 7 terms per pair, 4 pairs = 28.
    #
    # Let me just try rank 28 total (14 per block) with a direct search.
    # Actually, let me first try the straightforward 56-rank Strassen and
    # then attempt to reduce.
    
    # Build Strassen 2x2x2 local factors
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # Standard Strassen for 2x2x2
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True
    
    # First, build the 56-rank Strassen solution as baseline
    total_rank = 56
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)
    
    for j_pair in range(2):  # j in {0,1} or {2,3}
        for k_pair in range(4):  # k-pair index
            base_r = (j_pair * 4 + k_pair) * 7
            
            # A mapping: for j_pair=0: A cols 0,1; j_pair=1: A cols 2,3
            if j_pair == 0:
                a_map = [0, 1, 4, 5]
            else:
                a_map = [2, 3, 6, 7]
            
            # B mapping: for j_pair=0: B rows 0,1; j_pair=1: B rows 2,3
            k0 = k_pair * 2
            if j_pair == 0:
                b_map = [0*8 + k0, 0*8 + k0+1, 1*8 + k0, 1*8 + k0+1]
            else:
                b_map = [2*8 + k0, 2*8 + k0+1, 3*8 + k0, 3*8 + k0+1]
            
            # C mapping: C[0,k0], C[0,k0+1], C[1,k0], C[1,k0+1]
            c_map = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if not verify(U, V, W):
        # Fallback to naive 64
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = total_rank
    
    # Now try to reduce rank by finding linear dependencies
    # Flatten tensor and check rank
    tensor_size = 8 * 32 * 16  # 4096
    T = np.zeros((total_rank, tensor_size), dtype=float)
    for r in range(total_rank):
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    idx = i * 32 * 16 + j * 16 + k
                    T[r, idx] = float(U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r])
    
    try:
        rank_T = np.linalg.matrix_rank(T, tol=1e-10)
    except:
        rank_T = total_rank
    
    if rank_T < total_rank and time.time() < deadline - 5:
        # Try to find a lower-rank decomposition using null space
        # This is complex; skip for now and return 56
        pass
    
    # Try a direct rank-28 construction:
    # For each j-pair (2 blocks), use 14 terms.
    # 14 = 7 + 7, but with shared A-combinations across k-pairs.
    # 
    # Key insight: The 7 Strassen A-combinations are FIXED for a given j-pair.
    # They don't depend on k. So we can define 7 U-rows per j-pair.
    # Then for each k-pair (4 pairs), we need 7 V-rows and 7 W-rows.
    # But the U-rows are shared! So total U-rows = 2 j-pairs * 7 = 14.
    # Total V-rows = 2 j-pairs * 4 k-pairs * 7 = 56... no, that's per-term.
    #
    # Actually in the tensor decomposition, each term r has U[:,r], V[:,r], W[:,r].
    # The "rank" is the number of terms. We can't share U-rows across terms
    # in the standard formulation - each term is independent.
    #
    # BUT we can: if two terms have the same U-row, we can merge them by
    # taking linear combinations of V and W rows. This is the key!
    #
    # In the 56-rank solution, terms with the same (j_pair, l) have the same
    # U-pattern. There are 2*7=14 distinct U-patterns, each used 4 times
    # (for 4 k-pairs). We can merge the 4 terms with the same U-pattern into
    # a single term by combining V and W appropriately... but that changes
    # the tensor structure.
    #
    # Actually, merging doesn't work directly because the W patterns differ.
    # Let me just return the verified 56-rank solution.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
