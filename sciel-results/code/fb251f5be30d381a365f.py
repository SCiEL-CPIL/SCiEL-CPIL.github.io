import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Build the rank-36 Strassen-based decomposition
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if k_b != k_c or i_a != i_c:
                        s = np.dot(U[a], V[b] * W[c])
                        if abs(s) > tol:
                            return False
        return True

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank

    # Try to reduce rank by finding linear dependencies
    # and using local search to find a lower-rank decomposition
    if time.time() < deadline - 10:
        # Try rank 32: use a different blocking strategy
        # Split 4 into 2+2 but try to share factors
        # Alternative: try a direct rank-r search for r < 36
        
        # First, try to see if we can merge the two 2x2x1 blocks (rank 4 each = 8 total)
        # with parts of the Strassen blocks to save rank
        
        # Try constructing a 2x2x5 block with rank less than 18
        # For 2x2x5, try rank 10, 11, 12, ...
        
        # Let's try a parametric approach for 2x2xp
        # The 2x2x5 tensor: A is 2x2, B is 2x5, C is 2x5
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta
        
        # Try to find a rank-10 decomposition for 2x2x5
        # Using the structure: split B's 5 columns into groups
        
        # Actually, let's try a different approach: 
        # Use the fact that we can combine the two 2x2x5 subproblems
        # by allowing cross-terms between j={0,1} and j={2,3}
        
        # Try rank 32: use 4 Strassen-like blocks of rank 8 each
        # For a 2x2x5 with rank 8, we need a specific construction
        
        # Known: rank of M_{2,2,5} 
        # Let's try to construct it explicitly
        
        # For 2x2xp, one known construction uses rank 2p+3 for p>=2
        # But Strassen for p=2 gives 7, so for p=5 maybe 2*5+3=13? No...
        
        # Let me try a direct optimization approach
        # Start with rank 36 and try to reduce by finding null space vectors
        
        # Alternative: try a completely different decomposition
        # Use the fact that 2x4x5 can be seen as (2x2x5) x 2 in the j dimension
        # but with potential sharing
        
        # Let's try: can we do 2x4x5 with rank 32?
        # That would be 16 per 2x2x5 block, or some asymmetric split
        
        # Try: one 2x2x5 with rank 14 and one with rank 18 = 32
        # Or: both with rank 16 = 32
        
        # For 2x2x5 with rank 16:
        # Start from the 18-rank solution and try to eliminate 2 columns
        
        # Let's try a local search: perturb the rank-36 solution and project
        # to lower rank
        
        # Simpler approach: try all ways to split 5 into a+b where a+b=5
        # and use rank(2x2xa) + rank(2x2xb) for each half
        
        # rank(2x2x2)=7, rank(2x2x3)=?, rank(2x2x4)=?
        # If rank(2x2x3)=5 and rank(2x2x4)=7, then 7+5+7+5=24 per pair? No...
        
        # Let me just try to find if rank 32 is achievable with a different construction
        # by searching over small integer/half-integer patterns
        
        # Try: use 4 blocks of 2x2x2 (rank 7 each) = 28, plus 4 more for the k=4 column
        # But that's what we already have (28 + 8 = 36)
        
        # What if we can do the k=4 column with only 2 rank terms instead of 4?
        # The k=4 part: for each i, sum_j A[i,j]*B[j,4] = C[i,4]
        # This is 2x4x1 = rank 4 (schoolbook) or can it be less?
        # 2x4x1: A is 2x4, B is 4x1, C is 2x1
        # This is just a matrix-vector product, rank = min(2*4, 4*1, 2*1) = 2? 
        # No, the tensor rank of a 2x4x1 multiplication is 4 (you need 4 terms)
        # Actually for 2x4x1: C[i] = sum_j A[i,j] B[j]
        # This requires at least 4 rank-1 terms since B has 4 independent entries
        # So rank(2x4x1) = 4
        
        # So the minimum for the k=4 column alone is 4 per i-block, total 8.
        # And the k=0,1,2,3 part is 2x4x4.
        
        # 2x4x4: split into 2x2x4 + 2x2x4
        # 2x2x4: can we do better than 7+4=11?
        # rank(2x2x4): known to be 6? Let me check.
        # Actually for 2x2xp, I think rank = 2p-2 for p>=2 is not right.
        # rank(2x2x2)=7, rank(2x2x3)=? 
        
        # Let me just try to verify if our rank-36 is optimal or if 32 works
        # by attempting a rank-32 construction
        
        # Try: 2x4x5 with a single unified Strassen-like approach
        # This is getting complex. Let's try a numerical approach.
        
        # Try to find a rank-32 decomposition by optimization
        if time.time() < deadline - 20:
            # Try random initialization and gradient-free search for rank 32
            # This is unlikely to work well, but let's try a simple approach
            
            # Actually, let's try a smarter approach:
            # The current solution has 36 columns. Check if any 4 columns can be
            # eliminated (i.e., are linear combinations of others in a way that
            # preserves the tensor equation)
            
            # For tensor decomposition, rank reduction is hard. Let's try
            # a different structural decomposition instead.
            
            # New idea: instead of splitting j into {0,1} and {2,3},
            # split into {0,2} and {1,3} or other pairings
            
            # Try all 3 ways to split 4 elements into two pairs:
            # {0,1}|{2,3}, {0,2}|{1,3}, {0,3}|{1,2}
            
            for pairing in [([0,1],[2,3]), ([0,2],[1,3]), ([0,3],[1,2])]:
                if time.time() > deadline - 15:
                    break
                j1, j2 = pairing
                U2 = np.zeros((nm, 36), dtype=np.float64)
                V2 = np.zeros((mp, 36), dtype=np.float64)
                W2 = np.zeros((pn, 36), dtype=np.float64)
                
                # For each pair, do 2x2x5 with Strassen
                # j1 gets columns j1[0], j1[1]; j2 gets columns j2[0], j2[1]
                
                col2 = 0
                # For pair j1: i=0,1; j=j1[0],j1[1]; k=0..4
                # 2x2x2 for k=0,1
                for k_start in [0, 2, 4]:
                    k_end = min(k_start + 2, 5)
                    if k_end - k_start == 2:
                        # 2x2x2 Strassen
                        strassen_patterns = [
                            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                        ]
                        for u_pat, v_pat, w_pat in strassen_patterns:
                            for idx, val in enumerate(u_pat):
                                if val != 0:
                                    i_local = idx // 2
                                    j_local = idx % 2
                                    i_global = i_local
                                    j_global = j1[j_local]
                                    U2[i_global * m + j_global, col2] = val
                            for idx, val in enumerate(v_pat):
                                if val != 0:
                                    j_local = idx // 2
                                    k_local = idx % 2
                                    j_global = j1[j_local]
                                    k_global = k_start + k_local
                                    V2[j_global * p + k_global, col2] = val
                            for idx, val in enumerate(w_pat):
                                if val != 0:
                                    i_local = idx // 2
                                    k_local = idx % 2
                                    i_global = i_local
                                    k_global = k_start + k_local
                                    W2[k_global * n + i_global, col2] = val
                            col2 += 1
                    else:
                        # 2x2x1 schoolbook
                        for idx in range(4):
                            i_local = idx // 2
                            j_local = idx % 2
                            i_global = i_local
                            j_global = j1[j_local]
                            k_global = k_start
                            U2[i_global * m + j_global, col2] = 1
                            V2[j_global * p + k_global, col2] = 1
                            W2[k_global * n + i_global, col2] = 1
                            col2 += 1
                
                # For pair j2: i=0,1; j=j2[0],j2[1]; k=0..4
                for k_start in [0, 2, 4]:
                    k_end = min(k_start + 2, 5)
                    if k_end - k_start == 2:
                        strassen_patterns = [
                            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                        ]
                        for u_pat, v_pat, w_pat in strassen_patterns:
                            for idx, val in enumerate(u_pat):
                                if val != 0:
                                    i_local = idx // 2
                                    j_local = idx % 2
                                    i_global = i_local
                                    j_global = j2[j_local]
                                    U2[i_global * m + j_global, col2] = val
                            for idx, val in enumerate(v_pat):
                                if val != 0:
                                    j_local = idx // 2
                                    k_local = idx % 2
                                    j_global = j2[j_local]
                                    k_global = k_start + k_local
                                    V2[j_global * p + k_global, col2] = val
                            for idx, val in enumerate(w_pat):
                                if val != 0:
                                    i_local = idx // 2
                                    k_local = idx % 2
                                    i_global = i_local
                                    k_global = k_start + k_local
                                    W2[k_global * n + i_global, col2] = val
                            col2 += 1
                    else:
                        for idx in range(4):
                            i_local = idx // 2
                            j_local = idx % 2
                            i_global = i_local
                            j_global = j2[j_local]
                            k_global = k_start
                            U2[i_global * m + j_global, col2] = 1
                            V2[j_global * p + k_global, col2] = 1
                            W2[k_global * n + i_global, col2] = 1
                            col2 += 1
                
                if col2 < best_rank and verify_decomposition(U2[:, :col2], V2[:, :col2], W2[:, :col2]):
                    U2 = U2[:, :col2]
                    V2 = V2[:, :col2]
                    W2 = W2[:, :col2]
                    U_list = [[int(round(U2[i, r] * 2)) / 2 for r in range(col2)] for i in range(nm)]
                    V_list = [[int(round(V2[i, r] * 2)) / 2 for r in range(col2)] for i in range(mp)]
                    W_list = [[int(round(W2[i, r] * 2)) / 2 for r in range(col2)] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = col2

    # Final: try to verify and return best
    if best_rank == 36:
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
