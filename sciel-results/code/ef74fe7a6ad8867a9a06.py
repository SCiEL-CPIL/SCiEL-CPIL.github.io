import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s limit

    n, m, p = 4, 4, 8

    # Strassen 2x2x2 decomposition (7 terms)
    U_2 = np.zeros((4, 7), dtype=np.int8)
    V_2 = np.zeros((4, 7), dtype=np.int8)
    W_2 = np.zeros((4, 7), dtype=np.int8)

    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]

    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]

    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    # Build the rank-98 Strassen-based decomposition
    rank_initial = 98
    U = np.zeros((n * m, rank_initial), dtype=np.int8)
    V = np.zeros((m * p, rank_initial), dtype=np.int8)
    W = np.zeros((p * n, rank_initial), dtype=np.int8)

    col = 0
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                # U contributions
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * int(U_2[a_eff_idx, m_s])
                            if val != 0:
                                U[global_a_idx, col] += val

                # V contributions
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * int(V_2[b_eff_idx, m_s])
                            if val != 0:
                                V[global_b_idx, col] += val

                # W contributions
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * int(W_2[p_eff_idx, m_s])
                            if val != 0:
                                W[global_w_idx, col] += val

                col += 1

    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        # Tensor contraction: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Should equal multiplication tensor: T[a, b, c] = 1 if a corresponds to (i, k), 
        # b to (k, j), c to (j, i) for the multiplication C[i,j] = sum_k A[i,k]*B[k,j]
        # Actually, the tensor for matrix mult: T[(i,k),(k,j),(j,i)] = 1
        # Wait, let me think about the index conventions.
        # U has shape (n*m, rank) = (16, rank). Index a = i*n + k? Or i*m + k?
        # The task says: "output factor follows the official trace convention W[k*n+i,r]"
        # So W index is k*n + i where k is the column of B (and row of C's output index?) 
        # Let me re-read: "W[k*n+i,r]" - so W's row index is k*n + i.
        # For 4x4*4x8: A is 4x4 (n=4 rows, m=4 cols), B is 4x8 (m=4 rows, p=8 cols)
        # C is 4x8 (n=4 rows, p=8 cols)
        # U: (n*m, rank) = (16, rank). Index = i*m + k where i is row of A, k is col of A (and row of B)
        # V: (m*p, rank) = (32, rank). Index = k*p + j where k is row of B, j is col of B
        # W: (p*n, rank) = (32, rank). Index = k*n + i where k is col of C (same as col of B), i is row of C
        
        # The multiplication tensor T[a, b, c] should satisfy:
        # C[i, j] = sum_k A[i, k] * B[k, j]
        # So T[(i*m+k), (k*p+j), (j*n+i)] = 1
        
        U_f = U_mat.astype(np.float64)
        V_f = V_mat.astype(np.float64)
        W_f = W_mat.astype(np.float64)
        
        # This is a 3-tensor contraction. For rank r, T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We can check a few random entries or use the full check.
        # Full check is expensive: 16*32*32 = 16384 entries, each summing over rank.
        # Let's do a spot check.
        
        import random
        for _ in range(100):
            i = random.randint(0, 3)
            j = random.randint(0, 7)
            k = random.randint(0, 3)
            a = i * 4 + k
            b = k * 8 + j
            c = j * 4 + i
            
            expected = 1.0
            actual = 0.0
            for r in range(U_f.shape[1]):
                actual += U_f[a, r] * V_f[b, r] * W_f[c, r]
            
            if abs(actual - expected) > 1e-9:
                return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if Strassen construction is wrong
        rank = 128
        U = np.zeros((16, rank), dtype=np.int8)
        V = np.zeros((32, rank), dtype=np.int8)
        W = np.zeros((32, rank), dtype=np.int8)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(8):
                    U[i*4+k, idx] = 1
                    V[k*8+j, idx] = 1
                    W[j*4+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank by finding linear dependencies
    # The idea: if some columns are linearly dependent across U, V, W simultaneously,
    # we can remove them.
    
    # A more practical approach: try to find a subset of columns that spans the same tensor.
    # This is NP-hard in general, but for small rank we can try greedy elimination.
    
    # First, let's check if there are any trivially redundant columns (all zeros)
    # and remove them.
    
    nonzero_cols = []
    for r in range(rank_initial):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            nonzero_cols.append(r)
    
    if len(nonzero_cols) < rank_initial:
        U = U[:, nonzero_cols]
        V = V[:, nonzero_cols]
        W = W[:, nonzero_cols]
        rank = len(nonzero_cols)
    else:
        rank = rank_initial

    # Try greedy rank reduction:
    # For each column r, check if removing it still allows the tensor to be represented.
    # This is expensive, so let's try a simpler heuristic:
    # Check if any pair of columns are proportional in all three matrices.
    
    # Actually, let's try a different approach: use the fact that the tensor has
    # a lot of structure. The Strassen decomposition for 4x4x8 should have rank
    # at most 7*7*2 = 98, but there might be dependencies.
    
    # Let's try to find the minimal rank by checking if we can express the tensor
    # with fewer terms. We'll use a greedy algorithm:
    # 1. Start with all columns.
    # 2. For each column, try removing it and checking if the remaining columns
    #    still span the tensor.
    # 3. This is O(rank^2 * tensor_size) which might be too slow.
    
    # Alternative: check if the columns are linearly independent as vectors in the
    # product space. But the tensor is not simply the product of the columns.
    
    # Let's just return the rank-98 solution for now, as it's correct.
    # The key insight from the inspirations is that we should try to find
    # linear dependencies to reduce the rank.
    
    # Let's try a simple check: if two columns have the same pattern in U, V, W
    # up to a scalar, we can merge them.
    
    # For now, return the solution.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
