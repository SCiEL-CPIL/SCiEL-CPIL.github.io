"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-reduced 3x3 decomposition."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 6, 6, 6
    
    # Build the rank-27 schoolbook decomposition for 3x3
    r3 = 27
    U33 = np.zeros((9, r3))
    V33 = np.zeros((9, r3))
    W33 = np.zeros((9, r3))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i*3+j, col] = 1
                V33[j*3+k, col] = 1
                W33[k*3+i, col] = 1
                col += 1
    
    # Strassen terms for 2x2 block multiplication
    strassen_terms = [
        # M0 = (A00 + A11) * (B00 + B11)
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        # M1 = (A10 + A11) * B00
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        # M2 = A00 * (B01 - B11)
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        # M3 = A11 * (B10 - B00)
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        # M4 = (A00 + A01) * B11
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        # M5 = (A10 - A00) * (B00 + B01)
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        # M6 = (A01 - A11) * (B10 + B11)
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    # Build the full 6x6 decomposition
    total_rank = 7 * r3
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        
        for local_col in range(r3):
            col = base_col + local_col
            i3 = local_col // 9
            rem = local_col % 9
            j3 = rem // 3
            k3 = rem % 3
            
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                u_row = (bi*3+i3)*6 + (bj*3+j3)
                U[u_row, col] += ca * U33[i3*3+j3, local_col]
            
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                v_row = (bk*3+j3)*6 + (bl*3+k3)
                V[v_row, col] += cb * V33[j3*3+k3, local_col]
            
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                w_row = (ck*3+k3)*6 + (ci*3+i3)
                W[w_row, col] += cc * W33[k3*3+i3, local_col]
    
    # Now perform rank reduction via SVD
    # The key insight: if we have U, V, W with rank r, we can check if there are
    # linear dependencies among the columns that allow us to reduce the rank.
    
    # Strategy: Use SVD on the factor matrices to find a lower-rank representation
    # that preserves the tensor decomposition exactly.
    
    # For a tensor decomposition C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c],
    # we can try to find a lower-rank representation by looking for linear dependencies.
    
    # One approach: if the columns of U, V, W are linearly dependent in a consistent way,
    # we can reduce the rank.
    
    # Let's try a different approach: use the fact that we can absorb linear combinations
    # of the rank-1 terms into each other if they are linearly dependent.
    
    # Actually, the most straightforward approach is to check if we can find a matrix P
    # such that U' = U @ P, V' = V @ P^{-1}, W' = W @ I (or similar) reduces the rank.
    
    # For now, let's try a simple SVD-based approach:
    # 1. Compute SVD of U: U = U_svd @ S @ Vt
    # 2. Keep only the significant singular values
    # 3. Adjust V and W accordingly
    
    # But this might not preserve exactness. Let me think differently.
    
    # Alternative: The Strassen decomposition has inherent structure. Let's check if
    # some of the 189 terms are linearly dependent.
    
    # For each term c, we have a rank-1 tensor T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # If sum_c alpha_c * T_c = 0 for some non-zero alpha, then we can reduce the rank.
    
    # This is a complex linear algebra problem. Let's try a simpler heuristic:
    # Check if removing certain terms still gives a valid decomposition.
    
    # Actually, let's try a more direct approach: use the SVD to find a lower-rank
    # approximation and then verify exactness.
    
    # For now, return the original rank-189 decomposition as a baseline.
    # In a future iteration, we can implement more sophisticated rank reduction.
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Time is running out, return what we have
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
