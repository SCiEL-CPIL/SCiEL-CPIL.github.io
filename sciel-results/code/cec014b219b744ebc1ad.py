"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Try multiple block partition strategies to find minimum rank
    best_result = None
    best_rank = float('inf')
    
    def build_decomposition(j_pairs, j_naive, k_pairs):
        """Build decomposition given j-pair grouping and k-pair grouping."""
        rank = 7 * len(j_pairs) * len(k_pairs) + 4 * len(j_naive) * len(k_pairs)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1)][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] -= 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            
            # 1: A00 B00 -> C00
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

        # Apply Strassen blocks
        for j_pair in j_pairs:
            for k_pair in k_pairs:
                add_strassen_block(0, j_pair[0], k_pair[0])
        
        # Apply naive blocks
        for j_naive_idx in j_naive:
            for k_pair in k_pairs:
                add_naive_2112_block(0, j_naive_idx, k_pair[0])
        
        if col != rank:
            return None
        
        # Verify decomposition
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
                    
        T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
            return None
        
        return {"U": U, "V": V, "W": W}
    
    # Strategy 1: Original - j_pairs=(0,1),(2,3), j_naive=4, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        result = build_decomposition([(0,1),(2,3)], [4], [(0,1),(2,3),(4,5)])
        if result is not None:
            best_rank = 54
            best_result = result
    
    # Strategy 2: j_pairs=(0,1),(3,4), j_naive=2, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        result = build_decomposition([(0,1),(3,4)], [2], [(0,1),(2,3),(4,5)])
        if result is not None and 54 < best_rank:
            best_rank = 54
            best_result = result
    
    # Strategy 3: j_pairs=(0,2),(1,3), j_naive=4, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        result = build_decomposition([(0,2),(1,3)], [4], [(0,1),(2,3),(4,5)])
        if result is not None and 54 < best_rank:
            best_rank = 54
            best_result = result
    
    # Strategy 4: Try different k_pairings - (0,2),(1,3),(4,5)
    if time.time() < deadline:
        result = build_decomposition([(0,1),(2,3)], [4], [(0,2),(1,3),(4,5)])
        if result is not None and 54 < best_rank:
            best_rank = 54
            best_result = result
    
    # Strategy 5: Try k_pairs=(0,3),(1,4),(2,5)
    if time.time() < deadline:
        result = build_decomposition([(0,1),(2,3)], [4], [(0,3),(1,4),(2,5)])
        if result is not None and 54 < best_rank:
            best_rank = 54
            best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
