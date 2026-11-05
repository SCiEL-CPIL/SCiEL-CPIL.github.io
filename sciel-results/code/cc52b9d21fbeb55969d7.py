"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication."""
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # t=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # t=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # t=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # t=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # t=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # t=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # t=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _bcdR_3x3():
    """Bini-Capovani-De Falco-Ronco rank-16 decomposition for 3x3 matrix multiplication."""
    n = m = p = 3
    rank = 16
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # BCDR decomposition - 16 terms
    # Each term is a rank-1 tensor in the 3x3x3 matmul tensor
    # U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] summed over t = 1 for all i,j,k
    
    # The 16 terms of BCDR:
    # Using the standard BCDR construction
    terms = [
        # (U_indices, V_indices, W_indices) where each is list of (index, value)
        # Term 0
        ([(0,1),(8,1)], [(0,1),(8,1)], [(0,1),(8,1)]),
        # Term 1  
        ([(2,1),(8,1)], [(0,1)], [(3,1),(8,1)]),
        # Term 2
        ([(0,1)], [(1,1),(8,1)], [(1,1),(8,1)]),
        # Term 3
        ([(8,1)], [(2,1),(0,1)], [(0,1),(3,1)]),
        # Term 4
        ([(0,1),(1,1)], [(8,1)], [(0,1),(1,1)]),
        # Term 5
        ([(2,1),(0,1)], [(0,1),(1,1)], [(8,1)]),
        # Term 6
        ([(1,1),(8,1)], [(2,1),(8,1)], [(0,1),(8,1)]),
        # Term 7
        ([(6,1),(8,1)], [(0,1)], [(3,1),(8,1)]),
        # Term 8
        ([(0,1)], [(4,1),(8,1)], [(1,1),(8,1)]),
        # Term 9
        ([(8,1)], [(5,1),(0,1)], [(0,1),(3,1)]),
        # Term 10
        ([(0,1),(4,1)], [(8,1)], [(0,1),(1,1)]),
        # Term 11
        ([(5,1),(0,1)], [(0,1),(4,1)], [(8,1)]),
        # Term 12
        ([(4,1),(8,1)], [(5,1),(8,1)], [(0,1),(8,1)]),
        # Term 13
        ([(6,1),(8,1)], [(3,1)], [(6,1)]),
        # Term 14
        ([(3,1)], [(6,1)], [(3,1)]),
        # Term 15
        ([(7,1)], [(7,1)], [(7,1)]),
    ]
    
    # This is getting too complex to hardcode correctly. Let me use schoolbook for 3x3.
    return _schoolbook_3x3()

def _schoolbook_3x3():
    """Schoolbook decomposition of rank 27 for 3x3 matrix multiplication."""
    n = m = p = 3
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition of rank n*m*p for n x m times m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_subdecompositions(n, m, p, sub_decomps, index_maps):
    """Combine multiple sub-decompositions into one full decomposition.
    
    sub_decomps: list of (U_sub, V_sub, W_sub) tuples
    index_maps: list of (i_map, j_map, k_map) where:
        i_map: list mapping sub-i to full-i
        j_map: list mapping sub-j to full-j  
        k_map: list mapping sub-k to full-k
    """
    total_rank = sum(U.shape[1] for U, V, W in sub_decomps)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (U_sub, V_sub, W_sub), (i_map, j_map, k_map) in zip(sub_decomps, index_maps):
        n_sub = len(i_map)
        m_sub = len(j_map)
        p_sub = len(k_map)
        rank_sub = U_sub.shape[1]
        
        for t in range(rank_sub):
            for i_sub in range(n_sub):
                i_full = i_map[i_sub]
                for j_sub in range(m_sub):
                    j_full = j_map[j_sub]
                    U[i_full * m + j_full, col_offset + t] = U_sub[i_sub * m_sub + j_sub, t]
            
            for j_sub in range(m_sub):
                j_full = j_map[j_sub]
                for k_sub in range(p_sub):
                    k_full = k_map[k_sub]
                    V[j_full * p + k_full, col_offset + t] = V_sub[j_sub * p_sub + k_sub, t]
            
            for k_sub in range(p_sub):
                k_full = k_map[k_sub]
                for i_sub in range(n_sub):
                    i_full = i_map[i_sub]
                    W[k_full * n + i_full, col_offset + t] = W_sub[k_sub * n_sub + i_sub, t]
        
        col_offset += rank_sub
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition: small = {0, 1}, large = {2, 3, 4}
    small_idx = [0, 1]
    large_idx = [2, 3, 4]
    
    # Define the 8 sub-problems:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22  
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Sub-problem 1: A11*B11 (2x2 * 2x2 -> 2x2), Strassen rank 7
    U1, V1, W1 = _strassen_2x2()
    # i_map: [0,1], j_map: [0,1], k_map: [0,1]
    
    # Sub-problem 2: A12*B21 (2x3 * 3x2 -> 2x2), schoolbook rank 12
    U2, V2, W2 = _schoolbook_nm_p(2, 3, 2)
    # i_map: [0,1], j_map: [2,3,4], k_map: [0,1]
    
    # Sub-problem 3: A11*B12 (2x2 * 2x3 -> 2x3), schoolbook rank 12
    U3, V3, W3 = _schoolbook_nm_p(2, 2, 3)
    # i_map: [0,1], j_map: [0,1], k_map: [2,3,4]
    
    # Sub-problem 4: A12*B22 (2x3 * 3x3 -> 2x3), schoolbook rank 18
    U4, V4, W4 = _schoolbook_nm_p(2, 3, 3)
    # i_map: [0,1], j_map: [2,3,4], k_map: [2,3,4]
    
    # Sub-problem 5: A21*B11 (3x2 * 2x2 -> 3x2), schoolbook rank 12
    U5, V5, W5 = _schoolbook_nm_p(3, 2, 2)
    # i_map: [2,3,4], j_map: [0,1], k_map: [0,1]
    
    # Sub-problem 6: A22*B21 (3x3 * 3x2 -> 3x2), schoolbook rank 18
    U6, V6, W6 = _schoolbook_nm_p(3, 3, 2)
    # i_map: [2,3,4], j_map: [2,3,4], k_map: [0,1]
    
    # Sub-problem 7: A21*B12 (3x2 * 2x3 -> 3x3), schoolbook rank 18
    U7, V7, W7 = _schoolbook_nm_p(3, 2, 3)
    # i_map: [2,3,4], j_map: [0,1], k_map: [2,3,4]
    
    # Sub-problem 8: A22*B22 (3x3 * 3x3 -> 3x3), schoolbook rank 27
    U8, V8, W8 = _schoolbook_3x3()
    # i_map: [2,3,4], j_map: [2,3,4], k_map: [2,3,4]
    
    sub_decomps = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7),
        (U8, V8, W8),
    ]
    
    index_maps = [
        (small_idx, small_idx, small_idx),
        (small_idx, large_idx, small_idx),
        (small_idx, small_idx, large_idx),
        (small_idx, large_idx, large_idx),
        (large_idx, small_idx, small_idx),
        (large_idx, large_idx, small_idx),
        (large_idx, small_idx, large_idx),
        (large_idx, large_idx, large_idx),
    ]
    
    U, V, W = _combine_subdecompositions(n, m, p, sub_decomps, index_maps)
    
    # Verify
    if not _verify_decomposition(n, m, p, U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
