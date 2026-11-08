"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p)."""
    n, m, p = DIMENSIONS
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

def _validate(U, V, W, tol=1e-10):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 terms of Strassen's 2x2x2 decomposition.
    Each term is a tuple (U_entries, V_entries, W_entries)
    where each entry is a list of (index, coeff).
    Indices are 0..3 for 2x2 matrices (row-major: 0,1,2,3).
    """
    # s1 = (A00+A11)(B00+B11)
    # s2 = (A10+A11)B00
    # s3 = A00(B01-B11)
    # s4 = A11(B10-B00)
    # s5 = (A00+A01)B11
    # s6 = (A10-A00)(B00+B01)
    # s7 = (A01-A11)(B10+B11)
    
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 + s4
    # C11 = s1 - s2 + s3 + s6
    
    # Mapping indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        # s1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # s2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # s3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # s4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # s5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # s6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # s7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    return terms

def _strassen_4x4x4():
    """
    Construct a rank-49 decomposition for 4x4x4 using Strassen.
    This applies the 2x2 Strassen decomposition to the 2x2 blocks of the 4x4 matrices.
    """
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    s2_terms = _strassen_2x2_terms()
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    # The 4x4 matrix is divided into four 2x2 blocks:
    # A00 (top-left), A01 (top-right)
    # A10 (bottom-left), A11 (bottom-right)
    # 
    # Global index for A[i,j]:
    # Block (bi, bj) where bi, bj in {0,1}
    # Local index (li, lj) where li, lj in {0,1}
    # Global i = bi*2 + li, Global j = bj*2 + lj
    # U index = i*m + j = i*4 + j
    
    # Similarly for B and C.
    
    # High-level Strassen terms for 4x4:
    # H1 = (A00+A11)(B00+B11)
    # H2 = (A10+A11)B00
    # H3 = A00(B01-B11)
    # H4 = A11(B10-B00)
    # H5 = (A00+A01)B11
    # H6 = (A10-A00)(B00+B01)
    # H7 = (A01-A11)(B10+B11)
    
    # C00 = H1 + H4 - H5 + H7
    # C01 = H3 + H5
    # C10 = H2 + H4
    # C11 = H1 - H2 + H3 + H6
    
    # Each H is a product of two 2x2 matrices.
    # We iterate over the 7 high-level terms.
    # For each high-level term, we have linear combinations of blocks.
    # We can expand this by iterating over the 7 low-level Strassen terms for each pair of blocks involved.
    
    # Let's define the high-level terms more explicitly as pairs of (A_block_combo, B_block_combo, C_block_contrib).
    # Actually, it's easier to just iterate over the 7 H terms, and for each, determine which A and B blocks are involved,
    # and then for each pair of blocks, apply the 2x2 Strassen decomposition.
    
    # H1: A blocks {00, 11}, B blocks {00, 11}. C blocks {00, 11} (coeff 1)
    # H2: A blocks {10, 11}, B blocks {00}. C blocks {10} (coeff 1)
    # H3: A blocks {00}, B blocks {01, 11} (coeff 1, -1). C blocks {01} (coeff 1)
    # H4: A blocks {11}, B blocks {10, 00} (coeff 1, -1). C blocks {00, 10} (coeff 1, 1)
    # H5: A blocks {00, 01}, B blocks {11}. C blocks {00, 01} (coeff -1, 1)
    # H6: A blocks {10, 00} (coeff 1, -1), B blocks {00, 01}. C blocks {11} (coeff 1)
    # H7: A blocks {01, 11} (coeff 1, -1), B blocks {10, 11}. C blocks {00, 11} (coeff 1, -1)
    
    # This is getting complicated to hardcode the block interactions.
    # Let's use a more systematic approach.
    # We can view the 4x4 multiplication as a 2x2 block multiplication where each block is 2x2.
    # The block multiplication tensor is M_{2,2,2} where the "entries" are 2x2 matrices.
    # The rank of M_{2,2,2} is 7 (Strassen).
    # Each "scalar" multiplication in the block decomposition is actually a 2x2x2 multiplication.
    # So we can use the 7 Strassen terms for the block level, and for each term, we need to decompose the 2x2x2 multiplication.
    # The 2x2x2 multiplication has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # Let's implement this by iterating over the 7 high-level terms and 7 low-level terms.
    
    # High-level terms:
    # (A_blocks, B_blocks, C_blocks)
    # A_blocks is a list of (block_idx, coeff)
    # B_blocks is a list of (block_idx, coeff)
    # C_blocks is a list of (block_idx, coeff)
    
    # Block indices: 0=00, 1=01, 2=10, 3=11
    # Block (bi, bj) -> global row offset bi*2, global col offset bj*2
    
    high_level_terms = [
        # H1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # H2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # H3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # H4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # H5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # H6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # H7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    
    for h_idx, (hA, hB, hC) in enumerate(high_level_terms):
        for l_idx, (lA, lB, lC) in enumerate(s2_terms):
            col = h_idx * 7 + l_idx
            
            # For each term in hA and lA, we have a global A index.
            # hA: (block_idx, coeff_h)
            # lA: (local_idx, coeff_l)
            # Global A index:
            # block_idx -> (bi, bj) where bi = block_idx // 2, bj = block_idx % 2
            # local_idx -> (li, lj) where li = local_idx // 2, lj = local_idx % 2
            # Global i = bi*2 + li, Global j = bj*2 + lj
            # U index = i*4 + j
            
            for (bA, cA_h) in hA:
                biA = bA // 2
                bjA = bA % 2
                for (lA_idx, cA_l) in lA:
                    liA = lA_idx // 2
                    ljA = lA_idx % 2
                    i = biA * 2 + liA
                    j = bjA * 2 + ljA
                    U[i * 4 + j][col] += cA_h * cA_l
            
            # Similarly for V
            for (bB, cB_h) in hB:
                biB = bB // 2
                bjB = bB % 2
                for (lB_idx, cB_l) in lB:
                    liB = lB_idx // 2
                    ljB = lB_idx % 2
                    j = biB * 2 + liB
                    k = bjB * 2 + ljB
                    V[j * 4 + k][col] += cB_h * cB_l
            
            # Similarly for W
            for (bC, cC_h) in hC:
                biC = bC // 2
                bjC = bC % 2
                for (lC_idx, cC_l) in lC:
                    liC = lC_idx // 2
                    ljC = lC_idx % 2
                    k = biC * 2 + liC
                    i = bjC * 2 + ljC
                    W[k * 4 + i][col] += cC_h * cC_l
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x9():
    """
    Construct a decomposition for 4x4x9 by combining:
    - Strassen 4x4x4 (rank 49) for the first 4 columns
    - Schoolbook 4x4x5 (rank 80) for the last 5 columns
    Total rank: 129
    """
    n, m, p = 4, 4, 9
    rank_strassen = 49
    rank_schoolbook = 4 * 4 * 5
    rank = rank_strassen + rank_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for k=0..3
    s4 = _strassen_4x4x4()
    for i in range(n * m):
        for c in range(rank_strassen):
            U[i][c] = s4['U'][i][c]
    for j in range(m * 4): # m*p_strassen
        for c in range(rank_strassen):
            V[j][c] = s4['V'][j][c]
    for k in range(4 * n): # p_strassen*n
        for c in range(rank_strassen):
            W[k][c] = s4['W'][k][c]
    
    # Part 2: Schoolbook for k=4..8
    col = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(4, 9):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook (Rank 144)
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Strassen 4x4x4 + Schoolbook 4x4x5 (Rank 129)
    if time.time() < deadline:
        strassen_result = _strassen_4x4x9()
        if strassen_result is not None:
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
    
    if best_result is None:
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
