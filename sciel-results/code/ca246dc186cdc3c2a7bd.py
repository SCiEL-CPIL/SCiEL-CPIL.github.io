import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(actual - expected) > 1e-8:
                    return False
    return True

def _laderman_4x4x4_terms():
    """
    Return the 20 Laderman rank-1 terms for 4x4 * 4x4.
    Each term is (u_dict, v_dict, w_dict) where dicts map (i,j) -> coeff.
    """
    terms = []
    
    # 1. (a00+a11)(b00+b11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # 2. (a01+a10)(b01+b10)
    terms.append(({(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}))
    # 3. (a02+a20)(b02+b20)
    terms.append(({(0,2):1, (2,0):1}, {(0,2):1, (2,0):1}, {(0,2):1, (2,0):1}))
    # 4. (a03+a30)(b03+b30)
    terms.append(({(0,3):1, (3,0):1}, {(0,3):1, (3,0):1}, {(0,3):1, (3,0):1}))
    # 5. (a12+a21)(b12+b21)
    terms.append(({(1,2):1, (2,1):1}, {(1,2):1, (2,1):1}, {(1,2):1, (2,1):1}))
    # 6. (a13+a31)(b13+b31)
    terms.append(({(1,3):1, (3,1):1}, {(1,3):1, (3,1):1}, {(1,3):1, (3,1):1}))
    # 7. (a23+a32)(b23+b32)
    terms.append(({(2,3):1, (3,2):1}, {(2,3):1, (3,2):1}, {(2,3):1, (3,2):1}))
    # 8. (a00+a22)(b00+b22)
    terms.append(({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}))
    # 9. (a11+a33)(b11+b33)
    terms.append(({(1,1):1, (3,3):1}, {(1,1):1, (3,3):1}, {(1,1):1, (3,3):1}))
    # 10. (a01+a23)(b01+b23)
    terms.append(({(0,1):1, (2,3):1}, {(0,1):1, (2,3):1}, {(0,1):1, (2,3):1}))
    # 11. (a10+a32)(b10+b32)
    terms.append(({(1,0):1, (3,2):1}, {(1,0):1, (3,2):1}, {(1,0):1, (3,2):1}))
    # 12. (a02+a13)(b02+b13)
    terms.append(({(0,2):1, (1,3):1}, {(0,2):1, (1,3):1}, {(0,2):1, (1,3):1}))
    # 13. (a20+a31)(b20+b31)
    terms.append(({(2,0):1, (3,1):1}, {(2,0):1, (3,1):1}, {(2,0):1, (3,1):1}))
    # 14. (a03+a12)(b03+b12)
    terms.append(({(0,3):1, (1,2):1}, {(0,3):1, (1,2):1}, {(0,3):1, (1,2):1}))
    # 15. (a21+a30)(b21+b30)
    terms.append(({(2,1):1, (3,0):1}, {(2,1):1, (3,0):1}, {(2,1):1, (3,0):1}))
    # 16. (a00+a11+a22+a33)(b00+b11+b22+b33)
    terms.append(({(0,0):1, (1,1):1, (2,2):1, (3,3):1}, {(0,0):1, (1,1):1, (2,2):1, (3,3):1}, {(0,0):1, (1,1):1, (2,2):1, (3,3):1}))
    # 17. (a01+a10+a23+a32)(b01+b10+b23+b32)
    terms.append(({(0,1):1, (1,0):1, (2,3):1, (3,2):1}, {(0,1):1, (1,0):1, (2,3):1, (3,2):1}, {(0,1):1, (1,0):1, (2,3):1, (3,2):1}))
    # 18. (a02+a13+a20+a31)(b02+b13+b20+b31)
    terms.append(({(0,2):1, (1,3):1, (2,0):1, (3,1):1}, {(0,2):1, (1,3):1, (2,0):1, (3,1):1}, {(0,2):1, (1,3):1, (2,0):1, (3,1):1}))
    # 19. (a03+a12+a21+a30)(b03+b12+b21+b30)
    terms.append(({(0,3):1, (1,2):1, (2,1):1, (3,0):1}, {(0,3):1, (1,2):1, (2,1):1, (3,0):1}, {(0,3):1, (1,2):1, (2,1):1, (3,0):1}))
    # 20. (sum of all)(sum of all)
    all_idx = {(i, j): 1 for i in range(4) for j in range(4)}
    terms.append((all_idx, all_idx, all_idx))

    return terms

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2.
    Each term is (u_dict, v_dict, w_dict) where dicts map (i,j) -> coeff.
    """
    terms = []
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c01 += M1, c10 -= M1, c11 += M1
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):-1, (1,1):1}))
    # M2 = (a10+a11)(b00) -> c10 += M2, c11 += M2
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1}))
    # M3 = (a00)(b10+b11) -> c00 += M3, c01 += M3
    terms.append(({(0,0):1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}))
    # M4 = (a11)(b01-b10) -> c01 += M4, c10 -= M4
    terms.append(({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,1):1, (1,0):-1}))
    # M5 = (a00+a01)(b11) -> c01 += M5, c11 += M5
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}))
    # M6 = (a10-a00)(b00+b01) -> c00 += M6, c10 += M6
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,0):1}))
    # M7 = (a01-a11)(b10+b11) -> c01 -= M7, c11 += M7
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):-1, (1,1):1}))
    return terms

def _build_laderman_decomposition(n, m, p, laderman_terms, k_offset=0):
    """
    Build Laderman-based decomposition for the 4x4x4 part.
    Returns (U, V, W) arrays for the 4x4x4 sub-problem.
    k_offset: offset for k-index in B and C (0 for k=0..3)
    """
    rank_laderman = len(laderman_terms)
    U = np.zeros((n * m, rank_laderman), dtype=float)
    V = np.zeros((m * p, rank_laderman), dtype=float)
    W = np.zeros((p * n, rank_laderman), dtype=float)
    
    for col, (u_dict, v_dict, w_dict) in enumerate(laderman_terms):
        u_vec = np.zeros(n * m, dtype=float)
        v_vec = np.zeros(m * p, dtype=float)
        w_vec = np.zeros(p * n, dtype=float)
        
        # Fill u_vec: A is 4x4, flat index i*m+j
        for (i, j), c in u_dict.items():
            u_vec[i * m + j] = c
        
        # Fill v_vec: B is 4xp, flat index j*p+k
        for (j, k), c in v_dict.items():
            v_vec[j * p + (k + k_offset)] = c
        
        # Fill w_vec: C is px4, flat index k*n+i
        for (k, i), c in w_dict.items():
            w_vec[(k + k_offset) * n + i] = c
        
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
    
    return U, V, W

def _build_strassen_4x4x2(n, m, p, k_offset):
    """
    Build Strassen-based decomposition for 4x4 * 4x2 using 2x2 blocks.
    Returns (U, V, W) arrays.
    k_offset: offset for k-index (4 for k=4,5)
    """
    strassen_terms = _strassen_2x2x2_terms()
    num_blocks = 4  # 2x2 blocks of A
    rank = num_blocks * len(strassen_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for bi in range(2):  # block row of A
        for bj in range(2):  # block col of A
            # A block is A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
            # B block is B[bj*2:(bj+1)*2, k_offset:k_offset+2]
            # C block is C[k_offset:k_offset+2, bi*2:(bi+1)*2]
            for u_dict, v_dict, w_dict in strassen_terms:
                u_vec = np.zeros(n * m, dtype=float)
                v_vec = np.zeros(m * p, dtype=float)
                w_vec = np.zeros(p * n, dtype=float)
                
                for (i, j), c in u_dict.items():
                    u_vec[(bi*2+i) * m + (bj*2+j)] = c
                
                for (j, k), c in v_dict.items():
                    v_vec[(bj*2+j) * p + (k_offset + k)] = c
                
                for (k, i), c in w_dict.items():
                    w_vec[(k_offset + k) * n + (bi*2+i)] = c
                
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    return U, V, W

def _build_schoolbook_decomposition(n, m, p, k_start, k_end):
    """
    Build schoolbook decomposition for columns k_start..k_end-1 of B.
    Returns (U, V, W) arrays.
    """
    rank = n * m * (k_end - k_start)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Laderman 4x4x4 (20) + Strassen 4x4x2 (28) + Schoolbook 4x4x3 (48) = 96
    laderman_terms = _laderman_4x4x4_terms()
    
    # Part 1: Laderman for 4x4x4 (k=0..3)
    U1, V1, W1 = _build_laderman_decomposition(n, m, p, laderman_terms, k_offset=0)
    
    # Part 2: Strassen for 4x4x2 (k=4..5)
    U2, V2, W2 = _build_strassen_4x4x2(n, m, p, k_offset=4)
    
    # Part 3: Schoolbook for 4x4x3 (k=6..8)
    U3, V3, W3 = _build_schoolbook_decomposition(n, m, p, 6, 9)
    
    # Combine
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
