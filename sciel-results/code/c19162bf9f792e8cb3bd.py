import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We construct a rank 28 decomposition for 2x3x5
    # By splitting the problem into sub-problems:
    # 1. C1 = A * B1 (2x3 * 3x3) -> Rank 17
    #    - A1 * B11 (2x2 * 2x3) -> Rank 11
    #      - A1 * B111 (2x2 * 2x2) -> Rank 7 (Strassen)
    #      - A1 * B112 (2x2 * 2x1) -> Rank 4 (Schoolbook)
    #    - A2 * B12 (2x1 * 1x3) -> Rank 6 (Schoolbook)
    # 2. C2 = A * B2 (2x3 * 3x2) -> Rank 11
    #    - A1 * B21 (2x2 * 2x2) -> Rank 7 (Strassen)
    #    - A2 * B22 (2x1 * 1x2) -> Rank 4 (Schoolbook)
    
    total_rank = 7 + 4 + 6 + 7 + 4
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    rank_offset = 0
    
    # Helper to add Strassen decomposition
    # local_i: list of global i indices for A rows
    # local_j: list of global j indices for A cols / B rows
    # local_k: list of global k indices for B cols / C cols
    # For 2x2x2, we need 2 i's, 2 j's, 2 k's
    def add_strassen(i_idx, j_idx, k_idx, offset):
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        i0, i1 = i_idx
        j0, j1 = j_idx
        k0, k1 = k_idx
        
        # Term 1
        U[i0*m + j0, offset] = 1
        U[i1*m + j1, offset] = 1
        V[j0*p + k0, offset] = 1
        V[j1*p + k1, offset] = 1
        W[k0*n + i0, offset] = 1
        W[k1*n + i1, offset] = 1
        offset += 1
        
        # Term 2
        U[i1*m + j0, offset] = 1
        U[i1*m + j1, offset] = 1
        V[j0*p + k0, offset] = 1
        W[k0*n + i1, offset] = 1
        W[k1*n + i1, offset] = -1
        offset += 1
        
        # Term 3
        U[i0*m + j0, offset] = 1
        V[j0*p + k1, offset] = 1
        V[j1*p + k1, offset] = -1
        W[k0*n + i0, offset] = 1
        W[k1*n + i0, offset] = 1
        offset += 1
        
        # Term 4
        U[i1*m + j1, offset] = 1
        V[j1*p + k0, offset] = 1
        V[j0*p + k0, offset] = -1
        W[k0*n + i0, offset] = 1
        W[k0*n + i1, offset] = 1
        offset += 1
        
        # Term 5
        U[i0*m + j0, offset] = 1
        U[i0*m + j1, offset] = 1
        V[j1*p + k1, offset] = 1
        W[k0*n + i0, offset] = -1
        W[k1*n + i0, offset] = 1
        offset += 1
        
        # Term 6
        U[i1*m + j0, offset] = 1
        U[i0*m + j0, offset] = -1
        V[j0*p + k0, offset] = 1
        V[j0*p + k1, offset] = 1
        W[k1*n + i1, offset] = 1
        offset += 1
        
        # Term 7
        U[i0*m + j1, offset] = 1
        U[i1*m + j1, offset] = -1
        V[j1*p + k0, offset] = 1
        V[j1*p + k1, offset] = 1
        W[k0*n + i0, offset] = 1
        offset += 1
        
        return offset

    # Helper to add Schoolbook decomposition
    # i_idx: list of global i indices
    # j_idx: list of global j indices
    # k_idx: list of global k indices
    def add_schoolbook(i_idx, j_idx, k_idx, offset):
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    U[i*m + j, offset] = 1
                    V[j*p + k, offset] = 1
                    W[k*n + i, offset] = 1
                    offset += 1
        return offset

    # 1. C1 = A * B1 (2x3 * 3x3)
    # B1 uses k in {0,1,2}
    # A uses j in {0,1,2}
    
    # 1a. A1 * B11 (2x2 * 2x3)
    # A1 uses i in {0,1}, j in {0,1}
    # B11 uses j in {0,1}, k in {0,1,2}
    
    # 1a1. A1 * B111 (2x2 * 2x2)
    # B111 uses k in {0,1}
    rank_offset = add_strassen([0, 1], [0, 1], [0, 1], rank_offset)
    
    # 1a2. A1 * B112 (2x2 * 2x1)
    # B112 uses k in {2}
    rank_offset = add_schoolbook([0, 1], [0, 1], [2], rank_offset)
    
    # 1b. A2 * B12 (2x1 * 1x3)
    # A2 uses j in {2}
    # B12 uses j in {2}, k in {0,1,2}
    rank_offset = add_schoolbook([0, 1], [2], [0, 1, 2], rank_offset)
    
    # 2. C2 = A * B2 (2x3 * 3x2)
    # B2 uses k in {3,4}
    
    # 2a. A1 * B21 (2x2 * 2x2)
    # A1 uses j in {0,1}
    # B21 uses j in {0,1}, k in {3,4}
    rank_offset = add_strassen([0, 1], [0, 1], [3, 4], rank_offset)
    
    # 2b. A2 * B22 (2x1 * 1x2)
    # A2 uses j in {2}
    # B22 uses j in {2}, k in {3,4}
    rank_offset = add_schoolbook([0, 1], [2], [3, 4], rank_offset)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
