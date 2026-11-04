import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # We'll build the decomposition using a combination of Strassen and schoolbook
    # Strategy: Split the problem into manageable sub-problems
    
    # For 3x4x8, we can think of it as:
    # - 2x4x8 (using Strassen on the 2x2 parts) + 1x4x8 (schoolbook)
    # Or: 3x2x8 + 3x2x8 (splitting the m dimension)
    
    # Let's use a recursive approach but with better base cases
    
    terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: dict mapping (i,j) -> coeff for U[i*m+j]
        v_entries: dict mapping (j,k) -> coeff for V[j*p+k]  
        w_entries: dict mapping (k,i) -> coeff for W[k*n+i]
        """
        terms.append((u_entries, v_entries, w_entries))
    
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 subproblem at offset (i0,j0,k0)"""
        # M1 = (A11+A22)(B11+B22)
        add_term(
            {(i0,j0): 1, (i0+1,j0+1): 1},
            {(j0,k0): 1, (j0+1,k0+1): 1},
            {(k0,i0): 1, (k0+1,i0+1): 1}
        )
        # M2 = (A21+A22)B11
        add_term(
            {(i0+1,j0): 1, (i0+1,j0+1): 1},
            {(j0,k0): 1},
            {(k0,i0+1): 1, (k0+1,i0+1): -1}
        )
        # M3 = A11(B21-B11)
        add_term(
            {(i0,j0): 1},
            {(j0,k0+1): 1, (j0,k0): -1},
            {(k0,i0): 1, (k0+1,i0): 1}
        )
        # M4 = A22(B12-B22)
        add_term(
            {(i0+1,j0+1): 1},
            {(j0,k0+1): 1, (j0+1,k0+1): -1},
            {(k0,i0): 1, (k0+1,i0): 1}
        )
        # M5 = (A11+A12)B22
        add_term(
            {(i0,j0): 1, (i0,j0+1): 1},
            {(j0+1,k0+1): 1},
            {(k0+1,i0): -1, (k0,i0): 1}
        )
        # M6 = (A21-A11)(B11+B12)
        add_term(
            {(i0+1,j0): 1, (i0,j0): -1},
            {(j0,k0): 1, (j0,k0+1): 1},
            {(k0+1,i0+1): 1}
        )
        # M7 = (A12-A22)(B21+B22)
        add_term(
            {(i0,j0+1): 1, (i0+1,j0+1): -1},
            {(j0,k0+1): 1, (j0+1,k0+1): 1},
            {(k0,i0): 1}
        )
    
    def schoolbook(n_sub, m_sub, p_sub, i0, j0, k0):
        """Add schoolbook terms for n_sub x m_sub x p_sub at offset"""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    add_term(
                        {(i0+i, j0+j): 1},
                        {(j0+j, k0+k): 1},
                        {(k0+k, i0+i): 1}
                    )
    
    # Decompose 3x4x8
    # Split into 2x4x8 + 1x4x8
    # For 2x4x8: split m into 2+2, use Strassen for each 2x2x8, then split p
    # For 1x4x8: schoolbook
    
    # 2x4x8: split m into two 2x2x8 blocks
    # 2x2x8: split p into four 2x2x2 blocks, use Strassen for each
    for j_block in range(2):  # j = 0,1 and 2,3
        for k_block in range(4):  # k = 0-1, 2-3, 4-5, 6-7
            strassen_2x2x2(0, j_block*2, k_block*2)
    
    # 1x4x8: schoolbook
    schoolbook(1, 4, 8, 2, 0, 0)
    
    rank = len(terms)
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (i, j), c in u_entries.items():
            U[i * m + j][r] = c
        for (j, k), c in v_entries.items():
            V[j * p + k][r] = c
        for (k, i), c in w_entries.items():
            W[k * n + i][r] = c
    
    return {"U": U, "V": V, "W": W}
