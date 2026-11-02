"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    # Use block decomposition: split rows into 2+1, cols into 2+1
    # This gives a structured decomposition with lower rank than schoolbook
    
    # Block structure:
    # A is 3x3: split into A22 (2x2), A21 (2x1), A12 (1x2), A11 (1x1)
    # B is 3x6: split into B22 (2x6), B12 (1x6)
    # C is 3x6: split into C22 (2x6), C21 (2x6), C12 (1x6), C11 (1x6)
    #
    # Terms:
    # C22 = A22 * B22 : 2x2x6 -> use Strassen for each 2x2x2 block (3 blocks of 7 terms = 21)
    # C21 = A21 * B12 : 2x1x6 -> 12 terms (schoolbook)
    # C12 = A12 * B22 : 1x2x6 -> 12 terms (schoolbook)
    # C11 = A11 * B12 : 1x1x6 -> 6 terms (schoolbook)
    # Total: 21 + 12 + 12 + 6 = 51
    
    terms = []
    
    # Helper: add a term with specified U, V, W entries
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add contribution to a term"""
        if len(terms) == 0:
            terms.append({})
        terms[-1][('U', u_idx)] = terms[-1].get(('U', u_idx), 0) + u_val
        terms[-1][('V', v_idx)] = terms[-1].get(('V', v_idx), 0) + v_val
        terms[-1][('W', w_idx)] = terms[-1].get(('W', w_idx), 0) + w_val
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block starting at row i0, col j0, output col k0"""
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Term 1: M1
        add_term(i0 * 3 + j0, 1, j0 * 6 + k0, 1, k0 * 3 + i0, 1)
        add_term((i0+1) * 3 + (j0+1), 1, (j0+1) * 6 + (k0+1), 1, (k0+1) * 3 + (i0+1), 1)
        add_term(i0 * 3 + (j0+1), 1, (j0+1) * 6 + k0, 1, k0 * 3 + (i0+1), 1)
        add_term((i0+1) * 3 + j0, 1, j0 * 6 + (k0+1), 1, (k0+1) * 3 + i0, 1)
        
        # Term 2: M2
        add_term((i0+1) * 3 + j0, 1, j0 * 6 + k0, 1, k0 * 3 + (i0+1), 1)
        add_term((i0+1) * 3 + (j0+1), 1, (j0+1) * 6 + k0, 1, k0 * 3 + (i0+1), 1)
        
        # Term 3: M3
        add_term(i0 * 3 + j0, 1, (j0+1) * 6 + k0, 1, k0 * 3 + (i0+1), 1)
        add_term(i0 * 3 + j0, -1, j0 * 6 + k0, -1, k0 * 3 + (i0+1), -1)
        
        # Term 4: M4
        add_term((i0+1) * 3 + (j0+1), 1, (j0+1) * 6 + (k0+1), 1, (k0+1) * 3 + (i0+1), 1)
        add_term((i0+1) * 3 + (j0+1), -1, j0 * 6 + (k0+1), -1, (k0+1) * 3 + (i0+1), -1)
        
        # Term 5: M5
        add_term(i0 * 3 + j0, 1, (j0+1) * 6 + (k0+1), 1, (k0+1) * 3 + i0, 1)
        add_term(i0 * 3 + (j0+1), 1, (j0+1) * 6 + (k0+1), 1, (k0+1) * 3 + i0, 1)
        
        # Term 6: M6
        add_term((i0+1) * 3 + j0, 1, j0 * 6 + k0, 1, k0 * 3 + i0, 1)
        add_term(i0 * 3 + j0, -1, j0 * 6 + k0, -1, k0 * 3 + i0, -1)
        add_term((i0+1) * 3 + j0, 1, (j0+1) * 6 + k0, 1, k0 * 3 + (i0+1), 1)
        add_term(i0 * 3 + j0, -1, (j0+1) * 6 + k0, -1, k0 * 3 + (i0+1), -1)
        
        # Term 7: M7
        add_term(i0 * 3 + (j0+1), 1, j0 * 6 + (k0+1), 1, (k0+1) * 3 + (i0+1), 1)
        add_term((i0+1) * 3 + (j0+1), -1, j0 * 6 + (k0+1), -1, (k0+1) * 3 + (i0+1), -1)
        add_term(i0 * 3 + (j0+1), 1, (j0+1) * 6 + k0, 1, k0 * 3 + i0, 1)
        add_term((i0+1) * 3 + (j0+1), -1, (j0+1) * 6 + k0, -1, k0 * 3 + i0, -1)
    
    # Block 1: A22 (rows 0,1; cols 0,1) * B22 (rows 0,1; cols 0-5) -> C22 (rows 0,1; cols 0-5)
    # Three 2x2x2 Strassen blocks for k=0,2,4
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_strassen_2x2x2(0, 0, 4)
    
    # Block 2: A21 (rows 0,1; col 2) * B12 (row 2; cols 0-5) -> C21 (rows 0,1; cols 0-5)
    # Schoolbook: 2*1*6 = 12 terms
    for i in [0, 1]:
        for k in range(6):
            add_term(i * 3 + 2, 1, 2 * 6 + k, 1, k * 3 + i, 1)
    
    # Block 3: A12 (row 2; cols 0,1) * B22 (rows 0,1; cols 0-5) -> C12 (row 2; cols 0-5)
    # Schoolbook: 1*2*6 = 12 terms
    for j in [0, 1]:
        for k in range(6):
            add_term(2 * 3 + j, 1, j * 6 + k, 1, k * 3 + 2, 1)
    
    # Block 4: A11 (row 2; col 2) * B12 (row 2; cols 0-5) -> C11 (row 2; cols 0-5)
    # Schoolbook: 1*1*6 = 6 terms
    for k in range(6):
        add_term(2 * 3 + 2, 1, 2 * 6 + k, 1, k * 3 + 2, 1)
    
    # Build U, V, W matrices from terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(terms):
        for key, val in term.items():
            if key[0] == 'U':
                U[key[1]][col] = val
            elif key[0] == 'V':
                V[key[1]][col] = val
            elif key[0] == 'W':
                W[key[1]][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
