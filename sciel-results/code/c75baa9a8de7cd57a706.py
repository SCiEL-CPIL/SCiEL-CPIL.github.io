"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses block-structured decomposition with Strassen (rank 7) for 2x2 blocks and direct methods
    for mixed blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    U = []  # Will be (n*m, rank) = (12, rank)
    V = []  # Will be (m*p, rank) = (24, rank)
    W = []  # Will be (p*n, rank) = (18, rank)
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a single rank-1 term from dictionaries of row:val."""
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        for r, val in u_dict.items():
            u_row[r] = val
        for r, val in v_dict.items():
            v_row[r] = val
        for r, val in w_dict.items():
            w_row[r] = val
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
    
    # Strassen 2x2*2x2 decomposition
    def add_strassen(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        A00, A01, A10, A11: U row indices for A[0,0], A[0,1], A[1,0], A[1,1]
        B00, B01, B10, B11: V row indices for B[0,0], B[0,1], B[1,0], B[1,1]
        C00, C01, C10, C11: W row indices for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        # M1 = (A00+A11)(B00+B11)
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1})
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C11: 1})
        
        # M2 = (A10+A11)B00
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C11: -1})
        
        # M3 = A00(B10-B11)
        add_term({A00: 1}, {B10: 1, B11: -1}, {C01: 1})
        add_term({A00: 1}, {B10: 1, B11: -1}, {C11: 1})
        
        # M4 = A11(B01-B00)
        add_term({A11: 1}, {B01: 1, B00: -1}, {C00: -1})
        add_term({A11: 1}, {B01: 1, B00: -1}, {C10: 1})
        
        # M5 = (A00+A01)B11
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C01: 1})
        
        # M6 = (A10-A00)(B00+B01)
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        
        # M7 = (A01-A11)(B10+B11)
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: -1})
    
    # 1x2 * 2x2 multiplication (rank 4)
    def add_1x2_2x2(A0, A1, B00, B01, B10, B11, C0, C1):
        """
        Add decomposition for 1x2 * 2x2.
        A0, A1: U row indices for A[0,0], A[0,1]
        B00, B01, B10, B11: V row indices for B[0,0], B[0,1], B[1,0], B[1,1]
        C0, C1: W row indices for C[0,0], C[0,1]
        
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        """
        add_term({A0: 1}, {B00: 1}, {C0: 1})
        add_term({A1: 1}, {B10: 1}, {C0: 1})
        add_term({A0: 1}, {B01: 1}, {C1: 1})
        add_term({A1: 1}, {B11: 1}, {C1: 1})
    
    # Block indices:
    # A is 3x4, partitioned as n->1+2, m->2+2
    # A[0,0]: U rows 0,1 (row 0, cols 0,1)
    # A[0,1]: U rows 2,3 (row 0, cols 2,3)
    # A[1,0]: U rows 4,5,8,9 (rows 1,2, cols 0,1)
    # A[1,1]: U rows 6,7,10,11 (rows 1,2, cols 2,3)
    
    # B is 4x6, partitioned as m->2+2, p->2+2+2
    # B[0,0]: V rows 0,1,6,7 (rows 0,1, cols 0,1)
    # B[0,1]: V rows 2,3,8,9 (rows 0,1, cols 2,3)
    # B[0,2]: V rows 4,5,10,11 (rows 0,1, cols 4,5)
    # B[1,0]: V rows 12,13,18,19 (rows 2,3, cols 0,1)
    # B[1,1]: V rows 14,15,20,21 (rows 2,3, cols 2,3)
    # B[1,2]: V rows 16,17,22,23 (rows 2,3, cols 4,5)
    
    # C is 3x6, partitioned as n->1+2, p->2+2+2
    # C[0,0]: W rows 0,1 (row 0, cols 0,1)
    # C[0,1]: W rows 6,7 (row 0, cols 2,3)
    # C[0,2]: W rows 12,13 (row 0, cols 4,5)
    # C[1,0]: W rows 1,2,4,5 (rows 1,2, cols 0,1)
    # C[1,1]: W rows 7,8,10,11 (rows 1,2, cols 2,3)
    # C[1,2]: W rows 13,14,16,17 (rows 1,2, cols 4,5)
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    add_1x2_2x2(0, 1, 0, 1, 6, 7, 0, 1)
    add_1x2_2x2(2, 3, 12, 13, 18, 19, 0, 1)
    
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    add_1x2_2x2(0, 1, 2, 3, 8, 9, 6, 7)
    add_1x2_2x2(2, 3, 14, 15, 20, 21, 6, 7)
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    add_1x2_2x2(0, 1, 4, 5, 10, 11, 12, 13)
    add_1x2_2x2(2, 3, 16, 17, 22, 23, 12, 13)
    
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    add_strassen(4, 5, 8, 9, 0, 1, 6, 7, 1, 2, 4, 5)
    add_strassen(6, 7, 10, 11, 12, 13, 18, 19, 1, 2, 4, 5)
    
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    add_strassen(4, 5, 8, 9, 2, 3, 8, 9, 7, 8, 10, 11)
    add_strassen(6, 7, 10, 11, 14, 15, 20, 21, 7, 8, 10, 11)
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    add_strassen(4, 5, 8, 9, 4, 5, 10, 11, 13, 14, 16, 17)
    add_strassen(6, 7, 10, 11, 16, 17, 22, 23, 13, 14, 16, 17)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
