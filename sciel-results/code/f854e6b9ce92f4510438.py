import sys
import time

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Strategy: Split B (4x7) into column blocks:
      - B_1 (4x2): cols 0,1
      - B_2 (4x2): cols 2,3  
      - B_3 (4x3): cols 4,5,6
    
    For each 4x2 block, split A into A_L (2x2) and A_R (2x2),
    and the 4x2 block into two 2x2 sub-blocks.
    Each 2x2 * 2x2 -> 2x2 uses Strassen (rank 7).
    6 such blocks: 6 * 7 = 42
    
    For the 4x3 block, split into 4x2 + 4x1:
      - 4x2 part: same as above, 2 Strassen blocks = 14
      - 4x1 part: split into 2x1 + 2x1, each 2x2 * 2x1 -> 2x1 is schoolbook (rank 4)
        2 such blocks: 2 * 4 = 8
    
    Total rank: 42 + 8 = 50
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = 2, 4, 7
    rank = 50
    
    # Initialize factor matrices
    # U shape: (n*m, rank) = (8, 50)
    # V shape: (m*p, rank) = (28, 50)
    # W shape: (p*n, rank) = (14, 50)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def check_deadline():
        nonlocal col
        if time.time() > deadline:
            # If we exceed deadline, we return what we have. 
            # Given the deterministic nature and low rank, this should not happen.
            return True
        return False

    def add_strassen_block(A_indices, B_indices, C_indices):
        """Add a Strassen 2x2x2 decomposition (rank 7)."""
        nonlocal col
        # Strassen's 7 multiplications for 2x2 * 2x2 -> 2x2
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            if check_deadline():
                return
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1 = (a00+a11)(b00+b11)
        process_mk({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        # M2 = (a10+a11)b00
        process_mk({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        # M3 = a00(b01-b11)
        process_mk({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        # M4 = a11(b10-b00)
        process_mk({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        # M5 = (a00+a01)b11
        process_mk({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        # M6 = (a10-a00)(b00+b01)
        process_mk({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        # M7 = (a01-a11)(b10+b11)
        process_mk({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        """Add a schoolbook 2x2 * 2x1 -> 2x1 decomposition (rank 4)."""
        nonlocal col
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            if check_deadline():
                return
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    # A split: A_L (cols 0,1), A_R (cols 2,3)
    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    # Block 1: B cols 0,1 (4x2)
    B_1_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    B_1_R = [(2, 0), (2, 1), (3, 0), (3, 1)]
    C_1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # Block 2: B cols 2,3 (4x2)
    B_2_L = [(0, 2), (0, 3), (1, 2), (1, 3)]
    B_2_R = [(2, 2), (2, 3), (3, 2), (3, 3)]
    C_2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # Block 3: B cols 4,5,6 (4x3)
    # First 2 cols (4x2): Strassen
    B_3_L_L = [(0, 4), (0, 5), (1, 4), (1, 5)]
    B_3_L_R = [(2, 4), (2, 5), (3, 4), (3, 5)]
    C_3_L = [(0, 4), (0, 5), (1, 4), (1, 5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    # Last col (4x1): Schoolbook
    B_4_L = [(0, 6), (1, 6)]
    B_4_R = [(2, 6), (3, 6)]
    C_4 = [(0, 6), (1, 6)]
    
    add_schoolbook_221(A_L, B_4_L, C_4)
    add_schoolbook_221(A_R, B_4_R, C_4)
    
    # Verify: col should be exactly 50
    if col != rank:
        # If we didn't fill all ranks, we might have an error, but we return what we have.
        # Truncate if necessary, though shouldn't happen.
        if col < rank:
             # Zero out unused columns
             for r in range(col, rank):
                 for i in range(n*m): U[i][r] = 0
                 for i in range(m*p): V[i][r] = 0
                 for i in range(p*n): W[i][r] = 0
        else:
             # This case implies we exceeded rank, which is bad for score if rank is fixed to 50?
             # The interface says "factor matrices of shapes ... rank".
             # If col > rank, we have a bug. We assume col == rank.
             pass
    
    return {"U": U, "V": V, "W": W}
