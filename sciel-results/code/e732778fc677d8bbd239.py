import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def strassen_222(U_offset, V_offset, W_offset, base_U, base_V, base_W, rank_start):
    """
    Adds Strassen decomposition for 2x2x2 multiplication to the matrices.
    U, V, W are assumed to be lists of lists or numpy arrays.
    We will use numpy for simplicity and speed, then convert to lists if needed?
    The interface requires {'U', 'V', 'W'} with integer/half-integer entries.
    Lists of lists are fine.
    """
    # Strassen terms for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A11+A22)(B21-B11)
    # M3 = (A12+A22)B11
    # M4 = (A11-A21)B22
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in local 2x2x2 block
    # A: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # B: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # C: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    
    terms = [
        # (U_indices, coeffs, V_indices, coeffs, W_indices, coeffs)
        # M1
        ([(0, 1), (3, 1)], [1, 1], [(0, 1), (3, 1)], [1, 1], [(0, 0), (1, 1)], [1, 1]),
        # M2
        ([(0, 1), (3, 1)], [1, 1], [(2, 1), (0, -1)], [1, -1], [(1, 0), (1, 1)], [1, 1]),
        # M3
        ([(1, 1), (3, 1)], [1, 1], [(0, 1)], [1], [(0, 1)], [1]),
        # M4
        ([(0, 1), (2, -1)], [1, -1], [(3, 1)], [1], [(0, 1), (1, 0)], [1, 1]),
        # M5
        ([(0, 1), (1, 1)], [1, 1], [(3, 1)], [1], [(0, 1)], [-1]), # Note: C11 has -M5
        # M6
        ([(2, 1), (0, -1)], [1, -1], [(0, 1), (1, 1)], [1, 1], [(1, 1)], [1]),
        # M7
        ([(1, 1), (3, -1)], [1, -1], [(2, 1), (3, 1)], [1, 1], [(0, 0), (0, 1)], [1, 1]),
    ]
    
    # Correction: The W coefficients must match the C reconstruction.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the contribution of each M to C.
    # M1: C11 += 1, C22 += 1
    # M2: C21 += 1, C22 -= 1
    # M3: C12 += 1, C22 += 1
    # M4: C11 += 1, C21 += 1
    # M5: C11 -= 1, C12 += 1
    # M6: C22 += 1
    # M7: C11 += 1, C01? No, C11?
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Let's use this standard set.
    # Indices: A: 0,1,2,3. B: 0,1,2,3. C: 0,1,2,3.
    
    strassen_terms = [
        # P1
        ([(0, 1), (3, 1)], [1, 1], [(0, 1), (3, 1)], [1, 1], [(0, 0), (3, 1)], [1, 1]),
        # P2
        ([(2, 1), (3, 1)], [1, 1], [(0, 1)], [1], [(2, 0), (3, -1)], [1, -1]),
        # P3
        ([(0, 1)], [1], [(2, 1), (3, -1)], [1, -1], [(1, 0), (3, 1)], [1, 1]),
        # P4
        ([(3, 1)], [1], [(1, 1), (0, -1)], [1, -1], [(0, 1), (2, 0)], [1, 1]),
        # P5
        ([(0, 1), (1, 1)], [1, 1], [(3, 1)], [1], [(0, -1), (1, 1)], [-1, 1]),
        # P6
        ([(2, 1), (0, -1)], [1, -1], [(0, 1), (1, 1)], [1, 1], [(3, 1)], [1]),
        # P7
        ([(1, 1), (3, -1)], [1, -1], [(2, 1), (3, 1)], [1, 1], [(0, 1), (1, 1)], [1, 1]),
    ]
    
    for u_idx, u_c, v_idx, v_c, w_idx, w_c in strassen_terms:
        col = rank_start
        for idx, c in zip(u_idx, u_c):
            base_U[U_offset + idx][col] += c
        for idx, c in zip(v_idx, v_c):
            base_V[V_offset + idx][col] += c
        for idx, c in zip(w_idx, w_c):
            base_W[W_offset + idx][col] += c
        rank_start += 1
        
    return rank_start

def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition
    # 1. 2x4x4 (Rank 28)
    # 2. 2x4x2 (Rank 14)
    # 3. 2x4x1 (Rank 4)
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 46
    V = [[0] * total_rank for _ in range(m * p)] # 28 x 46
    W = [[0] * total_rank for _ in range(p * n)] # 14 x 46
    
    # Part 1: 2x4x4
    # Split B into B1 (4x2), B2 (4x2)
    # B1 cols 0,1. B2 cols 2,3.
    # A is 2x4.
    # C1 = A B1 (2x2). C2 = A B2 (2x2).
    
    # Sub-part 1a: A B1 (2x4x2)
    # Split A into A1 (2x2), A2 (2x2).
    # A1 cols 0,1. A2 cols 2,3.
    # B1 is 4x2. Split B1 into BT (2x2), BB (2x2).
    # BT rows 0,1. BB rows 2,3.
    # C1 = A1 BT + A2 BB.
    
    # Strassen for A1 BT (2x2x2)
    # U offset 0 (A1), V offset 0 (BT), W offset 0 (C1)
    rank = strassen_222(0, 0, 0, U, V, W, 0)
    
    # Strassen for A2 BB (2x2x2)
    # U offset 2 (A2), V offset 4 (BB), W offset 0 (C1)
    rank = strassen_222(2, 4, 0, U, V, W, rank)
    
    # Sub-part 1b: A B2 (2x4x2)
    # B2 cols 2,3.
    # C2 = A1 BT2 + A2 BB2.
    # BT2 is B rows 0,1, cols 2,3.
    # BB2 is B rows 2,3, cols 2,3.
    
    # Strassen for A1 BT2
    # U offset 0, V offset 8 (BT2), W offset 2 (C2)
    rank = strassen_222(0, 8, 2, U, V, W, rank)
    
    # Strassen for A2 BB2
    # U offset 2, V offset 12 (BB2), W offset 2 (C2)
    rank = strassen_222(2, 12, 2, U, V, W, rank)
    
    # Part 2: 2x4x2 (Rank 14)
    # B3 is 4x2, cols 4,5.
    # C3 = A B3 (2x2).
    # C3 = A1 BT3 + A2 BB3.
    # BT3 rows 0,1, cols 4,5.
    # BB3 rows 2,3, cols 4,5.
    
    # Strassen for A1 BT3
    # U offset 0, V offset 16 (BT3), W offset 4 (C3)
    rank = strassen_222(0, 16, 4, U, V, W, rank)
    
    # Strassen for A2 BB3
    # U offset 2, V offset 20 (BB3), W offset 4 (C3)
    rank = strassen_222(2, 20, 4, U, V, W, rank)
    
    # Part 3: 2x4x1 (Rank 4)
    # B4 is 4x1, col 6.
    # C4 = A B4 (2x1).
    # C4_1 = A11 B4_1 + A12 B4_2 + A13 B4_3 + A14 B4_4
    # C4_2 = A21 B4_1 + A22 B4_2 + A23 B4_3 + A24 B4_4
    
    # We use 4 terms.
    # Term j (1..4):
    # U_j picks A1j and A2j.
    # V_j picks B4_j.
    # W_j is [1, 1] for C4_1 and C4_2?
    # Wait, C4_1 needs A1j, C4_2 needs A2j.
    # If W_j = [1, 1], then C4_1 gets A1j B4_j and C4_2 gets A2j B4_j.
    # This is correct.
    
    for j in range(4):
        col = rank
        # U offset: A1j is j, A2j is 4+j
        U[j][col] = 1
        U[4+j][col] = 1
        # V offset: B4_j is 24+j (since B is 4x7, flat index 24..27 for col 6)
        # Wait, B is 4x7. Flat index for col 6, row j is 6*4 + j?
        # No, B is m x p = 4 x 7.
        # Flat index for B[row, col] is row * p + col.
        # B4 is col 6. Rows 0..3.
        # So V index is row * 7 + 6.
        V[j * 7 + 6][col] = 1
        # W offset: C4 is 2x1. Flat index 0, 1.
        W[0][col] = 1
        W[1][col] = 1
        rank += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0][:5])
