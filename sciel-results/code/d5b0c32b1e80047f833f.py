import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 4 Strassen (2x2x2) blocks for top 2 rows
    # 4 Dot (1x2x2) blocks for bottom 1 row
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4 (2x2 flattened)
    # u index: i*2+j
    # v index: j*2+k
    # w index: k*2+i
    
    strassen_terms = []
    # P1 = (A11+A22)(B11+B22)
    # u: (1, 0, 0, 1) -> A11, A22
    # v: (1, 0, 0, 1) -> B11, B22
    # w: (1, 1, 0, 0)? No, w contributes to C.
    # The term is u(A) * v(B) * w(C).
    # We need to add this to C.
    # C11 += P1, C12 += 0, C21 += 0, C22 += P1? No.
    # Strassen formula:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P6
    # C21 = P2 + P5
    # C22 = P1 - P3 + P2 + P6
    
    # We define each P as a rank-1 tensor in local space.
    # P1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
    # The product u(A)v(B) is a scalar. It contributes to C via w.
    # So for P1, the contribution to C is P1 * w_P1.
    # But P1 is not a matrix, it's a scalar.
    # Wait, the tensor decomposition sums rank-1 tensors.
    # Each term is u_r (vector in A space) x v_r (vector in B space) x w_r (vector in C space).
    # The value is sum_r <A, u_r> <B, v_r> w_r.
    # This equals C.
    
    # For Strassen:
    # P1 = (A11+A22)(B11+B22)
    # u_P1 = [1, 0, 0, 1] (A11, A22)
    # v_P1 = [1, 0, 0, 1] (B11, B22)
    # This scalar P1 adds to C11 and C22.
    # So w_P1 should have 1 at C11 and 1 at C22?
    # No, the decomposition is linear.
    # We need to express C as sum of terms.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P6
    # C21 = P2 + P5
    # C22 = P1 - P3 + P2 + P6
    
    # This means:
    # Term 1 (P1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (w indices: C11, C12, C21, C22)
    # Wait, w is a vector in C space.
    # If w=[1,0,0,1], it adds P1 to C11 and P1 to C22.
    # Term 2 (P2): u=[0,1,1,0] (A21, A22), v=[1,0,0,0] (B11), w=[0,0,1,0] (C21)
    # Term 3 (P3): u=[1,0,-1,0] (A11, -A21), v=[0,1,1,0] (B21, B22), w=[0,1,0,0] (C12)
    # Term 4 (P4): u=[1,0,0,0] (A11), v=[0,1,-1,0] (B12, -B22), w=[1,0,0,0] (C11)
    # Term 5 (P5): u=[0,0,0,1] (A22), v=[1,0,-1,0] (B11, -B12), w=[0,0,1,0] (C21) -- Wait, P5 adds to C21 and C22?
    # C21 = P2 + P5. C22 = ... - P5? No.
    # C22 = P1 - P3 + P2 + P6.
    # Let's re-verify Strassen coefficients.
    
    # Standard:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = (A11-A21)(B21+B22)
    # P4 = A11(B12-B22)
    # P5 = A22(B11-B12)
    # P6 = (A11+A12)B22
    # P7 = (A12-A22)(B11+B21)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P6
    # C21 = P2 + P5
    # C22 = P1 - P3 + P2 + P6
    
    # So:
    # Term 1 (P1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Term 2 (P2): u=[0,1,1,0], v=[1,0,0,0], w=[0,0,1,0]
    # Term 3 (P3): u=[1,0,-1,0], v=[0,1,1,0], w=[0,1,0,0]
    # Term 4 (P4): u=[1,0,0,0], v=[0,1,-1,0], w=[1,0,0,0]
    # Term 5 (P5): u=[0,0,0,1], v=[1,0,-1,0], w=[0,0,1,0] -- Wait, P5 is in C21 and C22?
    # C21 has +P5. C22 has ...?
    # C22 = P1 - P3 + P2 + P6. No P5.
    # So w for P5 is [0,0,1,0].
    # Term 6 (P6): u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,1]
    # Term 7 (P7): u=[0,1,0,-1], v=[1,1,0,0], w=[1,0,0,0]
    
    # Let's check C11: P1(1) + P4(1) - P5(0) + P7(1) = 3? No.
    # C11 coeff for P1 is 1. P4 is 1. P5 is -1. P7 is 1.
    # My w vectors above assume positive coefficients.
    # We can just put the coefficient in u, v or w.
    # Easier: just add the terms with signs in w.
    
    strassen_uvw = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # P1
        ([0,1,1,0], [1,0,0,0], [0,0,1,0]), # P2
        ([1,0,-1,0], [0,1,1,0], [0,1,0,0]), # P3
        ([1,0,0,0], [0,1,-1,0], [1,0,0,0]), # P4
        ([0,0,0,1], [1,0,-1,0], [0,0,1,0]), # P5
        ([1,1,0,0], [0,0,0,1], [0,1,0,1]), # P6
        ([0,1,0,-1], [1,1,0,0], [1,0,0,0])  # P7
    ]
    
    # Dot terms for 1x2x2
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # u len 2, v len 4, w len 2
    dot_uvw = [
        ([1,0], [1,0,0,0], [1,0]), # A0 B00 -> C0
        ([0,1], [0,1,0,0], [1,0]), # A1 B10 -> C0
        ([1,0], [0,0,1,0], [0,1]), # A0 B01 -> C1
        ([0,1], [0,0,0,1], [0,1])  # A1 B11 -> C1
    ]
    
    # Helper to add a sub-problem
    def add_strassen(row_offset, col_offset_a, col_offset_b, row_offset_c, col_offset_c):
        nonlocal col
        for u, v, w in strassen_uvw:
            # u is local A (2x2), v is local B (2x2), w is local C (2x2)
            # Map to global
            for li in range(2):
                for lj in range(2):
                    gi = row_offset + li
                    gj = col_offset_a + lj
                    U[gi * m + gj][col] += u[li * 2 + lj]
            
            for lj in range(2):
                for lk in range(2):
                    gj = col_offset_b + lj
                    gk = lk # B cols are global k? No, B is 4x4.
                    # B local (2x2) at (col_offset_b, row_offset_b?)
                    # B index j*p+k.
                    # Local j' in 0..1, k' in 0..1.
                    # Global j = col_offset_b + j'
                    # Global k = row_offset_c? No.
                    # B is multiplied by A.
                    # A cols are j. B rows are j.
                    # So B local row j' maps to global j = col_offset_b + j'.
                    # B local col k' maps to global k = ?
                    # C col is k.
                    # For C11 (cols 0-1), k' maps to global k = col_offset_c + k'.
                    # For C12 (cols 2-3), k' maps to global k = col_offset_c + k'.
                    # So we need col_offset_b for B rows and col_offset_c for B cols?
                    # No, B is fixed.
                    # B_{j, k}.
                    # In sub-problem 1: A11 (j 0-1) * B11 (j 0-1, k 0-1).
                    # Global j = j'. Global k = k'.
                    # In sub-problem 2: A12 (j 2-3) * B21 (j 2-3, k 0-1).
                    # Global j = 2+j'. Global k = k'.
                    
                    # So for V:
                    # Global row index = gj * p + gk.
                    # gj = col_offset_b + lj
                    # gk = ?
                    # The k index in B corresponds to the k index in C.
                    # So gk = col_offset_c + lk?
                    # Yes, because C_{i, k} depends on B_{j, k}.
                    
                    V[(col_offset_b + lj) * p + (col_offset_c + lk)][col] += v[lj * 2 + lk]
            
            for lk in range(2):
                for li in range(2):
                    gk = col_offset_c + lk
                    gi = row_offset + li
                    W[gk * n + gi][col] += w[lk * 2 + li]
            
            col += 1

    def add_dot(row_offset, col_offset_a, col_offset_b, row_offset_c, col_offset_c):
        nonlocal col
        for u, v, w in dot_uvw:
            # u len 2 (A row), v len 4 (B 2x2), w len 2 (C row)
            for li in range(1): # A is 1x2
                for lj in range(2):
                    gi = row_offset + li
                    gj = col_offset_a + lj
                    U[gi * m + gj][col] += u[li * 2 + lj]
            
            for lj in range(2):
                for lk in range(2):
                    gj = col_offset_b + lj
                    gk = col_offset_c + lk
                    V[gj * p + gk][col] += v[lj * 2 + lk]
            
            for lk in range(2):
                for li in range(1):
                    gk = col_offset_c + lk
                    gi = row_offset + li
                    W[gk * n + gi][col] += w[lk * 2 + li]
            
            col += 1

    # Sub-problem 1: A11 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C11 (rows 0-1, cols 0-1)
    add_strassen(0, 0, 0, 0, 0)
    
    # Sub-problem 2: A12 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C11 (rows 0-1, cols 0-1)
    add_strassen(0, 2, 2, 0, 0)
    
    # Sub-problem 3: A11 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-3) -> C12 (rows 0-1, cols 2-3)
    add_strassen(0, 0, 0, 0, 2)
    
    # Sub-problem 4: A12 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-3) -> C12 (rows 0-1, cols 2-3)
    add_strassen(0, 2, 2, 0, 2)
    
    # Sub-problem 5: A21 (row 2, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C21 (row 2, cols 0-1)
    add_dot(2, 0, 0, 2, 0)
    
    # Sub-problem 6: A22 (row 2, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C21 (row 2, cols 0-1)
    add_dot(2, 2, 2, 2, 0)
    
    # Sub-problem 7: A21 (row 2, cols 0-1) * B12 (rows 0-1, cols 2-3) -> C22 (row 2, cols 2-3)
    add_dot(2, 0, 0, 2, 2)
    
    # Sub-problem 8: A22 (row 2, cols 2-3) * B22 (rows 2-3, cols 2-3) -> C22 (row 2, cols 2-3)
    add_dot(2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
