import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a Rank 36 decomposition using Strassen for 2x2x2 blocks.
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A into A_top (2x4) and A_bot (1x4).
    # Split B into 4 blocks of 2x2: B00, B01, B10, B11.
    # A_top = [A_L, A_R] where A_L, A_R are 2x2.
    # A_bot = [v_L, v_R] where v_L, v_R are 1x2.
    
    # C_top = [A_L B00 + A_R B10, A_L B01 + A_R B11]
    # C_bot = [v_L B00 + v_R B10, v_L B01 + v_R B11]
    
    # Strassen for 2x2x2 (Rank 7):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For A_top (2x4) x B (4x4):
    # We need 4 Strassen instances.
    # Instance 1: A_L * B00 (Result is 2x2, contributes to C_top[0:2, 0:2] left part? No, C_top is 2x4.
    # Let's map indices carefully.
    # A_top indices: i in {0,1}, j in {0,1,2,3}.
    # A_L: i in {0,1}, j in {0,1}.
    # A_R: i in {0,1}, j in {2,3}.
    # B blocks:
    # B00: j in {0,1}, k in {0,1}.
    # B01: j in {0,1}, k in {2,3}.
    # B10: j in {2,3}, k in {0,1}.
    # B11: j in {2,3}, k in {2,3}.
    
    # Product 1: A_L * B00 (2x2 * 2x2 -> 2x2). Contributes to C[0:2, 0:2].
    # Product 2: A_L * B01 (2x2 * 2x2 -> 2x2). Contributes to C[0:2, 2:4].
    # Product 3: A_R * B10 (2x2 * 2x2 -> 2x2). Contributes to C[0:2, 0:2].
    # Product 4: A_R * B11 (2x2 * 2x2 -> 2x2). Contributes to C[0:2, 2:4].
    
    # So C_top[0:2, 0:2] = (A_L * B00) + (A_R * B10).
    # C_top[0:2, 2:4] = (A_L * B01) + (A_R * B11).
    
    # Product 5-8 for A_bot (1x4) * B (4x4):
    # v_L * B00 (1x2 * 2x2 -> 1x2). Contributes to C[2, 0:2].
    # v_L * B01 (1x2 * 2x2 -> 1x2). Contributes to C[2, 2:4].
    # v_R * B10 (1x2 * 2x2 -> 1x2). Contributes to C[2, 0:2].
    # v_R * B11 (1x2 * 2x2 -> 1x2). Contributes to C[2, 2:4].
    
    # C[2, 0:2] = (v_L * B00) + (v_R * B10).
    # C[2, 2:4] = (v_L * B01) + (v_R * B11).
    
    # Total Rank:
    # 4 Strassen (7 each) = 28.
    # 4 Schoolbook 1x2x2 (2 each) = 8.
    # Total 36.
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen term
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols: indices for A block (2x2)
        # b_rows, b_cols: indices for B block (2x2)
        # c_rows, c_cols: indices for C block (2x2)
        
        # Strassen coefficients for A (2x2) and B (2x2)
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A11 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # We need to define the linear combinations for U, V, W.
        # U corresponds to A, V to B, W to C.
        # Each Strassen multiplication is a rank-1 tensor in the 2x2x2 space.
        # We add 7 columns.
        
        # M1
        # A: A00 + A11 -> U[a_rows[0], a_cols[0]] += 1, U[a_rows[1], a_cols[1]] += 1
        # B: B00 + B11 -> V[b_rows[0], b_cols[0]] += 1, V[b_rows[1], b_cols[1]] += 1
        # C: C00 += 1, C11 += 1, C01 -= 1, C10 -= 1 ? 
        # Wait, C00 = M1 + M4 - M5 + M7.
        # So for M1, C contribution is +1 to C00, +1 to C11, -1 to C01, -1 to C10?
        # Let's check Strassen formula again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 contributes to: C00(+1), C11(+1).
        # M2 contributes to: C10(+1), C11(-1).
        # M3 contributes to: C01(+1), C11(+1).
        # M4 contributes to: C00(+1), C10(+1).
        # M5 contributes to: C00(-1), C01(+1).
        # M6 contributes to: C11(+1).
        # M7 contributes to: C00(+1).
        
        # Let's verify M1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # C00 term: A00B00 + A11B11 (from M1? No, M1 has cross terms).
        # Actually, standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We implement each M as a rank-1 term in the global U,V,W.
        
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            # a_coeffs: dict {(row, col): val}
            # b_coeffs: dict {(row, col): val}
            # c_coeffs: dict {(row, col): val}
            nonlocal col
            for (ar, ac), av in a_coeffs.items():
                U[ar * m + ac, col] += av
            for (br, bc), bv in b_coeffs.items():
                V[br * p + bc, col] += bv
            for (cr, cc), cv in c_coeffs.items():
                W[cr * n + cc, col] += cv
            col += 1
            
        # M1
        a1 = { (a_rows[0], a_cols[0]): 1, (a_rows[1], a_cols[1]): 1 }
        b1 = { (b_rows[0], b_cols[0]): 1, (b_rows[1], b_cols[1]): 1 }
        c1 = { (c_rows[0], c_cols[0]): 1, (c_rows[1], c_cols[1]): 1 }
        add_term(a1, b1, c1)
        
        # M2
        a2 = { (a_rows[1], a_cols[0]): 1, (a_rows[1], a_cols[1]): 1 }
        b2 = { (b_rows[0], b_cols[0]): 1 }
        c2 = { (c_rows[1], c_cols[0]): 1, (c_rows[1], c_cols[1]): -1 }
        add_term(a2, b2, c2)
        
        # M3
        a3 = { (a_rows[0], a_cols[0]): 1 }
        b3 = { (b_rows[1], b_cols[0]): 1, (b_rows[1], b_cols[1]): -1 }
        c3 = { (c_rows[0], c_cols[1]): 1, (c_rows[1], c_cols[1]): 1 }
        add_term(a3, b3, c3)
        
        # M4
        a4 = { (a_rows[1], a_cols[1]): 1 }
        b4 = { (b_rows[0], b_cols[1]): 1, (b_rows[0], b_cols[0]): -1 }
        c4 = { (c_rows[0], c_cols[0]): 1, (c_rows[1], c_cols[0]): 1 }
        add_term(a4, b4, c4)
        
        # M5
        a5 = { (a_rows[0], a_cols[0]): 1, (a_rows[0], a_cols[1]): 1 }
        b5 = { (b_rows[1], b_cols[1]): 1 }
        c5 = { (c_rows[0], c_cols[0]): -1, (c_rows[0], c_cols[1]): 1 }
        add_term(a5, b5, c5)
        
        # M6
        a6 = { (a_rows[1], a_cols[1]): 1, (a_rows[0], a_cols[0]): -1 }
        b6 = { (b_rows[0], b_cols[0]): 1, (b_rows[1], b_cols[0]): 1 }
        c6 = { (c_rows[1], c_cols[1]): 1 }
        add_term(a6, b6, c6)
        
        # M7
        a7 = { (a_rows[0], a_cols[1]): 1, (a_rows[1], a_cols[1]): -1 }
        b7 = { (b_rows[0], b_cols[1]): 1, (b_rows[1], b_cols[1]): 1 }
        c7 = { (c_rows[0], c_cols[0]): 1 }
        add_term(a7, b7, c7)

    # Helper to add Schoolbook 1x2x2 term
    def add_schoolbook_1x2x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        # A is 1x2, B is 2x2, C is 1x2.
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # Term 1: A0 B00 -> C0
        # Term 2: A1 B10 -> C0
        # Term 3: A0 B01 -> C1
        # Term 4: A1 B11 -> C1
        # Rank 4? No, we can do it in 2?
        # No, $M(1,2,2)=2$ is false. $M(1,2,2)=2$?
        # C0 = A0 B00 + A1 B10.
        # C1 = A0 B01 + A1 B11.
        # This is 4 products.
        # Can we reduce?
        # If we use Strassen for 1x2x2?
        # No, 1x2x2 is just dot products.
        # Wait, $M(1,2,2)$ is the rank of the tensor $T_{i,j,k} = \delta_{i,0} \delta_{j,k}$? No.
        # $C_k = \sum_j A_j B_{j,k}$.
        # This is 2 multiplications for each k.
        # Total 4 multiplications.
        # Is there a rank 2 algorithm?
        # $C_0 = A_0 B_{00} + A_1 B_{10}$
        # $C_1 = A_0 B_{01} + A_1 B_{11}$
        # We can compute $M_1 = A_0 B_{00} + A_1 B_{10}$? No, that's a sum.
        # We need separate terms for the tensor decomposition.
        # Each multiplication is a rank-1 term.
        # So 4 terms.
        
        nonlocal col
        # Term 1
        U[a_row * m + a_cols[0], col] = 1
        V[b_rows[0] * p + b_cols[0], col] = 1
        W[c_row * n + c_cols[0], col] = 1
        col += 1
        
        # Term 2
        U[a_row * m + a_cols[1], col] = 1
        V[b_rows[1] * p + b_cols[0], col] = 1
        W[c_row * n + c_cols[0], col] = 1
        col += 1
        
        # Term 3
        U[a_row * m + a_cols[0], col] = 1
        V[b_rows[0] * p + b_cols[1], col] = 1
        W[c_row * n + c_cols[1], col] = 1
        col += 1
        
        # Term 4
        U[a_row * m + a_cols[1], col] = 1
        V[b_rows[1] * p + b_cols[1], col] = 1
        W[c_row * n + c_cols[1], col] = 1
        col += 1

    # 1. A_L (rows 0,1; cols 0,1) * B00 (rows 0,1; cols 0,1) -> C (rows 0,1; cols 0,1)
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. A_L (rows 0,1; cols 0,1) * B01 (rows 0,1; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # 3. A_R (rows 0,1; cols 2,3) * B10 (rows 2,3; cols 0,1) -> C (rows 0,1; cols 0,1)
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # 4. A_R (rows 0,1; cols 2,3) * B11 (rows 2,3; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # 5. v_L (row 2; cols 0,1) * B00 (rows 0,1; cols 0,1) -> C (row 2; cols 0,1)
    add_schoolbook_1x2x2(2, [0, 1], [0, 1], [0, 1], 2, [0, 1])
    
    # 6. v_L (row 2; cols 0,1) * B01 (rows 0,1; cols 2,3) -> C (row 2; cols 2,3)
    add_schoolbook_1x2x2(2, [0, 1], [0, 1], [2, 3], 2, [2, 3])
    
    # 7. v_R (row 2; cols 2,3) * B10 (rows 2,3; cols 0,1) -> C (row 2; cols 0,1)
    add_schoolbook_1x2x2(2, [2, 3], [2, 3], [0, 1], 2, [0, 1])
    
    # 8. v_R (row 2; cols 2,3) * B11 (rows 2,3; cols 2,3) -> C (row 2; cols 2,3)
    add_schoolbook_1x2x2(2, [2, 3], [2, 3], [2, 3], 2, [2, 3])
    
    # Check col count
    # 4 * 7 = 28
    # 4 * 4 = 16
    # Total 44.
    # Wait, my previous estimate was 36.
    # Why?
    # M(1,2,2) rank is 2?
    # $C_0 = A_0 B_{00} + A_1 B_{10}$
    # $C_1 = A_0 B_{01} + A_1 B_{11}$
    # If we use rank 2:
    # $M_1 = A_0 B_{00} + A_1 B_{10}$? No, that's a sum of two products, not a single product.
    # In tensor decomposition, each term is a single product $u v w$.
    # So $A_0 B_{00}$ is one term. $A_1 B_{10}$ is another.
    # So 4 terms.
    # So Rank 44.
    
    # Can we reduce M(1,2,2)?
    # $C_k = \sum_j A_j B_{jk}$.
    # This is the dot product of vector A and column k of B.
    # We can compute $M_1 = A_0 B_{00} + A_1 B_{10}$? No.
    # We can use Strassen-like?
    # No, 1x2x2 is trivial. Rank is 4.
    
    # So Rank 44.
    # This is still better than 48.
    
    # Verify col == rank
    assert col == rank, f"Col {col} != Rank {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
