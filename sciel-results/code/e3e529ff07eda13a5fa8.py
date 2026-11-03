"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank 36 decomposition using Strassen for 2x2x2 blocks.
    # Matrix multiply: A (2x4) * B (4x5) = C (2x5)
    #
    # We split B into three parts: B1 (4x2), B2 (4x2), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    #
    # For A*B1 and A*B2 (each is 2x4 * 4x2 = 2x2), we use Strassen (rank 7).
    # But A is 2x4, so we split A into AL (2x2) and AR (2x2).
    # For B1, we split into BL1 (2x2) and BR1 (2x2) based on rows 0-1 and 2-3.
    # Similarly for B2.
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For A*B3 (2x4 * 4x1 = 2x1), we use schoolbook (rank 4).
    #
    # Total rank: 7 + 7 + 4 = 18? Wait, that doesn't seem right.
    # Actually, for 2x4 * 4x2, we can't directly use 2x2 Strassen.
    # We need to think about this more carefully.
    #
    # Let me reconsider. A is 2x4, B is 4x5.
    # Split A into AL (2x2) columns 0,1 and AR (2x2) columns 2,3.
    # Split B into 5 blocks of 4x1: B0, B1, B2, B3, B4.
    # C = AL*B0 + AR*B0 + AL*B1 + AR*B1 + ... + AL*B4 + AR*B4
    #
    # For each Bk (4x1), AL*Bk is 2x2 * 2x1 and AR*Bk is 2x2 * 2x1.
    # 2x2 * 2x1 can be done with rank 4 (schoolbook) or we can group.
    #
    # Actually, let's use a different approach. The known result for 2x4x5:
    # The rank is at most 2*4*5 = 40 (schoolbook).
    # With Strassen-like techniques, we can do better.
    #
    # Let me try a direct construction. We'll use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as a sum of rank-1 terms. Each rank-1 term is:
    # u * v^T * w^T where u is 2x1, v is 4x1, w is 5x1.
    #
    # Actually, the tensor is:
    # T[i,j,k] = delta[i,k] (for the multiplication to work out)
    #
    # Let me just implement the Strassen-based decomposition for 2x4x4 first,
    # then add the 5th column.
    #
    # For 2x4x4: Split A into AL, AR (2x2 each). Split B into B0, B1 (4x2 each).
    # C = AL*B0 + AR*B0 + AL*B1 + AR*B1
    #
    # AL*B0: A is 2x2, B is 2x2 -> use Strassen (rank 7)
    # AR*B0: A is 2x2, B is 2x2 -> use Strassen (rank 7)
    # AL*B1: A is 2x2, B is 2x2 -> use Strassen (rank 7)
    # AR*B1: A is 2x2, B is 2x2 -> use Strassen (rank 7)
    #
    # But we can share some Strassen products if we group them.
    #
    # Actually, let me just do it straightforwardly:
    # For each of the 4 sub-problems (2x2 * 2x2), use Strassen with rank 7.
    # Total rank for 2x4x4: 4*7 = 28.
    # Then add 2x4x1 (the 5th column of B): schoolbook rank 4.
    # Total: 28 + 4 = 32.
    #
    # But wait, we need to be careful. Let me define the indices.
    #
    # A[i,j] where i in {0,1}, j in {0,1,2,3}
    # B[j,k] where j in {0,1,2,3}, k in {0,1,2,3,4}
    # C[i,k] where i in {0,1}, k in {0,1,2,3,4}
    #
    # Split:
    # AL = A[:, 0:2] (2x2)
    # AR = A[:, 2:4] (2x2)
    #
    # For k in {0,1,2,3}:
    #   Bk = B[:, k] (4x1)
    #   BLk = Bk[0:2] (2x1)
    #   BRk = Bk[2:4] (2x1)
    #   C[:,k] = AL * BLk + AR * BRk
    #
    # AL * BLk: 2x2 * 2x1, rank 4 schoolbook
    # AR * BRk: 2x2 * 2x1, rank 4 schoolbook
    # Total for k=0..3: 4*4 = 16
    # For k=4: same, 4
    # Total: 20. That's better than 40 but not using Strassen.
    #
    # To use Strassen, we need 2x2 * 2x2. So group two k values.
    # For k in {0,1}: B01 = B[:, 0:2] (4x2)
    #   BL01 = B01[0:2, :] (2x2)
    #   BR01 = B01[2:4, :] (2x2)
    #   C[:, 0:2] = AL * BL01 + AR * BR01
    #
    # AL * BL01: 2x2 * 2x2 -> Strassen rank 7
    # AR * BR01: 2x2 * 2x2 -> Strassen rank 7
    # Total for k=0,1: 14
    #
    # Similarly for k=2,3: 14
    # For k=4: 2x2 * 2x1 + 2x2 * 2x1 = 4 + 4 = 8
    #
    # Total: 14 + 14 + 8 = 36.
    #
    # Now let me implement this.
    
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term.
        u_vals: dict mapping (i,j) -> value for A
        v_vals: dict mapping (j,k) -> value for B
        w_vals: dict mapping (i,k) -> value for C
        """
        global col
        for (i, j), val in u_vals.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            if val != 0:
                V[j * p + k][col] += val
        for (i, k), val in w_vals.items():
            if val != 0:
                W[k * n + i][col] += val
        col += 1
    
    def add_strassen(i_base, j_base, k_base):
        """Add Strassen decomposition for 2x2*2x2 subproblem.
        i_base: base index for i (always 0 since n=2)
        j_base: base index for j in A and B (0 or 2)
        k_base: base index for k (0, 2, or 4)
        
        A sub-block: rows i_base:i_base+2, cols j_base:j_base+2
        B sub-block: rows j_base:j_base+2, cols k_base:k_base+2
        C sub-block: rows i_base:i_base+2, cols k_base:k_base+2
        """
        global col
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Map to global: i_global = i_base + i, j_global = j_base + j, k_global = k_base + k
        
        # M1 = (A00+A11)*(B00+B11)
        add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
                 {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1})
        
        # M2 = (A10+A11)*B00
        add_term({(1,0): 1, (1,1): 1}, {(0,0): 1},
                 {(1,0): 1, (1,1): 1})
        
        # M3 = A00*(B10-B11)
        add_term({(0,0): 1}, {(1,0): 1, (1,1): -1},
                 {(0,1): 1, (1,0): 1})
        
        # M4 = A11*(B01-B00)
        add_term({(1,1): 1}, {(0,0): -1, (0,1): 1},
                 {(0,0): 1, (1,0): 1})
        
        # M5 = (A00+A01)*B11
        add_term({(0,0): 1, (0,1): 1}, {(1,1): 1},
                 {(0,0): -1, (0,1): 1})
        
        # M6 = (A10-A00)*(B00+B10)
        add_term({(0,0): -1, (1,0): 1}, {(0,0): 1, (1,0): 1},
                 {(1,1): 1})
        
        # M7 = (A01-A11)*(B01+B11)
        add_term({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1},
                 {(0,1): 1, (1,1): -1})
    
    # Add Strassen for AL*BL01 (i_base=0, j_base=0, k_base=0)
    add_strassen(0, 0, 0)
    
    # Add Strassen for AR*BR01 (i_base=0, j_base=2, k_base=0)
    add_strassen(0, 2, 0)
    
    # Add Strassen for AL*BL23 (i_base=0, j_base=0, k_base=2)
    add_strassen(0, 0, 2)
    
    # Add Strassen for AR*BR23 (i_base=0, j_base=2, k_base=2)
    add_strassen(0, 2, 2)
    
    # For k=4: schoolbook for AL*B4 and AR*B4
    # AL*B4: A[i,0:2] * B[0:2,4] -> 2x2 * 2x1
    # Terms:
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    add_term({(0,0): 1}, {(0,4): 1}, {(0,4): 1})
    add_term({(0,1): 1}, {(1,4): 1}, {(0,4): 1})
    add_term({(1,0): 1}, {(0,4): 1}, {(1,4): 1})
    add_term({(1,1): 1}, {(1,4): 1}, {(1,4): 1})
    
    # AR*B4: A[i,2:4] * B[2:4,4] -> 2x2 * 2x1
    add_term({(0,2): 1}, {(2,4): 1}, {(0,4): 1})
    add_term({(0,3): 1}, {(3,4): 1}, {(0,4): 1})
    add_term({(1,2): 1}, {(2,4): 1}, {(1,4): 1})
    add_term({(1,3): 1}, {(3,4): 1}, {(1,4): 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
