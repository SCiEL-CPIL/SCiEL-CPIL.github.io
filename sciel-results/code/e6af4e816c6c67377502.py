"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p

    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))

    # Schoolbook decomposition
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try to improve using Strassen-style block decomposition
    # Split the 4x5 matrix into blocks of 2x2 and 2x3
    # A (4x5) = [A11 A12; A21 A22] where A11,A21 are 2x2, A12,A22 are 2x3
    # B (5x6) = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C (4x6) = [C11 C12; C21 C22] where Cij are 2x3 blocks

    # For the 2x2 * 2x6 part, we can use Strassen on 2x2 * 2x2 blocks
    # But 2x6 is not a power of 2, so let's try a different approach

    # Let's try to use the fact that rank(2x2x2) = 7 (Strassen)
    # Split A into 2x2 blocks and B into 2x2 blocks where possible

    # Actually, let's try a more systematic approach:
    # Use block decomposition with 2x2 blocks for A and B
    # A (4x5): split into [A_top (2x5); A_bottom (2x5)]
    # B (5x6): split into [B_left (5x4) B_right (5x2)]
    # C (4x6): split into [C_left (4x4) C_right (4x2)]

    # C_left = A * B_left (4x4)
    # C_right = A * B_right (4x2)

    # For C_left (4x4), we can use Strassen: rank 49
    # For C_right (4x2), schoolbook: rank 32

    # Total: 49 + 32 = 81

    # Let's implement Strassen for 4x4x4
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix multiplication with rank 49."""
        # Strassen for 4x4 uses 7 multiplications of 2x2 blocks
        # Each 2x2 multiplication has rank 7 (Strassen)
        # Total: 7 * 7 = 49
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 2x2x2 decomposition
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # For 4x4, we apply this to 2x2 blocks of 2x2 matrices
        # Let's build it step by step
        
        col = 0
        # The 4x4 matrix has 4x4 = 16 elements
        # We'll use the standard Strassen recursion
        
        # Actually, let's just use the known rank-49 decomposition
        # This is complex, so let's try a simpler approach first
        
        # For now, return schoolbook for 4x4x4 (rank 64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        
        return U, V, W

    # Let's try a different approach: use the known result that
    # rank(4x5x6) can be improved over schoolbook
    # Try rank 108 using Strassen on 2x2 blocks
    
    # Split A (4x5) into 2x2 blocks:
    # A = [A11 A12 A13; A21 A22 A23] where each Aij is 2x2 (except last column)
    # Actually, 4x5 doesn't divide evenly into 2x2 blocks
    
    # Let's try: A (4x5) = [A_left (4x4) A_right (4x1)]
    # B (5x6) = [B_top (2x6); B_bottom (3x6)]
    
    # This is getting complicated. Let's just try to find a better rank
    # by using the fact that some tensor decompositions have lower rank.
    
    # For now, return the schoolbook decomposition as it's guaranteed correct
    return best_result
# EVOLVE-BLOCK-END
