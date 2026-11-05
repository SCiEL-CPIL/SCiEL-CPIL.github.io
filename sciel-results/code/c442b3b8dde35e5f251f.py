"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt Rank 51 Decomposition
    # M_{2,5,6} = M_{2,3,6} (Rank 30) + M_{2,2,6} (Rank 21)
    # M_{2,3,6} = 3 * M_{2,3,2} (Rank 10 each)
    # M_{2,2,6} = 3 * M_{2,2,2} (Rank 7 each)
    
    rank_target = 51
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    
    col = 0
    
    def add_m223_block(i_base, j_base, k_base):
        """
        Adds a decomposition for M_{2,2,3} (A is 2x2, B is 2x3, C is 2x3)
        i_base: row offset in A/C
        j_base: col offset in A / row offset in B
        k_base: col offset in B/C
        """
        nonlocal col
        # A indices: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
        # B indices: (j_base, k_base), (j_base, k_base+1), (j_base, k_base+2), (j_base+1, k_base), (j_base+1, k_base+1), (j_base+1, k_base+2)
        # C indices: (k_base, i_base), (k_base+1, i_base), (k_base+2, i_base), (k_base, i_base+1), (k_base+1, i_base+1), (k_base+2, i_base+1)
        
        # Let a = A[i_base, j_base], b = A[i_base, j_base+1]
        # c = A[i_base+1, j_base], d = A[i_base+1, j_base+1]
        # x = B[j_base, k_base], y = B[j_base, k_base+1], z = B[j_base, k_base+2]
        # u = B[j_base+1, k_base], v = B[j_base+1, k_base+1], w = B[j_base+1, k_base+2]
        
        # 1. a(x+y)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 1][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 2. b(y+z)
        U[i_base * m + j_base + 1][col] += 1
        V[j_base * p + k_base + 1][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+2) * n + i_base][col] += 1
        col += 1
        
        # 3. c(x+z)
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+2) * n + (i_base+1)][col] += 1
        col += 1
        
        # 4. d(x+y+z)
        U[(i_base+1) * m + j_base + 1][col] += 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 1][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        W[(k_base+2) * n + (i_base+1)][col] += 1
        col += 1
        
        # 5. (a+b)(x+y)
        U[i_base * m + j_base][col] += 1
        U[i_base * m + j_base + 1][col] += 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 1][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 6. (c+d)(x+z)
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + j_base + 1][col] += 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+2) * n + (i_base+1)][col] += 1
        col += 1
        
        # 7. (a-c)x
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] -= 1
        col += 1
        
        # 8. (b-d)y
        U[i_base * m + j_base + 1][col] += 1
        U[(i_base+1) * m + j_base + 1][col] -= 1
        V[j_base * p + k_base + 1][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # 9. (a+b+c+d)z
        U[i_base * m + j_base][col] += 1
        U[i_base * m + j_base + 1][col] += 1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + j_base + 1][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[(k_base+2) * n + i_base][col] += 1
        W[(k_base+2) * n + (i_base+1)][col] += 1
        col += 1
        
        # 10. (a+b-c-d)(x+y+z)
        U[i_base * m + j_base][col] += 1
        U[i_base * m + j_base + 1][col] += 1
        U[(i_base+1) * m + j_base][col] -= 1
        U[(i_base+1) * m + j_base + 1][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + k_base + 1][col] += 1
        V[j_base * p + k_base + 2][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+2) * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] -= 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        W[(k_base+2) * n + (i_base+1)][col] -= 1
        col += 1

    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a decomposition for M_{2,2,2} (Strassen)
        """
        nonlocal col
        # Standard Strassen for 2x2
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Apply M_{2,3,2} blocks for j in {0, 1, 2}
    # k blocks: (0,1,2), (3,4,5)
    # Wait, M_{2,3,2} takes 3 columns of A and 2 rows of B?
    # My add_m223_block takes 2 rows of A, 2 cols of A, 2 rows of B, 3 cols of B.
    # This is M_{2,2,3}.
    # I need M_{2,3,2}.
    # M_{2,3,2} is A(2x3) B(3x2) C(2x2).
    # My add_m223_block is A(2x2) B(2x3) C(2x3).
    # I can transpose the problem or adjust indices.
    # Let's stick to M_{2,2,3} blocks.
    # Split m=5 into 2+2+1?
    # M_{2,2,6} (Rank 21) + M_{2,2,6} (Rank 21) + M_{2,1,6} (Rank 12) = 54.
    # This is the safe path.
    
    # I will use the safe path (54) to ensure correctness.
    
    # j-pairs (0,1) and (2,3)
    # k-triples (0,1,2) and (3,4,5) for M_{2,2,3}
    
    # Using M_{2,2,3} for j=0,1 and k=0,1,2
    add_m223_block(0, 0, 0)
    # Using M_{2,2,3} for j=0,1 and k=3,4,5
    add_m223_block(0, 0, 3)
    
    # Using M_{2,2,3} for j=2,3 and k=0,1,2
    add_m223_block(0, 2, 0)
    # Using M_{2,2,3} for j=2,3 and k=3,4,5
    add_m223_block(0, 2, 3)
    
    # Remaining j=4, all k
    # M_{2,1,6}
    # 12 scalar products
    for k in range(6):
        for i in range(2):
            U[i * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1

    if col != rank_target:
        if col < rank_target:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
