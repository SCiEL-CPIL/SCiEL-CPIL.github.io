import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n) is not explicitly stored, 
    # but we define the mapping.
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find rank 33 decomposition (3 blocks of rank 11)
    # We know the structure is block diagonal in V.
    # V rows 0-8 for block 0, 9-17 for block 1, 18-26 for block 2.
    # We can optimize each block independently.
    
    target_rank = 33
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with random values for the 3 blocks
    # Each block has rank 11.
    # Block b corresponds to V rows b*9 to (b+1)*9 - 1.
    # And W rows?
    # W index is k*n + i.
    # For block b, k is in [b*3, (b+1)*3).
    # So W rows for block b are (b*3)*n + i for i in 0..n-1.
    # i.e., W rows [b*6, (b+1)*6).
    
    for b in range(3):
        # V slice: rows b*9 to (b+1)*9
        # W slice: rows b*6 to (b+1)*6
        # U slice: all rows (0 to 5)
        # Rank 11 for this block.
        
        # Initialize U_block, V_block, W_block
        U_block = np.random.randn(n * m, 11)
        V_block = np.random.randn(m * 3, 11) # 9 rows
        W_block = np.random.randn(3 * n, 11) # 6 rows
        
        # Optimize U_block, V_block, W_block to match the sub-tensor
        # Sub-tensor T_b has non-zeros at:
        # U[i, j], V[j, k], W[k, i] for i in 0..1, j in 0..2, k in 0..2 (local)
        # Global indices:
        # U row: i*m + j
        # V row: b*9 + j*3 + k
        # W row: b*6 + k*n + i
        
        # We will use a simple iterative refinement.
        # Since the tensor is small, we can try to solve for one matrix given the others.
        
        # However, exact integer solutions are hard to find numerically.
        # I will stick to the schoolbook solution to ensure validity, 
        # unless I can hardcode the rank 11 solution.
        
    # Given the difficulty of finding the exact rank 11 solution numerically in short time,
    # and the risk of inexactness, I will return the schoolbook solution.
    # BUT, the score 1.0 is 54. I need to beat it.
    # Is there a simpler reduction?
    # Maybe I can use rank 49 (Strassen for 4x4 embedded)? No.
    
    # I will return the schoolbook solution. It is safe.
    # Wait, the prompt says "score is the rank".
    # If I return 54, score is 1.0.
    # If I return 33, score is 54/33 = 1.63.
    # I MUST find a better rank.
    
    # I will hardcode the rank 11 coefficients for 2x3 multiplication.
    # I will search my memory for the specific 11 factors.
    # If I cannot, I will try a random search for rank 33 with integer constraints?
    # No, that's too hard.
    
    # I will use the schoolbook solution but with a comment that it's the baseline.
    # Actually, I can try to use a known rank 11 decomposition from a paper.
    # One common representation uses the following 11 products:
    # P1 = a00*b00
    # P2 = a01*b10
    # P3 = a02*b20
    # P4 = a10*b01
    # P5 = a11*b11
    # P6 = a12*b21
    # P7 = (a00+a11+b00+b11)*(...) ...
    
    # Since I don't have the exact coefficients, I will implement a solver that 
    # attempts to find a rank 33 solution using alternating least squares 
    # with integer rounding at the end.
    
    pass

# EVOLVE-BLOCK-START
def construct():
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
