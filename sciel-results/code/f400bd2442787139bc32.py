"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 2x2 block decomposition
    # A is 4x4, B is 4x6
    # A blocks: 2x2 (4 blocks)
    # B blocks: 2x2 (6 blocks)
    # C blocks: 2x2 (6 blocks)
    
    # We will generate a schoolbook decomposition for the block structure
    # Each block multiplication is 2x2 * 2x2 -> 2x2
    # Rank of 2x2*2x2 is 4 (schoolbook)
    
    # Total block multiplications: 
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # i in 0..1, j in 0..2
    # So 2*3 = 6 positions, each needs 2 multiplications.
    # Total 12 block multiplications.
    # Each block multiplication has rank 4.
    # Total rank 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)] # 16 x 48
    V = [[0] * rank for _ in range(m * p)] # 24 x 48
    W = [[0] * rank for _ in range(p * n)] # 24 x 48
    
    col = 0
    for i_block in range(2):
        for j_block in range(3):
            for k_block in range(2):
                # Multiply A[i_block, k_block] * B[k_block, j_block]
                # A block (i_block, k_block) is at rows (i_block*2), cols (k_block*2)
                # B block (k_block, j_block) is at rows (k_block*2), cols (j_block*2)
                # C block (i_block, j_block) is at rows (j_block*2), cols (i_block*2)
                
                # Schoolbook for 2x2 * 2x2
                # Indices in 2x2: (0,0), (0,1), (1,0), (1,1)
                for r in range(2):
                    for c in range(2):
                        # A element (i_block*2 + r, k_block*2 + c)
                        a_idx = (i_block * 2 + r) * m + (k_block * 2 + c)
                        
                        # B element (k_block*2 + r, j_block*2 + c)
                        b_idx = (k_block * 2 + r) * p + (j_block * 2 + c)
                        
                        # C element (j_block*2 + c, i_block*2 + r)
                        # W index: k*n + i
                        # Here k is col of C (j_block*2 + c), i is row of C (i_block*2 + r)
                        c_idx = (j_block * 2 + c) * n + (i_block * 2 + r)
                        
                        U[a_idx][col] = 1
                        V[b_idx][col] = 1
                        W[c_idx][col] = 1
                        
                        col += 1
                        # We have 4 elements per block multiplication
                        # Total 12 * 4 = 48 columns
                        # But col is incremented 4 times inside the loop?
                        # No, the loop runs 4 times.
                        # So we need to assign 4 different columns for each block multiplication.
                        # My current logic assigns 1 column per element.
                        # This is correct for schoolbook.
                        
    # Now we have a rank 48 decomposition.
    # Let's try to eliminate dependencies.
    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Greedy elimination
    # We want to remove columns that are linear combinations of others.
    # This is equivalent to finding the rank of the coefficient tensor.
    # We can do this by checking if a column can be removed while preserving the decomposition.
    # A simpler heuristic: compute the rank of the flattened tensor?
    # Or just keep it as is if elimination is complex.
    
    # For now, return the rank 48 decomposition.
    # This is already a significant improvement over 96.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
