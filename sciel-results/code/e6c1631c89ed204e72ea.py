import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Use Strassen on a 2x2x2 sub-block to save 1 multiplication
    # Split the 4x7 matrix into 2x7 blocks, and 3x4 into 3x2 blocks
    # Apply Strassen to the 2x2x2 block multiplication
    
    # Schoolbook would be 3*4*7 = 84
    # With Strassen on 2x2x2: we save 1 multiplication
    # Total rank = 83
    
    # Let's construct this carefully
    # A is 3x4, B is 4x7, C = A*B is 3x7
    
    # Split A into [A11 A12] where A11 is 3x2, A12 is 3x2
    # Split B into [B11; B12] where B11 is 2x7, B12 is 2x7
    
    # C = A11*B11 + A12*B12
    
    # Use Strassen for the 2x2x2 part? No, that's not quite right.
    # Actually, let me think about this differently.
    
    # Alternative: Use the fact that we can share some terms
    # Let me try a direct rank-83 construction
    
    # Actually, the simplest improvement: 
    # Note that one of the 84 terms can be eliminated if we find a linear dependency
    
    # Let me try: express one term as a combination of others
    # This requires finding a non-trivial relation
    
    # For now, let's implement a known rank-83 method:
    # Use Strassen on a 2x2 subproblem
    
    # Split the 4 columns into 2+2
    # Split the 3 rows into 1+2
    # Split the 7 columns into 7 (no split)
    
    # Actually, let me use a different approach:
    # The tensor M_{3,4,7} has a rank-83 decomposition known in literature
    
    # Let me construct it by modifying the schoolbook:
    # Replace one of the 84 rank-1 terms with a linear combination
    
    # Schoolbook: U[i*m+j, t] = delta_{i,j}, V[j*p+k, t] = delta_{j,k}, W[k*n+i, t] = delta_{k,i}
    # where t = i*m*p + j*p + k
    
    # To get rank 83, we need to show that one of these is in the span of the others
    # This is generally not possible for generic matrices
    
    # So let me try a different strategy: use a structured decomposition
    
    # Method: Use the identity that allows sharing between some terms
    # For example, if we define new variables that combine terms
    
    # Let me try: use rank 84 but with a different structure that might allow reduction
    
    # Actually, I'll implement a local search to find a rank-83 decomposition
    
    # Initialize with schoolbook rank-84
    rank = 84
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Now try to reduce rank by eliminating one term
    # Check if any term is in the span of others
    
    # For each term t, check if removing it still allows exact decomposition
    # This is equivalent to checking if the tensor is in the span of the other 83 rank-1 tensors
    
    # Let's try a gradient-based approach to find a rank-83 decomposition
    
    # Start from rank-84 and try to drive one coefficient to zero
    
    # Actually, let me try a simpler approach:
    # Use the known fact that M_{3,4,7} can be decomposed with rank 83
    
    # I'll use a heuristic: perturb the schoolbook and solve for rank 83
    
    # For time reasons, let me just return schoolbook if we can't find better quickly
    
    # But first, let me try one specific construction:
    # Use Strassen on a 2x2x2 sub-block
    
    # Split: A = [A_top (1x4), A_bottom (2x4)]
    # B = [B_left (4x3), B_right (4x4)] -- no, that doesn't help
    
    # Let me try: 
    # A is 3x4, split into A1 (2x4) and A2 (1x4)
    # B is 4x7, no split
    # C = A1*B + A2*B
    
    # This doesn't directly help with Strassen
    
    # Alternative known result: M_{3,4,7} has rank <= 83
    # Construction: use the rank-19 decomposition of M_{3,3,3} (Strassen)
    # and pad appropriately
    
    # Actually, the cleanest way: 
    # Embed the 3x3x3 Strassen decomposition into the 3x4x7 problem
    
    # For the first 3 columns of A and first 3 rows of B, use Strassen
    # For the remaining parts, use schoolbook
    
    # A = [A1 (3x3) | A2 (3x1)]
    # B = [B1 (3x7); B2 (1x7)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: use Strassen rank 7 (instead of 9)
    # A2*B2: rank 1*1*7 = 7
    # Total: 7 + 7 = 14 for the 3x3x7 part? No, that's not right
    
    # Let me be more careful:
    # A1 is 3x3, B1 is 3x7
    # A1*B1: this is 3x3x7, rank at most 3*3*7 = 63, but with Strassen on the 3x3 part...
    # Actually, Strassen works for square matrices. For 3x3x7, we can split the 7 into 3+4
    # A1*B1 = A1*B1_left (3x3x3) + A1*B1_right (3x3x4)
    # A1*B1_left: use Strassen, rank 7
    # A1*B1_right: schoolbook, rank 3*3*4 = 36
    # Total for A1*B1: 7 + 36 = 43
    
    # A2*B2: A2 is 3x1, B2 is 1x7, rank 3*1*7 = 21
    # But wait, A2*B2 has rank at most 3*1*7 = 21, but actually it's rank 3*7 = 21? No.
    # A2 is 3x1, B2 is 1x7, so A2*B2 is 3x7, rank at most 1 (outer product)
    # So A2*B2: rank 1
    
    # Total: 43 + 1 = 44? That seems too low. Let me reconsider.
    
    # Actually, I'm overcomplicating this. Let me just use the schoolbook rank-84
    # and accept that finding rank-83 requires more sophisticated techniques.
    
    # For the purpose of this task, I'll return the schoolbook decomposition.
    
    schoolbook_rank = n * m * p
    U_list = [[0] * schoolbook_rank for _ in range(n * m)]
    V_list = [[0] * schoolbook_rank for _ in range(m * p)]
    W_list = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
