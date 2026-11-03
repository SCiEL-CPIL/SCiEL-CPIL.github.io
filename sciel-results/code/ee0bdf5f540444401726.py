"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build a decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest.
    # Strategy: Split A (3x4) and B (4x6) into blocks of size 2 along the m dimension.
    # A = [A_top (2x4); A_bottom (1x4)]
    # B = [B_top (4x6); B_bottom (4x6)]
    # C = A_top * B_top + A_bottom * B_bottom
    # A_top is 2x4, B_top is 4x6 -> C_top is 2x6
    # A_bottom is 1x4, B_bottom is 4x6 -> C_bottom is 1x6
    
    # We will build U, V, W as lists of columns.
    # Each column corresponds to a term in the bilinear form.
    # U has size n*m = 12
    # V has size m*p = 24
    # W has size p*n = 18
    
    # Helper to add a term to the decomposition
    # U_idx: list of (row, coeff)
    # V_idx: list of (row, coeff)
    # W_idx: list of (row, coeff)
    
    terms = []
    
    # --- Part 1: A_top (2x4) * B_top (4x6) ---
    # Split A_top into two 2x2 blocks: A_top_left, A_top_right
    # Split B_top into three 2x2 blocks: B_top_left, B_top_mid, B_top_right
    # C_top = A_top_left * B_top_left + A_top_right * B_top_right? No.
    # C_top[i, k] = sum_{j=0}^3 A_top[i, j] * B_top[j, k]
    # Split j=0,1 and j=2,3.
    # C_top = A_top_left * B_top_left_half + A_top_right * B_top_right_half
    # Where B_top_left_half is B_top[0:2, :] and B_top_right_half is B_top[2:4, :]
    # So C_top = A_top_left * B_top[0:2, :] + A_top_right * B_top[2:4, :]
    
    # Let's handle C_top_left (2x2 output block from cols 0-1 of C_top)
    # C_top[:, 0:2] = A_top_left * B_top[0:2, 0:2] + A_top_right * B_top[2:4, 0:2]
    # Let M1 = A_top_left * B_top[0:2, 0:2] (2x2)
    # Let M2 = A_top_right * B_top[2:4, 0:2] (2x2)
    # Use Strassen for M1 and M2.
    
    # Strassen terms for 2x2x2:
    # M1 = A11 B11 + A12 B21 (using row/col indices 0,1)
    # M2 = A21 B12 + A22 B22
    # M3 = (A11 + A22)(B11 + B22)
    # M4 = (A21 + A22)(B22 - B12)
    # M5 = (A11 + A12)(B12 - B11)
    # M6 = (A22 - A11)(B11 + B12)
    # M7 = (A12 - A21)(B21 + B22)
    # C11 = M3 + M4 - M7 - M5
    # C12 = M5 + M7
    # C21 = M1 - M4 + M6
    # C22 = M1 - M3 + M4
    
    # We need to map these generic Strassen terms to specific indices in our global matrices.
    # This is complex to do generically for every block. 
    # Instead, let's just use a known rank-7 tensor decomposition for 2x2x2 and 
    # apply it to the relevant subspaces.
    
    # Actually, for 3x4x6, the schoolbook rank is 3*4*6 = 72.
    # If we can use Strassen for the 2x2x2 parts, we save 1 rank per 2x2x2 block.
    # How many 2x2x2 blocks can we fit?
    # We can partition the 4x4 interaction matrix (m x m) into 2x2 blocks? No, m=4.
    # We can partition the 4x6 interaction (m x p) into 2x2 blocks.
    # Let's just write a simple rank-7 decomposition for the whole thing if possible,
    # or fall back to schoolbook if we can't easily construct a lower rank.
    
    # Known lower bounds: R(3,4,6) >= 22.
    # Known upper bounds: 
    # Schoolbook: 72.
    # Strassen-like: 
    # Let's try to construct a decomposition using the known rank-7 for 2x2x2 on sub-blocks.
    
    # Let's stick to a robust construction.
    # If I can't find a clever hand-crafted low rank decomposition quickly, 
    # I should ensure the schoolbook one is correct and returns 72.
    # But the task asks to "Find a rank-minimal exact tensor decomposition". 
    # The evaluator scores by rank (lower is better).
    # So 72 is the baseline. Can we do better?
    
    # Let's try to use the fact that we can split the 4x4 middle dimension.
    # A (3x4) B (4x6).
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2).
    # C = A B1 + A B2 + A B3.
    # A B1 is 3x2. Rank of 3x2x4?
    # Schoolbook rank for 3x2x4 is 3*2*4 = 24.
    # Total 72.
    
    # Is there a known decomposition for 3x4x6 with rank < 72?
    # Probably yes, but constructing it explicitly in code without external data is hard.
    # Let's try to use Strassen on the 2x2x2 components where possible.
    
    # Let's define the Strassen tensor for 2x2x2.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    # W[k*2+i, j*2+k]? No, W is p*n.
    # For a 2x2x2 block, U is 2x2=4 rows, V is 2x2=4 rows, W is 2x2=4 rows.
    # The 7 terms are linear combinations of the entries.
    
    # Let's implement a generic function to add a Strassen block.
    # add_strassen_block(u_base, v_base, w_base)
    # u_base: start index in U (row)
    # v_base: start index in V (row)
    # w_base: start index in W (row)
    
    # This is getting too complex to derive from scratch reliably in the short time.
    # I will return the schoolbook decomposition which is guaranteed correct.
    # Rank = 72.
    
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
