"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def _build_2x2x6_strassen_block(A_rows, B_cols, offset=0):
    """
    Build a rank-21 tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    A_rows: list of 2 row indices into the 3x3 A matrix
    B_cols: list of 2 column indices into the 3x6 B matrix
    Returns (U_block, V_block, W_block) where each is a list of (row, value) pairs,
    to be merged into the global matrices.
    """
    # Strassen for 2x2x2 (rank 7), applied 3 times for the 6 columns (3 pairs of 2)
    # Each Strassen multiplication computes (2x2) * (2x2) -> 7 products
    # We have (2x2) * (2x6) -> split B into three (2x2) blocks
    # Total rank: 3 * 7 = 21
    
    U_entries = []  # list of (row_in_U, value)
    V_entries = []  # list of (row_in_V, value)
    W_entries = []  # list of (row_in_W, value)
    
    a0, a1 = A_rows  # A indices
    b0, b1 = B_cols  # B indices (each is a pair of columns into B)
    
    # Strassen for 2x2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # For our case, we're computing A_block (2x2) * B_block (2x2)
    # A_block[i][j] corresponds to A[a_i][j] where a_i is the i-th of A_rows
    # B_block[i][j] corresponds to B[i][B_cols[j]] where i is 0 or 1 (rows of B)
    
    # We need to be careful about the indices.
    # A is 3x3, B is 3x6.
    # A_rows = [i, i+1] (two consecutive rows of A)
    # B_cols = [j, j+1] (two consecutive columns of B)
    # The multiplication is A[A_rows, 0:2] * B[0:2, B_cols]
    # Wait, no. The multiplication is:
    #   A is 3x3, B is 3x6.
    #   We want C = A * B, where C is 3x6.
    #   C[i][k] = sum_j A[i][j] * B[j][k]
    #   For the top-left 2x2 block of A multiplying the top-left 2x2 block of B:
    #     i in {0,1}, k in {0,1}, j in {0,1}
    #   But we need to handle the full B which is 3x6.
    
    # Let me reconsider the structure.
    # C = A * B
    # C[i][k] = sum_{j=0}^{2} A[i][j] * B[j][k]
    # 
    # For the main block: i in {0,1}, k in {0,5} (all 6 columns)
    # We split the j dimension and k dimension.
    # j in {0,1} for the first part, j=2 for the cross terms.
    # k in {0,1}, {2,3}, {4,5} for Strassen.
    
    # For the main block (i in {0,1}, j in {0,1}):
    #   A_block = A[0:2, 0:2] (2x2)
    #   B_block = B[0:2, :] (2x6)
    #   Split B_block into three 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    #   Each gives a 2x2 result, total 6 columns.
    
    # For each 2x2x2 block, we use Strassen (rank 7).
    # The A indices are A[0:2, 0:2], so A_rows = [0, 1], and the j indices are 0, 1.
    # The B rows are 0, 1 (B[0:2, ...]).
    
    # In the global indexing:
    #   U is indexed by (i, j) -> i*m + j = i*3 + j
    #   V is indexed by (j, k) -> j*p + k = j*6 + k
    #   W is indexed by (k, i) -> k*n + i = k*3 + i
    
    # For the main block:
    #   A_rows = [0, 1] (i indices)
    #   j indices for A: [0, 1]
    #   B rows (j indices for B): [0, 1]
    #   B column groups: [(0,1), (2,3), (4,5)]
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    j_indices = [0, 1]  # The j indices for the A rows and B rows
    i_indices = [0, 1]  # The i indices for the result
    
    col_groups = [(0, 1), (2, 3), (4, 5)]
    
    for (k0, k1) in col_groups:
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2]
        # A00 = A[0][0], A01 = A[0][1], A10 = A[1][0], A11 = A[1][1]
        # B00 = B[0][k0], B01 = B[0][k1], B10 = B[1][k0], B11 = B[1][k1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: (0, 0) and (1, 1) with +1
        # V: (0, k0) and (1, k1) with +1
        # W: (k0, 0), (k1, 0), (k0, 1), (k1, 1) with +1, -1, -1, +1
        U_entries.append((0 * 3 + 0, 1))  # A00
        U_entries.append((1 * 3 + 1, 1))  # A11
        V_entries.append((0 * 6 + k0, 1))  # B00
        V_entries.append((1 * 6 + k1, 1))  # B11
        W_entries.append((k0 * 3 + 0, 1))   # C00
        W_entries.append((k1 * 3 + 0, -1))  # C01
        W_entries.append((k0 * 3 + 1, -1))  # C10
        W_entries.append((k1 * 3 + 1, 1))   # C11
        
        # M2 = (A10 + A11) * B00
        U_entries.append((1 * 3 + 0, 1))  # A10
        U_entries.append((1 * 3 + 1, 1))  # A11
        V_entries.append((0 * 6 + k0, 1))  # B00
        W_entries.append((k0 * 3 + 1, 1))   # C10
        W_entries.append((k1 * 3 + 1, 1))   # C11
        
        # M3 = A00 * (B10 - B11)
        U_entries.append((0 * 3 + 0, 1))  # A00
        V_entries.append((1 * 6 + k0, 1))  # B10
        V_entries.append((1 * 6 + k1, -1)) # B11
        W_entries.append((k0 * 3 + 0, 1))   # C00
        W_entries.append((k1 * 3 + 0, 1))   # C01
        
        # M4 = A11 * (B01 - B00)
        U_entries.append((1 * 3 + 1, 1))  # A11
        V_entries.append((0 * 6 + k1, 1))  # B01
        V_entries.append((0 * 6 + k0, -1)) # B00
        W_entries.append((k0 * 3 + 1, 1))   # C10
        W_entries.append((k1 * 3 + 1, 1))   # C11
        
        # M5 = (A00 + A01) * B11
        U_entries.append((0 * 3 + 0, 1))  # A00
        U_entries.append((0 * 3 + 1, 1))  # A01
        V_entries.append((1 * 6 + k1, 1))  # B11
        W_entries.append((k0 * 3 + 0, 1))   # C00
        W_entries.append((k1 * 3 + 0, 1))   # C01
        
        # M6 = (A10 - A00) * (B00 + B10)
        U_entries.append((1 * 3 + 0, 1))  # A10
        U_entries.append((0 * 3 + 0, -1)) # A00
        V_entries.append((0 * 6 + k0, 1))  # B00
        V_entries.append((1 * 6 + k0, 1))  # B10
        W_entries.append((k0 * 3 + 1, 1))   # C10
        W_entries.append((k1 * 3 + 1, 1))   # C11
        
        # M7 = (A01 - A11) * (B01 + B11)
        U_entries.append((0 * 3 + 1, 1))  # A01
        U_entries.append((1 * 3 + 1, -1)) # A11
        V_entries.append((0 * 6 + k1, 1))  # B01
        V_entries.append((1 * 6 + k1, 1))  # B11
        W_entries.append((k0 * 3 + 0, 1))   # C00
        W_entries.append((k1 * 3 + 0, 1))   # C01
    
    return U_entries, V_entries, W_entries


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    U_entries = []  # list of (row, value)
    V_entries = []
    W_entries = []
    
    # Part 1: Main block - A[0:2, 0:2] * B[0:2, :] (rank 21)
    u1, v1, w1 = _build_2x2x6_strassen_block([0, 1], None)
    U_entries.extend(u1)
    V_entries.extend(v1)
    W_entries.extend(w1)
    
    # Part 2: Cross term - A[0:2, 2] * B[2, :] (rank 12)
    # C[i][k] += A[i][2] * B[2][k] for i in {0,1}, k in {0,5}
    for i in [0, 1]:
        for k in range(6):
            U_entries.append((i * 3 + 2, 1))  # A[i][2]
            V_entries.append((2 * 6 + k, 1))  # B[2][k]
            W_entries.append((k * 3 + i, 1))  # C[k][i]
    
    # Part 3: Cross term - A[2, 0:2] * B[0:2, :] (rank 12)
    # C[2][k] += sum_{j=0}^{1} A[2][j] * B[j][k] for k in {0,5}
    for j in [0, 1]:
        for k in range(6):
            U_entries.append((2 * 3 + j, 1))  # A[2][j]
            V_entries.append((j * 6 + k, 1))  # B[j][k]
            W_entries.append((k * 3 + 2, 1))  # C[k][2]
    
    # Part 4: Corner - A[2, 2] * B[2, :] (rank 6)
    for k in range(6):
        U_entries.append((2 * 3 + 2, 1))  # A[2][2]
        V_entries.append((2 * 6 + k, 1))  # B[2][k]
        W_entries.append((k * 3 + 2, 1))  # C[k][2]
    
    # Now, merge the entries into dense matrices
    # Count the total rank
    rank = len(set(range(len(U_entries))))  # This is wrong, let me count properly
    
    # Each "entry" in the lists corresponds to one rank component.
    # But multiple entries in U_entries, V_entries, W_entries can share the same rank index.
    # Actually, in my construction, each group of 8 W_entries, etc., corresponds to one Strassen product.
    # Let me restructure to properly track rank indices.
    
    # Let me redo this more carefully.
    U_entries = []  # list of (rank_idx, row, value)
    V_entries = []
    W_entries = []
    
    rank_idx = 0
    
    # Part 1: Main block - A[0:2, 0:2] * B[0:2, :] (rank 21)
    col_groups = [(0, 1), (2, 3), (4, 5)]
    
    for (k0, k1) in col_groups:
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2]
        
        # M1 = (A00 + A11) * (B00 + B11)
        U_entries.append((rank_idx, 0 * 3 + 0, 1))  # A00
        U_entries.append((rank_idx, 1 * 3 + 1, 1))  # A11
        V_entries.append((rank_idx, 0 * 6 + k0, 1))  # B00
        V_entries.append((rank_idx, 1 * 6 + k1, 1))  # B11
        W_entries.append((rank_idx, k0 * 3 + 0, 1))   # C00
        W_entries.append((rank_idx, k1 * 3 + 0, -1))  # C01
        W_entries.append((rank_idx, k0 * 3 + 1, -1))  # C10
        W_entries.append((rank_idx, k1 * 3 + 1, 1))   # C11
        rank_idx += 1
        
        # M2 = (A10 + A11) * B00
        U_entries.append((rank_idx, 1 * 3 + 0, 1))  # A10
        U_entries.append((rank_idx, 1 * 3 + 1, 1))  # A11
        V_entries.append((rank_idx, 0 * 6 + k0, 1))  # B00
        W_entries.append((rank_idx, k0 * 3 + 1, 1))   # C10
        W_entries.append((rank_idx, k1 * 3 + 1, 1))   # C11
        rank_idx += 1
        
        # M3 = A00 * (B10 - B11)
        U_entries.append((rank_idx, 0 * 3 + 0, 1))  # A00
        V_entries.append((rank_idx, 1 * 6 + k0, 1))  # B10
        V_entries.append((rank_idx, 1 * 6 + k1, -1)) # B11
        W_entries.append((rank_idx, k0 * 3 + 0, 1))   # C00
        W_entries.append((rank_idx, k1 * 3 + 0, 1))   # C01
        rank_idx += 1
        
        # M4 = A11 * (B01 - B00)
        U_entries.append((rank_idx, 1 * 3 + 1, 1))  # A11
        V_entries.append((rank_idx, 0 * 6 + k1, 1))  # B01
        V_entries.append((rank_idx, 0 * 6 + k0, -1)) # B00
        W_entries.append((rank_idx, k0 * 3 + 1, 1))   # C10
        W_entries.append((rank_idx, k1 * 3 + 1, 1))   # C11
        rank_idx += 1
        
        # M5 = (A00 + A01) * B11
        U_entries.append((rank_idx, 0 * 3 + 0, 1))  # A00
        U_entries.append((rank_idx, 0 * 3 + 1, 1))  # A01
        V_entries.append((rank_idx, 1 * 6 + k1, 1))  # B11
        W_entries.append((rank_idx, k0 * 3 + 0, 1))   # C00
        W_entries.append((rank_idx, k1 * 3 + 0, 1))   # C01
        rank_idx += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U_entries.append((rank_idx, 1 * 3 + 0, 1))  # A10
        U_entries.append((rank_idx, 0 * 3 + 0, -1)) # A00
        V_entries.append((rank_idx, 0 * 6 + k0, 1))  # B00
        V_entries.append((rank_idx, 1 * 6 + k0, 1))  # B10
        W_entries.append((rank_idx, k0 * 3 + 1, 1))   # C10
        W_entries.append((rank_idx, k1 * 3 + 1, 1))   # C11
        rank_idx += 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        U_entries.append((rank_idx, 0 * 3 + 1, 1))  # A01
        U_entries.append((rank_idx, 1 * 3 + 1, -1)) # A11
        V_entries.append((rank_idx, 0 * 6 + k1, 1))  # B01
        V_entries.append((rank_idx, 1 * 6 + k1, 1))  # B11
        W_entries.append((rank_idx, k0 * 3 + 0, 1))   # C00
        W_entries.append((rank_idx, k1 * 3 + 0, 1))   # C01
        rank_idx += 1
    
    # Part 2: Cross term - A[0:2, 2] * B[2, :] (rank 12)
    for i in [0, 1]:
        for k in range(6):
            U_entries.append((rank_idx, i * 3 + 2, 1))  # A[i][2]
            V_entries.append((rank_idx, 2 * 6 + k, 1))  # B[2][k]
            W_entries.append((rank_idx, k * 3 + i, 1))  # C[k][i]
            rank_idx += 1
    
    # Part 3: Cross term - A[2, 0:2] * B[0:2, :] (rank 12)
    for j in [0, 1]:
        for k in range(6):
            U_entries.append((rank_idx, 2 * 3 + j, 1))  # A[2][j]
            V_entries.append((rank_idx, j * 6 + k, 1))  # B[j][k]
            W_entries.append((rank_idx, k * 3 + 2, 1))  # C[k][2]
            rank_idx += 1
    
    # Part 4: Corner - A[2, 2] * B[2, :] (rank 6)
    for k in range(6):
        U_entries.append((rank_idx, 2 * 3 + 2, 1))  # A[2][2]
        V_entries.append((rank_idx, 2 * 6 + k, 1))  # B[2][k]
        W_entries.append((rank_idx, k * 3 + 2, 1))  # C[k][2]
        rank_idx += 1
    
    # Build the dense matrices
    U = [[0.0] * rank_idx for _ in range(n * m)]
    V = [[0.0] * rank_idx for _ in range(m * p)]
    W = [[0.0] * rank_idx for _ in range(p * n)]
    
    for (r, row, val) in U_entries:
        U[row][r] += val
    for (r, row, val) in V_entries:
        V[row][r] += val
    for (r, row, val) in W_entries:
        W[row][r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
