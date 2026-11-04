import numpy as np

DIMENSIONS = [4, 4, 8]

def strassen_decomp(N):
    """
    Returns U, V, W for N x N x N matrix multiplication using Strassen's algorithm.
    U shape: (N*N, rank)
    V shape: (N*N, rank)
    W shape: (N*N, rank)
    """
    if N == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    half = N // 2
    # Recursively get decomposition for half x half
    U_sub, V_sub, W_sub = strassen_decomp(half)
    rank_sub = len(U_sub[0])
    
    # We need to construct U, V, W for N x N
    # The 7 Strassen products M1..M7 are computed using linear combinations of blocks.
    # Each Mk is a half x half multiplication.
    # So we need 7 * rank_sub columns in total.
    
    total_rank = 7 * rank_sub
    U = [[0] * total_rank for _ in range(N * N)]
    V = [[0] * total_rank for _ in range(N * N)]
    W = [[0] * total_rank for _ in range(N * N)]
    
    # Indices for blocks
    # A:
    # A11: i in [0, half), j in [0, half) -> idx i*N + j
    # A12: i in [0, half), j in [half, N) -> idx i*N + j
    # A21: i in [half, N), j in [0, half) -> idx i*N + j
    # A22: i in [half, N), j in [half, N) -> idx i*N + j
    
    # B:
    # B11: i in [0, half), j in [0, half)
    # B12: i in [0, half), j in [half, N)
    # B21: i in [half, N), j in [0, half)
    # B22: i in [half, N), j in [half, N)
    
    # C:
    # C11, C12, C21, C22 similarly.
    
    # Strassen formulas:
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # We will fill U, V, W by adding contributions from each Mk.
    # For each Mk, we use the U_sub, V_sub, W_sub.
    # The U_sub corresponds to the left operand of Mk.
    # The V_sub corresponds to the right operand of Mk.
    # The W_sub corresponds to the result of Mk.
    
    def add_block(U, V, W, op_U, op_V, op_W, sign_U, sign_V, sign_W, offset):
        # op_U: list of (row_idx_in_NxN, coeff) for the left operand
        # op_V: list of (row_idx_in_NxN, coeff) for the right operand
        # op_W: list of (row_idx_in_NxN, coeff) for the result
        # Actually, it's easier to map the sub-blocks.
        
        # The sub-decomposition U_sub has rows 0..half*half-1.
        # These correspond to the local indices of the half x half block.
        # We need to map local (i, j) to global (i, j) in the N x N matrix.
        
        pass

    # This manual mapping is error-prone. Let's use a helper to extract block indices.
    
    def get_block_indices(start_i, start_j, size):
        indices = []
        for i in range(size):
            for j in range(size):
                indices.append((start_i + i) * N + (start_j + j))
        return indices

    # Block indices
    A11_idx = get_block_indices(0, 0, half)
    A12_idx = get_block_indices(0, half, half)
    A21_idx = get_block_indices(half, 0, half)
    A22_idx = get_block_indices(half, half, half)
    
    B11_idx = get_block_indices(0, 0, half)
    B12_idx = get_block_indices(0, half, half)
    B21_idx = get_block_indices(half, 0, half)
    B22_idx = get_block_indices(half, half, half)
    
    C11_idx = get_block_indices(0, 0, half)
    C12_idx = get_block_indices(0, half, half)
    C21_idx = get_block_indices(half, 0, half)
    C22_idx = get_block_indices(half, half, half)
    
    # Sub-decomposition rows correspond to local indices 0..half*half-1.
    # Local index l corresponds to i=l//half, j=l%half.
    # Global index for A11 local (i,j) is (0+i)*N + (0+j).
    
    def map_sub_to_global(local_indices, block_indices):
        # local_indices: list of local row indices in sub-decomp
        # block_indices: list of global row indices for the block
        # Returns list of (global_idx, coeff)
        # Since the sub-decomp is for a generic half x half matrix,
        # the rows in U_sub are 0..half*half-1.
        # We assume the block is stored in row-major order.
        # So local row l maps to block_indices[l].
        return [(block_indices[l], 1) for l in local_indices]

    # We need to add 7 times the sub-decomp.
    for k in range(7):
        col_offset = k * rank_sub
        
        # Determine which blocks are involved in M_k and their signs
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # For each part, we add the sub-decomp rows to U, V, W.
        # U gets the left operand contributions.
        # V gets the right operand contributions.
        # W gets the result contributions.
        
        # Left operand blocks and signs
        if k == 0:
            left_blocks = [(A11_idx, 1), (A22_idx, 1)]
            right_blocks = [(B11_idx, 1), (B22_idx, 1)]
            result_blocks = [(C11_idx, 1), (C12_idx, 0), (C21_idx, 0), (C22_idx, 1)] # M1 contributes to C11 and C22
            # Wait, W_sub maps to the result of Mk.
            # The result of Mk is a half x half matrix.
            # We need to add W_sub to the appropriate C blocks.
            # M1 contributes to C11 (sign +1) and C22 (sign +1).
            # But W_sub is a single half x half matrix.
            # So we add W_sub to C11 and C22.
            
        # This is getting complicated to map W_sub to multiple C blocks.
        # It is easier to just build the full U, V, W by iterating over the sub-decomp rows.
        
    # Let's restart the construction loop with a clearer logic.
    
    # For each of the 7 Strassen products:
    for k in range(7):
        col_offset = k * rank_sub
        
        # Define the linear combination for the left operand (A part)
        # and right operand (B part)
        # and where the result (Mk) goes in C.
        
        # Mk is a half x half matrix.
        # U_sub rows correspond to the A-part of Mk.
        # V_sub rows correspond to the B-part of Mk.
        # W_sub rows correspond to the C-part of Mk (which is Mk itself).
        
        # We need to map:
        # U_sub row r -> U row global_idx_A
        # V_sub row r -> V row global_idx_B
        # W_sub row r -> W row global_idx_C
        
        # The global indices depend on which block A11, A12 etc. is used.
        
        # Let's define the contributions.
        # For U:
        # For each row r in U_sub (0..half*half-1):
        #   val = U_sub[r]
        #   We add val to U[global_idx, col_offset + r] for each block involved in the left operand.
        
        # This is inefficient. Let's pre-compute the mapping.
        
        # Left operand blocks
        if k == 0: # M1
            left = [(A11_idx, 1), (A22_idx, 1)]
            right = [(B11_idx, 1), (B22_idx, 1)]
            res = [(C11_idx, 1), (C22_idx, 1)]
        elif k == 1: # M2
            left = [(A21_idx, 1), (A22_idx, 1)]
            right = [(B11_idx, 1)]
            res = [(C21_idx, 1), (C22_idx, -1)]
        elif k == 2: # M3
            left = [(A11_idx, 1)]
            right = [(B12_idx, 1), (B22_idx, -1)]
            res = [(C12_idx, 1), (C22_idx, 1)]
        elif k == 3: # M4
            left = [(A22_idx, 1)]
            right = [(B21_idx, 1), (B11_idx, -1)]
            res = [(C11_idx, 1), (C21_idx, 1)]
        elif k == 4: # M5
            left = [(A11_idx, 1), (A12_idx, 1)]
            right = [(B22_idx, 1)]
            res = [(C11_idx, -1), (C12_idx, 1)]
        elif k == 5: # M6
            left = [(A21_idx, 1), (A11_idx, -1)]
            right = [(B11_idx, 1), (B12_idx, 1)]
            res = [(C22_idx, 1)]
        elif k == 6: # M7
            left = [(A12_idx, 1), (A22_idx, -1)]
            right = [(B21_idx, 1), (B22_idx, 1)]
            res = [(C11_idx, 1), (C12_idx, 0)] # M7 contributes to C11?
            # C11 = M1 + M4 - M5 + M7. Yes.
            # C12 = M3 + M5.
            # C21 = M2 + M4.
            # C22 = M1 - M2 + M3 + M6.
            # So M7 only contributes to C11.
            res = [(C11_idx, 1)]

        # Now add to U, V, W
        for r in range(half * half):
            u_val = U_sub[r]
            v_val = V_sub[r]
            w_val = W_sub[r]
            
            # U
            for block_idx, sign in left:
                for local_i in range(half):
                    for local_j in range(half):
                        local_r = local_i * half + local_j
                        if local_r == r:
                            global_idx = block_idx[local_r]
                            U[global_idx][col_offset + r] += sign * u_val
            # This inner loop is inefficient.
            # r is the local row index.
            # block_idx is a list of global indices.
            # So global_idx = block_idx[r].
            
            # Let's optimize.
            pass

    # I will rewrite the loop to be efficient.
    
    U = [[0] * total_rank for _ in range(N * N)]
    V = [[0] * total_rank for _ in range(N * N)]
    W = [[0] * total_rank for _ in range(N * N)]
    
    for k in range(7):
        col_offset = k * rank_sub
        
        if k == 0:
            left = [(A11_idx, 1), (A22_idx, 1)]
            right = [(B11_idx, 1), (B22_idx, 1)]
            res = [(C11_idx, 1), (C22_idx, 1)]
        elif k == 1:
            left = [(A21_idx, 1), (A22_idx, 1)]
            right = [(B11_idx, 1)]
            res = [(C21_idx, 1), (C22_idx, -1)]
        elif k == 2:
            left = [(A11_idx, 1)]
            right = [(B12_idx, 1), (B22_idx, -1)]
            res = [(C12_idx, 1), (C22_idx, 1)]
        elif k == 3:
            left = [(A22_idx, 1)]
            right = [(B21_idx, 1), (B11_idx, -1)]
            res = [(C11_idx, 1), (C21_idx, 1)]
        elif k == 4:
            left = [(A11_idx, 1), (A12_idx, 1)]
            right = [(B22_idx, 1)]
            res = [(C11_idx, -1), (C12_idx, 1)]
        elif k == 5:
            left = [(A21_idx, 1), (A11_idx, -1)]
            right = [(B11_idx, 1), (B12_idx, 1)]
            res = [(C22_idx, 1)]
        elif k == 6:
            left = [(A12_idx, 1), (A22_idx, -1)]
            right = [(B21_idx, 1), (B22_idx, 1)]
            res = [(C11_idx, 1)]

        for r in range(half * half):
            u_val = U_sub[r]
            v_val = V_sub[r]
            w_val = W_sub[r]
            
            for block_idx, sign in left:
                global_idx = block_idx[r]
                U[global_idx][col_offset + r] += sign * u_val
                
            for block_idx, sign in right:
                global_idx = block_idx[r]
                V[global_idx][col_offset + r] += sign * v_val
                
            for block_idx, sign in res:
                global_idx = block_idx[r]
                W[global_idx][col_offset + r] += sign * w_val
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Get 4x4x4 Strassen decomposition
    U_s, V_s, W_s = strassen_decomp(4)
    rank_s = len(U_s[0]) # 49
    
    total_rank = 2 * rank_s # 98
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill first half (B1, C1)
    for i in range(n * m):
        for c in range(rank_s):
            U[i][c] = U_s[i][c]
            U[i][c + rank_s] = U_s[i][c]
            
    # V mapping
    # V_s rows: j*4 + k' (k' in 0..3)
    # V rows: j*8 + k (k in 0..7)
    for j in range(m):
        for k_prime in range(4):
            v_val_first = V_s[j * 4 + k_prime]
            v_val_second = V_s[j * 4 + k_prime]
            
            row_first = j * 8 + k_prime
            row_second = j * 8 + 4 + k_prime
            
            for c in range(rank_s):
                V[row_first][c] = v_val_first[c]
                V[row_second][c + rank_s] = v_val_second[c]
                
    # W mapping
    # W_s rows: k'*4 + i (k' in 0..3)
    # W rows: k*4 + i (k in 0..7)
    for k_prime in range(4):
        for i in range(n):
            w_val_first = W_s[k_prime * 4 + i]
            w_val_second = W_s[k_prime * 4 + i]
            
            row_first = k_prime * 4 + i
            row_second = (4 + k_prime) * 4 + i
            
            for c in range(rank_s):
                W[row_first][c] = w_val_first[c]
                W[row_second][c + rank_s] = w_val_second[c]
                
    return {"U": U, "V": V, "W": W}
