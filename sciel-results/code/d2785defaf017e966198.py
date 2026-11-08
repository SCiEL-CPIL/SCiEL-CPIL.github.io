"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Decompose the 2x3 * 3x9 multiplication into three 2x3 * 3x3 block multiplications.
    # The tensor T[2,3,9] is the direct sum of three T[2,3,3] tensors.
    # Rank is sum of ranks of blocks. Schoolbook rank for each 2x3*3x3 is 18.
    # Total rank = 3 * 18 = 54.
    
    n, m, p = DIMENSIONS
    # p is 9, which is 3 * 3. Let p_block = 3.
    num_blocks = p // 3
    p_block = 3
    
    rank = 0
    U = []
    V = []
    W = []
    
    # We will build the matrices column by column for each block
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # Initialize lists to append columns
    # It's easier to build column vectors and then transpose/assemble.
    # Or just append to row lists.
    
    U_rows = [[0] * 0 for _ in range(n * m)] # Placeholder, we will grow these
    # Actually, let's just collect columns first.
    cols_U = []
    cols_V = []
    cols_W = []
    
    for b in range(num_blocks):
        # Block b corresponds to columns b*3 to (b+1)*3 - 1 in V and W
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    # Local indices within the block
                    # Global indices:
                    # U index: i*m + j
                    # V index: j*p + (b*p_block + k)  <-- Note: V index is j*p + k_global
                    # W index: k_global * n + i       <-- Note: W index is k_global * n + i
                    
                    u_idx = i * m + j
                    v_idx = j * p + (b * p_block + k)
                    w_idx = (b * p_block + k) * n + i
                    
                    # Create column vectors of zeros
                    col_u = [0] * (n * m)
                    col_v = [0] * (m * p)
                    col_w = [0] * (p * n)
                    
                    col_u[u_idx] = 1
                    col_v[v_idx] = 1
                    col_w[w_idx] = 1
                    
                    cols_U.append(col_u)
                    cols_V.append(col_v)
                    cols_W.append(col_w)
                    
    rank = len(cols_U)
    
    # Transpose columns to rows
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
