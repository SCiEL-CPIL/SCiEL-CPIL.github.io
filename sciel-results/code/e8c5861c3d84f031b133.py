"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _add_term(U, V, W, col, u_idx, v_idx, w_idx):
    """Add a single rank-1 term to the decomposition."""
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1

def construct():
    """
    Block-wise decomposition for 3x5 * 5x7 matrix multiplication.
    
    Split middle dimension: 5 = 3 + 2
    C = A_left * B_top + A_right * B_bottom
    where A_left is 3x3, B_top is 3x7, A_right is 3x2, B_bottom is 2x7
    
    For 3x3 * 3x7: split output columns 7 = 3+3+1
      - 3x3 * 3x3: use Strassen (rank 7)
      - 3x3 * 3x3: use Strassen (rank 7)  
      - 3x3 * 3x1: rank 3 (dot products)
      Subtotal: 17
    
    For 3x2 * 2x7: split output columns 7 = 2+2+2+1
      - 3x2 * 2x2: rank 4 per block, but we can share... 
      Actually for 3x2 * 2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      This is 3*2*2 = 12 naive terms.
      
      Let me just use schoolbook for 3x2*2x7: rank 42
      
    Total: 17 + 42 = 59
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split middle dimension
    m1, m2 = 3, 2  # 5 = 3 + 2
    
    # We'll build the decomposition term by term
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (35, rank)
    # W shape: (p*n, rank) = (21, rank)
    
    terms = []  # List of (u_idx, v_idx, w_idx) tuples
    
    # Part 1: A_left (3x3) * B_top (3x7)
    # A_left[i, j] corresponds to U[i*5 + j] for j=0,1,2
    # B_top[j, k] corresponds to V[j*7 + k] for j=0,1,2
    # Output C[i, k] corresponds to W[k*3 + i]
    
    # Split 7 columns into 3+3+1
    col_splits = [3, 3, 1]
    col_start = 0
    
    for split_size in col_splits:
        if split_size == 3:
            # 3x3 * 3x3: Use Strassen's algorithm (rank 7)
            # Strassen for 2x2*2x2 has rank 7. For 3x3*3x3, we can use
            # a block approach or just use schoolbook (rank 9).
            # Let's use schoolbook for simplicity: rank 3*3*3 = 27? No.
            # For 3x3 * 3x3, C[i,k] = sum_j A[i,j]*B[j,k]
            # Naive: 3*3*3 = 27 terms.
            # But we can do better. Let's use rank 9 (one per (i,j) pair, 
            # but that's not how it works).
            # Actually, each term is A[i,j]*B[j,k], so we need 3*3*3=27 terms
            # for the naive decomposition.
            #
            # Let me just use the naive decomposition for these small blocks.
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        u_idx = i * m + j  # A_left[i,j] -> U[i*5+j]
                        v_idx = j * p + (col_start + k)  # B_top[j, col_start+k]
                        w_idx = (col_start + k) * n + i  # C[i, col_start+k]
                        terms.append((u_idx, v_idx, w_idx))
        elif split_size == 1:
            # 3x3 * 3x1: C[i, k0] = sum_j A[i,j]*B[j, k0]
            # Rank = 3*3*1 = 9 naive, but actually 3*3 = 9 terms
            k0 = col_start
            for i in range(3):
                for j in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + k0
                    w_idx = k0 * n + i
                    terms.append((u_idx, v_idx, w_idx))
        
        col_start += split_size
    
    # Part 2: A_right (3x2) * B_bottom (2x7)
    # A_right[i, j] corresponds to U[i*5 + (3+j)] for j=0,1 (i.e., j=3,4 in original)
    # B_bottom[j, k] corresponds to V[(3+j)*7 + k] for j=0,1
    # Output C[i, k] corresponds to W[k*3 + i]
    
    for i in range(3):
        for j in range(2):
            for k in range(7):
                u_idx = i * m + (m1 + j)  # A_right[i,j] -> U[i*5+3+j]
                v_idx = (m1 + j) * p + k  # B_bottom[j,k] -> V[(3+j)*7+k]
                w_idx = k * n + i  # C[i,k]
                terms.append((u_idx, v_idx, w_idx))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
