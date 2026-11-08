"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # Split p=10 into 5 blocks of size 2.
    # Each block is M_{2,3,2}.
    # Rank of M_{2,3,2} is 11 (7 for M_{2,2,2} part + 4 for M_{2,1,2} part).
    # Total rank = 5 * 11 = 55.
    
    num_blocks = 5
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Indices for A: i in {0,1}, j in {0,1,2}
    # Indices for B: j in {0,1,2}, k in {0,1} (local k)
    # Indices for C: i in {0,1}, k in {0,1} (local k)
    # W index: k * n + i
    
    # Strassen part (7 terms) for A[:, 0:2] and B[0:2, :]
    # Local indices: i, j in {0,1}, k in {0,1}
    strassen_terms = [
        # (U_coeffs: list of (i,j, val), V_coeffs: list of (j,k, val), W_coeffs: list of (k,i, val))
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00, C11
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 +=, C11 -=
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)] # C01 +=, C11 +=
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 +=, C10 +=
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 -=, C01 +=
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 +=
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)] # C00 +=
        }
    ]
    
    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    # Local indices: i in {0,1}, j=2, k in {0,1}
    outer_terms = [
        # i=0, k=0
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        # i=0, k=1
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        # i=1, k=0
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        # i=1, k=1
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            # Fill U
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            # Fill V
            # Local k maps to global k = b * block_size + k_local
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            # Fill W
            # Local k maps to global k = b * block_size + k_local
            # W index is k * n + i
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
