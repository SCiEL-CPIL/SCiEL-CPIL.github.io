"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # Strategy: Block decomposition.
    # Split p=10 into 5 blocks of size 2.
    # Rank of M_{2,3,2} is 11.
    # Total rank = 5 * 11 = 55.
    
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    # Initialize U, V, W as NumPy arrays for efficiency
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Indices:
    # A: i in {0,1}, j in {0,1,2}
    # B: j in {0,1,2}, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    # W index: k * n + i
    
    # Strassen part (7 terms) for A[:, 0:2] and B[0:2, :]
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    outer_terms = [
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for term in all_terms:
            col = col_offset
            # Fill U
            for i, j, val in term['U']:
                U[i * m + j, col] += val
            # Fill V
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global, col] += val
            # Fill W
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i, col] += val
            col_offset += 1
            
    # Convert to lists of integers if possible
    U_out = [[int(U[i][j]) if abs(U[i][j] - round(U[i][j])) < 1e-9 else U[i][j] for j in range(total_rank)] for i in range(n*m)]
    V_out = [[int(V[i][j]) if abs(V[i][j] - round(V[i][j])) < 1e-9 else V[i][j] for j in range(total_rank)] for i in range(m*p)]
    W_out = [[int(W[i][j]) if abs(W[i][j] - round(W[i][j])) < 1e-9 else W[i][j] for j in range(total_rank)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
