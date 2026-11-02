"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # We split the 9 columns of B into 4 blocks of 2 columns and 1 block of 1 column.
    # For each 2x2 block, we use Strassen's algorithm (rank 7).
    # For the 2x1 block, we use schoolbook (rank 4).
    # Total rank = 4 * 7 + 4 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # Indices for A: 0:11, 1:12, 2:21, 3:22 (i*m + j)
    # Indices for B: j*p + k
    # Indices for W: k*n + i
    
    # Block 1: cols 0, 1 (k1=0, k2=1)
    # Block 2: cols 2, 3 (k1=2, k2=3)
    # Block 3: cols 4, 5 (k1=4, k2=5)
    # Block 4: cols 6, 7 (k1=6, k2=7)
    # Block 5: col 8 (k=8)
    
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    r_offset = 0
    
    for k1, k2 in blocks:
        # Strassen terms
        # 1. (A11+A22)(B11+B22) -> C11, C22
        add_term(0, 1, 0, 0, 0, 0, r_offset) # U[0]=A11, V[0]=B11? No, need to map indices correctly.
        
        # Let's use a more robust mapping function
        def get_u(i, j): return i * m + j
        def get_v(j, k): return j * p + k
        def get_w(k, i): return k * n + i

        # Reset local state for this block's rank 7
        # We will fill U, V, W directly
        
        # Term 1: (A11+A22)(B11+B22)
        r = r_offset
        U[get_u(0,0)][r] += 1
        U[get_u(1,1)][r] += 1
        V[get_v(0,k1)][r] += 1
        V[get_v(1,k2)][r] += 1
        W[get_w(k1,0)][r] += 1
        W[get_w(k2,1)][r] += 1
        
        # Term 2: (A21+A22)B11
        r = r_offset + 1
        U[get_u(1,0)][r] += 1
        U[get_u(1,1)][r] += 1
        V[get_v(0,k1)][r] += 1
        W[get_w(k1,1)][r] += 1
        W[get_w(k2,1)][r] += -1
        
        # Term 3: A11(B12-B22)
        r = r_offset + 2
        U[get_u(0,0)][r] += 1
        V[get_v(0,k2)][r] += 1
        V[get_v(1,k2)][r] += -1
        W[get_w(k2,0)][r] += 1
        W[get_w(k2,1)][r] += 1
        
        # Term 4: A22(B21-B11)
        r = r_offset + 3
        U[get_u(1,1)][r] += 1
        V[get_v(1,k1)][r] += 1
        V[get_v(0,k1)][r] += -1
        W[get_w(k1,0)][r] += 1
        W[get_w(k1,1)][r] += 1
        
        # Term 5: (A11+A12)B22
        r = r_offset + 4
        U[get_u(0,0)][r] += 1
        U[get_u(0,1)][r] += 1
        V[get_v(1,k2)][r] += 1
        W[get_w(k1,0)][r] += -1
        W[get_w(k2,0)][r] += 1
        
        # Term 6: (A21-A11)(B11+B12)
        r = r_offset + 5
        U[get_u(1,0)][r] += 1
        U[get_u(0,0)][r] += -1
        V[get_v(0,k1)][r] += 1
        V[get_v(0,k2)][r] += 1
        W[get_w(k2,1)][r] += 1
        
        # Term 7: (A12-A22)(B21+B22)
        r = r_offset + 6
        U[get_u(0,1)][r] += 1
        U[get_u(1,1)][r] += -1
        V[get_v(1,k1)][r] += 1
        V[get_v(1,k2)][r] += 1
        W[get_w(k1,0)][r] += 1
        
        r_offset += 7

    # Block 5: col 8 (k=8)
    # Schoolbook
    k = 8
    # Term 8: A11 B1k -> C1k
    r = r_offset
    U[get_u(0,0)][r] += 1
    V[get_v(0,k)][r] += 1
    W[get_w(k,0)][r] += 1
    
    # Term 9: A12 B2k -> C1k
    r = r_offset + 1
    U[get_u(0,1)][r] += 1
    V[get_v(1,k)][r] += 1
    W[get_w(k,0)][r] += 1
    
    # Term 10: A21 B1k -> C2k
    r = r_offset + 2
    U[get_u(1,0)][r] += 1
    V[get_v(0,k)][r] += 1
    W[get_w(k,1)][r] += 1
    
    # Term 11: A22 B2k -> C2k
    r = r_offset + 3
    U[get_u(1,1)][r] += 1
    V[get_v(1,k)][r] += 1
    W[get_w(k,1)][r] += 1
    
    r_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
