"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0

# EVOLVE-BLOCK-START

def _get_rank7_decomp():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    u: 0=00, 1=01, 2=10, 3=11 (i*2+j)
    v: 0=00, 1=01, 2=10, 3=11 (j*2+k)
    io: 0=00, 1=01, 2=10, 3=11 (k*2+i)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # P5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = -1
    
    # P6 = (c-a)(e+h)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[0, 5] = 1
    
    # P7 = (b-c)(g+h)
    U[1, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _get_rank49_decomp():
    """
    Returns U, V, W for M_{4,4,4} using recursive Strassen.
    U shape (16, 49), V shape (16, 49), W shape (16, 49).
    """
    U2, V2, W2 = _get_rank7_decomp()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen terms definitions
    # P1: A00+A11, B00+B11 -> C00(+1), C11(+1)
    # P2: A10+A11, B00 -> C10(+1), C11(+1)
    # P3: A00, B01-B11 -> C01(+1), C11(-1)
    # P4: A11, B10-B00 -> C00(-1), C10(+1)
    # P5: A00+A01, B11 -> C01(+1), C11(-1)
    # P6: A10-A00, B00+B11 -> C00(+1)
    # P7: A01-A10, B10+B11 -> C00(-1)
    
    # Block index maps for 2x2 blocks within 4x4
    # A blocks:
    # A00: i in 0,1; j in 0,1 -> u in 0,1,4,5
    # A01: i in 0,1; j in 2,3 -> u in 2,3,6,7
    # A10: i in 2,3; j in 0,1 -> u in 8,9,12,13
    # A11: i in 2,3; j in 2,3 -> u in 10,11,14,15
    # B blocks:
    # B00: j in 0,1; k in 0,1 -> v in 0,1,4,5
    # B01: j in 0,1; k in 2,3 -> v in 2,3,6,7
    # B10: j in 2,3; k in 0,1 -> v in 8,9,12,13
    # B11: j in 2,3; k in 2,3 -> v in 10,11,14,15
    # C blocks:
    # C00: k in 0,1; i in 0,1 -> io in 0,1,4,5
    # C01: k in 0,1; i in 2,3 -> io in 2,3,6,7
    # C10: k in 2,3; i in 0,1 -> io in 8,9,12,13
    # C11: k in 2,3; i in 2,3 -> io in 10,11,14,15
    
    def get_block_indices(block_id, dim):
        # dim is 'u', 'v', 'io'
        if dim == 'u':
            # i, j
            if block_id == 0: return [0, 1, 4, 5]
            if block_id == 1: return [2, 3, 6, 7]
            if block_id == 2: return [8, 9, 12, 13]
            if block_id == 3: return [10, 11, 14, 15]
        elif dim == 'v':
            # j, k
            if block_id == 0: return [0, 1, 4, 5]
            if block_id == 1: return [2, 3, 6, 7]
            if block_id == 2: return [8, 9, 12, 13]
            if block_id == 3: return [10, 11, 14, 15]
        elif dim == 'io':
            # k, i
            if block_id == 0: return [0, 1, 4, 5]
            if block_id == 1: return [2, 3, 6, 7]
            if block_id == 2: return [8, 9, 12, 13]
            if block_id == 3: return [10, 11, 14, 15]
        return []

    # Strassen term definitions:
    # term: (A_blocks, signs_A, B_blocks, signs_B, C_blocks, signs_C)
    # A_blocks: list of block ids
    # signs_A: list of signs
    # etc.
    
    terms = [
        # P1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # Simplified below
    ]
    
    # Let's do it explicitly for clarity and correctness
    # r is 0..6
    # s is 0..6
    
    for r in range(7):
        # Determine A, B, C block contributions for P_r
        if r == 0: # P1
            a_blocks = [(0, 1), (3, 1)]
            b_blocks = [(0, 1), (3, 1)]
            c_blocks = [(0, 1), (3, 1)]
        elif r == 1: # P2
            a_blocks = [(2, 1), (3, 1)]
            b_blocks = [(0, 1)]
            c_blocks = [(2, 1), (3, 1)]
        elif r == 2: # P3
            a_blocks = [(0, 1)]
            b_blocks = [(1, 1), (3, -1)]
            c_blocks = [(1, 1), (3, -1)]
        elif r == 3: # P4
            a_blocks = [(3, 1)]
            b_blocks = [(2, 1), (0, -1)]
            c_blocks = [(0, -1), (2, 1)]
        elif r == 4: # P5
            a_blocks = [(0, 1), (1, 1)]
            b_blocks = [(3, 1)]
            c_blocks = [(1, 1), (3, -1)]
        elif r == 5: # P6
            a_blocks = [(2, 1), (0, -1)]
            b_blocks = [(0, 1), (3, 1)]
            c_blocks = [(0, 1)]
        elif r == 6: # P7
            a_blocks = [(1, 1), (2, -1)]
            b_blocks = [(2, 1), (3, 1)]
            c_blocks = [(0, -1)]
            
        for s in range(7):
            col = r * 7 + s
            
            # U construction
            # For each A block involved in P_r, add the corresponding U2 row scaled by sign
            for bid, sign in a_blocks:
                u_indices = get_block_indices(bid, 'u')
                # Map local 2x2 indices to global
                # Local indices 0,1,2,3 correspond to u_indices[0..3]
                for local_u in range(4):
                    val = U2[local_u, s] * sign
                    if val != 0:
                        U[u_indices[local_u], col] += val
                        
            # V construction
            for bid, sign in b_blocks:
                v_indices = get_block_indices(bid, 'v')
                for local_v in range(4):
                    val = V2[local_v, s] * sign
                    if val != 0:
                        V[v_indices[local_v], col] += val
                        
            # W construction
            for bid, sign in c_blocks:
                io_indices = get_block_indices(bid, 'io')
                for local_io in range(4):
                    val = W2[local_io, s] * sign
                    if val != 0:
                        W[io_indices[local_io], col] += val
                        
    return U, V, W

def _get_rank16_decomp():
    """
    Returns U, V, W for M_{4,4,1}.
    U shape (16, 16), V shape (4, 16), W shape (4, 16).
    Note: V has 4 rows (j*1+k, k=0).
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    c = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, c] = 1
            V[j, c] = 1
            W[i, c] = 1
            c += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get components
    U49, V49, W49 = _get_rank49_decomp()
    U16, V16, W16 = _get_rank16_decomp()
    
    # Combine into rank 114
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: k=0..3
    # Map v indices: std v' = 4j+k -> new v = 9j+k
    # Map io indices: std io' = 4k+i -> new io = 9k+i? No.
    # Output C is 4x9. io = k*4 + i.
    # In M_{4,4,4}, io' = k'*4 + i.
    # Here k' = k. So io = k*4 + i. Same mapping.
    
    for c in range(49):
        # U is same
        U[:, c] = U49[:, c]
        
        # V mapping
        for j in range(4):
            for k in range(4):
                v_std = 4 * j + k
                v_new = 9 * j + k
                V[v_new, c] = V49[v_std, c]
                
        # W mapping
        for k in range(4):
            for i in range(4):
                io_std = 4 * k + i
                io_new = 4 * k + i # io = k*n+i, n=4. So io = 4k+i.
                W[io_new, c] = W49[io_std, c]
                
    # Block 2: k=4..7
    # v_std = 4j + (k-4)
    # v_new = 9j + k
    # io_std = 4(k-4) + i
    # io_new = 4k + i
    
    for c in range(49):
        col = 49 + c
        U[:, col] = U49[:, c]
        
        for j in range(4):
            for k in range(4, 8):
                k_std = k - 4
                v_std = 4 * j + k_std
                v_new = 9 * j + k
                V[v_new, col] = V49[v_std, c]
                
        for k in range(4, 8):
            k_std = k - 4
            for i in range(4):
                io_std = 4 * k_std + i
                io_new = 4 * k + i
                W[io_new, col] = W49[io_std, c]
                
    # Block 3: k=8
    # v_std = j (since p=1, v=j*1+0=j)
    # v_new = 9j + 8
    # io_std = i (since p=1, io=0*4+i=i? No. io = k*n+i. k=0 for M_{4,4,1}. So io=i.)
    # Wait, M_{4,4,1} output is 4x1. io = 0*4+i = i.
    # In M_{4,4,9}, io = 8*4+i = 32+i.
    
    for c in range(16):
        col = 98 + c
        U[:, col] = U16[:, c]
        
        for j in range(4):
            v_std = j
            v_new = 9 * j + 8
            V[v_new, col] = V16[v_std, c]
            
        for i in range(4):
            io_std = i
            io_new = 32 + i
            W[io_new, col] = W16[io_std, c]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
