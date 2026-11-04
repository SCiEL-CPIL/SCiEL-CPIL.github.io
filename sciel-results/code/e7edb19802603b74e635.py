"""Program for matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def generate_partitions(n, max_part=2):
    """Generate all partitions of n into parts of size 1 or max_part."""
    if n == 0:
        yield []
        return
    if n >= 1:
        for rest in generate_partitions(n - 1, max_part):
            yield [1] + rest
    if n >= max_part:
        for rest in generate_partitions(n - max_part, max_part):
            yield [max_part] + rest

def get_strassen_decomposition():
    """
    Returns U, V, W for 2x2x2 Strassen multiplication.
    Indices:
    A: 2x2 -> indices 0,1 (row), 2,3 (col) in U (size 4)
    B: 2x2 -> indices 0,1 (row), 2,3 (col) in V (size 4)
    C: 2x2 -> indices 0,1 (row), 2,3 (col) in W (size 4)
    Convention: A[i, j], B[j, k], C[i, k]
    U index: i*2 + j
    V index: j*2 + k
    W index: k*2 + i  <-- Note: Task says W[k*n+i, r]. For 2x2, n=2, so W[k*2+i, r].
    Wait, the task says "output factor follows the official trace convention W[k*n+i,r]".
    In the seed program: W[k * n + i][col] = 1.
    So for a block of size p x n (where p is cols of B, n is rows of A/C),
    the W index is k * n + i.
    
    For 2x2x2 block:
    A is 2x2. U indices: i*2 + j.
    B is 2x2. V indices: j*2 + k.
    C is 2x2. W indices: k*2 + i.
    
    Let's verify the seed program's W indexing:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i][col] = 1
    Yes, W index is k*n + i.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00 + A11)(B10 - B11)
    # A00: U[0], A11: U[3]
    # B10: V[2], B11: V[3]
    U[0][0] = 1
    U[3][0] = 1
    V[2][0] = 1
    V[3][0] = -1
    # C00 += M1, C11 -= M1
    W[0*2+0][0] = 1   # C00 (k=0, i=0)
    W[1*2+1][0] = -1  # C11 (k=1, i=1)
    
    # M2 = (A10 + A11) B00
    # A10: U[2], A11: U[3]
    # B00: V[0]
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    # C10 += M2
    W[0*2+1][1] = 1   # C10 (k=0, i=1)
    
    # M3 = A00 (B01 - B11)
    # A00: U[0]
    # B01: V[1], B11: V[3]
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    # C01 += M3
    W[1*2+0][2] = 1   # C01 (k=1, i=0)
    
    # M4 = A11 B11
    # A11: U[3]
    # B11: V[3]
    U[3][3] = 1
    V[3][3] = 1
    # C00 += M4, C10 += M4
    W[0*2+0][3] = 1   # C00
    W[0*2+1][3] = 1   # C10
    
    # M5 = (A00 + A01) B11
    # A00: U[0], A01: U[1]
    # B11: V[3]
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    # C00 -= M5
    W[0*2+0][4] = -1  # C00
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10: U[2], A00: U[0]
    # B00: V[0], B01: V[1]
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    # C11 += M6
    W[1*2+1][5] = 1   # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01: U[1], A11: U[3]
    # B10: V[2], B11: V[3]
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    # C00 += M7, C01 += M7
    W[0*2+0][6] = 1   # C00
    W[1*2+0][6] = 1   # C01
    
    return U, V, W

def get_schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    
    # Precompute Strassen
    U_str, V_str, W_str = get_strassen_decomposition()
    
    # Generate partitions
    parts_n = list(generate_partitions(n, 2))
    parts_m = list(generate_partitions(m, 2))
    parts_p = list(generate_partitions(p, 2))
    
    best_rank = float('inf')
    best_config = None
    
    for pn in parts_n:
        for pm in parts_m:
            for pp in parts_p:
                # Check if partitions are compatible?
                # The middle dimension m must be partitioned consistently for A and B.
                # In our formulation, we partition the indices 0..m-1 into subsets.
                # The same partition is used for cols of A and rows of B.
                # So pm is the partition of the middle dimension.
                # pn is partition of rows of A (and C).
                # pp is partition of cols of B (and C).
                
                rank = 0
                valid = True
                for i_size in pn:
                    for j_size in pm:
                        for k_size in pp:
                            # Block size: i_size x j_size x k_size
                            if i_size == 1 and j_size == 1 and k_size == 1:
                                rank += 1
                            elif i_size == 2 and j_size == 2 and k_size == 2:
                                rank += 7
                            else:
                                # Use schoolbook for mixed sizes
                                rank += i_size * j_size * k_size
                                # If we want to be smarter, we could decompose further,
                                # but for now, schoolbook is safe.
                                # Actually, if i_size=2, j_size=1, k_size=1, schoolbook is 2.
                                # Is there a better way? 2x1x1 is just 2 mults.
                                # 2x2x1 is 4 mults.
                                # 2x1x2 is 4 mults.
                                # So schoolbook is optimal for non-2x2x2 blocks of size <=2.
                                
                if rank < best_rank:
                    best_rank = rank
                    best_config = (pn, pm, pp)
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_config is None:
        # Fallback to schoolbook
        pn, pm, pp = [n], [m], [p]
        best_rank = n*m*p
    else:
        pn, pm, pp = best_config
    
    # Construct the final matrices
    total_rank = best_rank
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    i_offset = 0
    for i_size in pn:
        j_offset = 0
        for j_size in pm:
            k_offset = 0
            for k_size in pp:
                # Block at (i_offset, j_offset, k_offset) with size (i_size, j_size, k_size)
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Copy Strassen
                    for r in range(7):
                        for u_idx in range(4):
                            val = U_str[u_idx][r]
                            if val != 0:
                                global_u_idx = (i_offset + u_idx//2) * m + (j_offset + u_idx%2)
                                U[global_u_idx][col_offset + r] = val
                        for v_idx in range(4):
                            val = V_str[v_idx][r]
                            if val != 0:
                                global_v_idx = (j_offset + v_idx//2) * p + (k_offset + v_idx%2)
                                V[global_v_idx][col_offset + r] = val
                        for w_idx in range(4):
                            val = W_str[w_idx][r]
                            if val != 0:
                                # w_idx is k*2 + i
                                w_k = w_idx // 2
                                w_i = w_idx % 2
                                global_w_idx = (k_offset + w_k) * n + (i_offset + w_i)
                                W[global_w_idx][col_offset + r] = val
                    col_offset += 7
                else:
                    # Schoolbook for this block
                    block_rank = i_size * j_size * k_size
                    for ii in range(i_size):
                        for jj in range(j_size):
                            for kk in range(k_size):
                                global_u_idx = (i_offset + ii) * m + (j_offset + jj)
                                global_v_idx = (j_offset + jj) * p + (k_offset + kk)
                                global_w_idx = (k_offset + kk) * n + (i_offset + ii)
                                U[global_u_idx][col_offset] = 1
                                V[global_v_idx][col_offset] = 1
                                W[global_w_idx][col_offset] = 1
                                col_offset += 1
            k_offset += k_size
        j_offset += j_size
    i_offset += i_size
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
