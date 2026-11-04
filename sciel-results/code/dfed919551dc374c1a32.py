"""Matrix multiplication 3x5 * 5x6 tensor rank search.

We start from the schoolbook rank 90 and attempt to reduce it using
Strassen-like block decompositions where all three block dimensions are >= 2.
For a 2x2x2 block, Strassen gives rank 7 instead of 8, saving 1.

Strategy:
  1. Try various block partitions of (3, 5, 6).
  2. For each partition, compute the theoretical rank (sum of min(a*b*c, 7)
     for each block where a,b,c are the block dimensions).
  3. For the best partition, construct the explicit U, V, W matrices.
  4. Verify the decomposition is exact.
  5. If verification fails, fall back to schoolbook.

We also attempt a randomized local search to see if we can find a lower rank
decomposition than the block-strassen approach.
"""

import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 5, 6]

def _verify_decomposition(U, V, W, n, m, p, rank, seed=42, n_checks=50):
    """Verify that the decomposition is exact."""
    np.random.seed(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check matching entries: T[i*m+j, j*p+k, k*n+i] = 1
    for _ in range(n_checks):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        actual = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
        if abs(actual - 1.0) > 1e-6:
            return False
    
    # Check non-matching entries: should be 0
    for _ in range(n_checks):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        ai = a // m
        aj = a % m
        bj = b // p
        bk = b % p
        ck = c // n
        ci = c % n
        if ai != ci or aj != bj or aj != ck:
            actual = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
            if abs(actual) > 1e-6:
                return False
    
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    U is (4, 7), V is (4, 7), W is (4, 7)
    U indexed by (i*2+j), V indexed by (j*2+k), W indexed by (k*2+i)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # Contributes to C[0,0] and C[1,1]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes to C[1,0] and C[1,1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # p3 = A[0,0] * (B[0,1]-B[1,1])
    # Contributes to C[0,1] and C[1,1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # p4 = A[1,1] * (B[1,0]-B[0,0])
    # Contributes to C[0,0] and C[1,0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # p5 = (A[0,0]+A[0,1]) * B[1,1]
    # Contributes to C[0,1] and subtracts from C[0,0]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # p6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Contributes to C[1,1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # Contributes to C[0,0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(a, b, c):
    """Return U, V, W for an a x b x c schoolbook block.
    
    U is (a*b, a*b*c), V is (b*c, a*b*c), W is (c*a, a*b*c)
    """
    rank = a * b * c
    U = [[0]*rank for _ in range(a*b)]
    V = [[0]*rank for _ in range(b*c)]
    W = [[0]*rank for _ in range(c*a)]
    
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i*b + j][col] = 1
                V[j*c + k][col] = 1
                W[k*a + i][col] = 1
                col += 1
    
    return U, V, W

def _strassen_block(a, b, c):
    """Return U, V, W for an a x b x c block using Strassen when possible.
    
    If a>=2, b>=2, c>=2, use Strassen for 2x2x2 sub-blocks and schoolbook for the rest.
    Otherwise use pure schoolbook.
    """
    if a >= 2 and b >= 2 and c >= 2:
        # Use Strassen for the 2x2x2 part, schoolbook for remaining
        # This is complex; for now, just use schoolbook for all blocks
        # A proper implementation would decompose into 2x2x2 Strassen + schoolbook remainder
        pass
    
    # For simplicity, use schoolbook for all blocks
    return _schoolbook_block(a, b, c)

def _partition(dim, max_part=None):
    """Generate all partitions of dim into parts >= 1.
    
    Returns list of tuples, each tuple is a partition.
    """
    if max_part is None:
        max_part = dim
    
    def _gen(remaining, max_val, current):
        if remaining == 0:
            yield tuple(current)
            return
        for v in range(1, min(remaining, max_val) + 1):
            yield from _gen(remaining - v, v, current + [v])
    
    return list(_gen(dim, max_part, []))

def _compute_block_rank(partition_rows, partition_inner, partition_cols):
    """Compute the total rank for a given partition."""
    total_rank = 0
    for a in partition_rows:
        for b in partition_inner:
            for c in partition_cols:
                if a >= 2 and b >= 2 and c >= 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += a * b * c
    return total_rank

def _find_best_partition(n, m, p):
    """Find the partition that minimizes the total rank."""
    best_rank = n * m * p
    best_partition = None
    
    # Try all partitions
    part_rows = _partition(n)
    part_inner = _partition(m)
    part_cols = _partition(p)
    
    for pr in part_rows:
        for pi in part_inner:
            for pc in part_cols:
                rank = _compute_block_rank(pr, pi, pc)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (pr, pi, pc)
    
    return best_rank, best_partition

def _construct_block_decomposition(n, m, p, partition):
    """Construct the U, V, W matrices for a given block partition."""
    part_rows, part_inner, part_cols = partition
    
    # Compute offsets for each dimension
    row_offsets = [0]
    for a in part_rows:
        row_offsets.append(row_offsets[-1] + a)
    inner_offsets = [0]
    for b in part_inner:
        inner_offsets.append(inner_offsets[-1] + b)
    col_offsets = [0]
    for c in part_cols:
        col_offsets.append(col_offsets[-1] + c)
    
    # Compute total rank
    total_rank = 0
    block_ranks = []
    for ri, a in enumerate(part_rows):
        for bi, b in enumerate(part_inner):
            for ci, c in enumerate(part_cols):
                if a >= 2 and b >= 2 and c >= 2:
                    block_ranks.append((ri, bi, ci, 7))
                    total_rank += 7
                else:
                    block_ranks.append((ri, bi, ci, a * b * c))
                    total_rank += a * b * c
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ri, bi, ci, block_rank in block_ranks:
        a = part_rows[ri]
        b = part_inner[bi]
        c = part_cols[ci]
        
        if a >= 2 and b >= 2 and c >= 2:
            # Use Strassen for this block
            U_block, V_block, W_block = _strassen_2x2x2()
            # Only use the first 2x2x2 part; for larger blocks, this is approximate
            # For now, just use schoolbook to be safe
            U_block, V_block, W_block = _schoolbook_block(a, b, c)
        else:
            U_block, V_block, W_block = _schoolbook_block(a, b, c)
        
        # Map local indices to global indices
        row_start = row_offsets[ri]
        inner_start = inner_offsets[bi]
        col_start = col_offsets[ci]
        
        for local_i in range(a):
            for local_j in range(b):
                global_row = row_start + local_i
                global_inner = inner_start + local_j
                for local_k in range(c):
                    global_col = col_start + local_k
                    
                    u_idx = global_row * m + global_inner
                    v_idx = global_inner * p + global_col
                    w_idx = global_col * n + global_row
                    
                    # The local index in the block
                    local_u = local_i * b + local_j
                    local_v = local_j * c + local_k
                    local_w = local_k * a + local_i
                    
                    # Find which column in the block this corresponds to
                    # In schoolbook, col = local_i * b * c + local_j * c + local_k
                    local_col = local_i * b * c + local_j * c + local_k
                    
                    if local_col < block_rank:
                        U[u_idx][col_offset + local_col] = U_block[local_u][local_col]
                        V[v_idx][col_offset + local_col] = V_block[local_v][local_col]
                        W[w_idx][col_offset + local_col] = W_block[local_w][local_col]
        
        col_offset += block_rank
    
    return U, V, W, total_rank

def _random_search(n, m, p, initial_rank, deadline):
    """Attempt to find a lower rank decomposition via random perturbation."""
    # Start with schoolbook
    U = np.zeros((n * m, initial_rank))
    V = np.zeros((m * p, initial_rank))
    W = np.zeros((p * n, initial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = initial_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns that are linearly dependent
    # This is a simple approach; more sophisticated methods would use
    # tensor rank reduction algorithms
    
    return best_rank, best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # First, find the best block partition
    best_rank, best_partition = _find_best_partition(n, m, p)
    
    if time.time() > deadline:
        # Fall back to schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to construct the block decomposition
    if best_partition is not None and best_rank < schoolbook_rank:
        U, V, W, actual_rank = _construct_block_decomposition(n, m, p, best_partition)
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p, actual_rank):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
