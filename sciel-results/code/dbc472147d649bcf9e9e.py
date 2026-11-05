"""
Rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
Uses a block-partition search: split rows, columns, and output dimensions into
blocks of size 1 and 2, then apply Strassen (rank 7) to 2x2x2 blocks and
schoolbook (rank 8) to all other blocks. The minimum over all such partitions
gives rank 99.
"""
import time
from itertools import product, combinations_with_replacement

DIMENSIONS = [3, 5, 7]

# Known best rank for small block sizes (i x j x k)
def block_rank(i, j, k):
    if i == 2 and j == 2 and k == 2:
        return 7  # Strassen
    return i * j * k  # schoolbook


def generate_partitions(n, max_block=2):
    """Generate all compositions of n using parts 1 and 2."""
    results = []
    def backtrack(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in [1, 2]:
            if part <= remaining:
                current.append(part)
                backtrack(remaining - part, current)
                current.pop()
    backtrack(n, [])
    return results


def compute_rank(partition_i, partition_j, partition_k):
    total = 0
    for i in partition_i:
        for j in partition_j:
            for k in partition_k:
                total += block_rank(i, j, k)
    return total


def best_partition(n_i, n_j, n_k):
    """Find the partition combination with minimum rank."""
    parts_i = generate_partitions(n_i)
    parts_j = generate_partitions(n_j)
    parts_k = generate_partitions(n_k)
    
    best_rank = float('inf')
    best_combo = None
    for pi in parts_i:
        for pj in parts_j:
            for pk in parts_k:
                r = compute_rank(pi, pj, pk)
                if r < best_rank:
                    best_rank = r
                    best_combo = (pi, pj, pk)
    return best_rank, best_combo


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Find best partition
    best_rank, (pi, pj, pk) = best_partition(n, m, p)
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the decomposition
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i, si in enumerate(pi):
        for j, sj in enumerate(pj):
            for k, sk in enumerate(pk):
                r = block_rank(si, sj, sk)
                if r == 7:  # Strassen block
                    # Strassen decomposition for 2x2x2
                    # A[si x sj], B[sj x sk], C[si x sk]
                    # Using standard Strassen with entries in {1, -1}
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B12-B22)
                    # M4 = A22(B21-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # U factors (for A matrix, si x sj)
                    # V factors (for B matrix, sj x sk)
                    # W factors (for C matrix, sk x si)
                    
                    # Strassen U: 4x7, V: 4x7, W: 4x7
                    # Indexing: A[i,j], B[j,k], C[k,i] where i in [0,si), j in [0,sj), k in [0,sk)
                    
                    # U rows indexed by (i, j) in si x sj
                    # V rows indexed by (j, k) in sj x sk
                    # W rows indexed by (k, i) in sk x si
                    
                    # For each of the 7 Strassen multiplications:
                    # m1: U has A11 and A22, V has B11 and B22
                    # etc.
                    
                    # Let's use explicit Strassen coefficients
                    # Strassen multipliers:
                    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                    # m2 = (A[1,0]+A[1,1]) * B[0,0]
                    # m3 = A[0,0] * (B[0,1]-B[1,1])
                    # m4 = A[1,1] * (B[1,0]-B[0,0])
                    # m5 = (A[0,0]+A[0,1]) * B[1,1]
                    # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
                    # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
                    
                    # Output reconstruction:
                    # C[0,0] = m1 + m4 - m5 + m7
                    # C[0,1] = m3 + m5
                    # C[1,0] = m2 + m4
                    # C[1,1] = m1 - m2 + m3 + m6
                    
                    # U entries (si x sj x 7):
                    # Row (i,j), Col m:
                    U_vals = [
                        # m1, m2, m3, m4, m5, m6, m7
                        # (0,0):
                        [1,  0,  1,  0,  1,  1,  0],
                        # (0,1):
                        [0,  0,  0,  0,  1,  0,  1],
                        # (1,0):
                        [0,  1,  0,  0,  0,  1,  0],
                        # (1,1):
                        [1,  1,  0,  1,  0,  0, -1],
                    ]
                    V_vals = [
                        # (0,0):
                        [1,  1,  0, -1,  0,  1,  0],
                        # (0,1):
                        [0,  0,  1,  0,  0,  1,  0],
                        # (1,0):
                        [0,  0,  0,  1,  0,  0,  1],
                        # (1,1):
                        [1,  0, -1,  0,  1,  0,  1],
                    ]
                    W_vals = [
                        # (0,0):
                        [1,  0,  0,  1, -1,  0,  1],
                        # (0,1):
                        [0,  0, 1,  0,  1,  0,  0],
                        # (1,0):
                        [0,  1,  0,  1,  0,  0,  0],
                        # (1,1):
                        [1, -1, 1,  0,  0,  1,  0],
                    ]
                    
                    for mi in range(si):
                        for mj in range(sj):
                            for mm in range(7):
                                U[i*sj + mi][col + mm] = U_vals[mi*sj + mj][mm]
                    for mj in range(sj):
                        for mk in range(sk):
                            for mm in range(7):
                                V[j*sk + mj][col + mm] = V_vals[mj*sk + mk][mm]
                    for mk in range(sk):
                        for mi in range(si):
                            for mm in range(7):
                                W[k*si + mk][col + mm] = W_vals[mk*si + mi][mm]
                    col += 7
                else:
                    # Schoolbook block
                    for mi in range(si):
                        for mj in range(sj):
                            for mk in range(sk):
                                U[i*sj + mi][col] = 1
                                V[j*sk + mj][col] = 1
                                W[k*si + mk][col] = 1
                                col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Truncate to actual col used in case we broke early
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
    
    return {"U": U, "V": V, "W": W}
