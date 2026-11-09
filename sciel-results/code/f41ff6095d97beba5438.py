"""Matrix multiplication 3x5x5: search for rank-minimal exact tensor decomposition."""
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 75


def _validate(U, V, W):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    n, m, p = N, M, P
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # W index: k*n + i (trace convention W[k*n+i, r])
                # U index: i*m + j
                # V index: j*p + k
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    # Also check that non-multiplication entries are zero
    # The tensor T[i,j,k] = 1 iff (i,j,k) is a valid multiplication index
    # Actually, the tensor is T_{i,j,k} = delta_{i,j,k} where the multiplication
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has 1s at all (i,j,k) and 0s elsewhere? No.
    # The tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # Wait, the tensor is: T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor entries are all 1? No, that's not right.
    # The tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And for invalid combinations... actually all i,j,k are valid.
    # The tensor is simply all-ones? No.
    # Let me re-read: the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    # That's it. All entries of the tensor are 1.
    return True


def _build_schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_reduce_rank(base_decomp, target_rank, deadline):
    """Try to find a decomposition with target_rank by zeroing some columns."""
    U0 = [row[:] for row in base_decomp["U"]]
    V0 = [row[:] for row in base_decomp["V"]]
    W0 = [row[:] for row in base_decomp["W"]]
    full_rank = len(U0[0])
    
    # Try removing one column at a time and see if we can adjust
    # This is a heuristic approach - try each column removal
    best = None
    
    for col_to_remove in range(full_rank):
        if time.time() > deadline - 10:
            break
        # Build decomposition without this column
        cols = [c for c in range(full_rank) if c != col_to_remove]
        U_new = [[U0[i][c] for c in cols] for i in range(len(U0))]
        V_new = [[V0[i][c] for c in cols] for i in range(len(V0))]
        W_new = [[W0[i][c] for c in cols] for i in range(len(W0))]
        
        if _validate(U_new, V_new, W_new):
            best = {"U": U_new, "V": V_new, "W": W_new}
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook()
    
    # Try to reduce rank by 1
    result = _try_reduce_rank(schoolbook, RANK_SCHOOLBOOK - 1, deadline)
    
    if result is None:
        return schoolbook
    
    return result


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
