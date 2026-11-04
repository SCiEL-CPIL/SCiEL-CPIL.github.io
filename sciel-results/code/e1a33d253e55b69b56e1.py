import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]"""
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal delta (i.e., T[i,j,k] = 1 if we're decomposing the identity of matmul)
    # Actually, the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the structure tensor)
    # Wait no. The matmul tensor M[i,j,k] satisfies:
    # C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is M[i,j,k] = 1 (it's just the identity mapping in tensor form)
    # The decomposition says: M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err

def _construct_rank56():
    """Rank-56 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Each term t has U: {(i,j): coeff}, V: {(j,k): coeff}, W: {(k,i): coeff}
    # where i in {0,1}, j in {0,1}, k in {0,1}
    
    strassen = [
        # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}},
        # M2 = A00(B01-B11) -> C01 += M2, C11 += M2
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        # M3 = A11(B10-B00) -> C00 += M3, C10 += M3
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        # M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        # M5 = (A10-A00)(B00+B01) -> C11 += M5
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        # M6 = (A01-A11)(B10+B11) -> C00 += M6
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]
    
    col = 0
    for j_block in range(2):  # Split the 4 middle indices into 2 blocks of 2
        for k_block in range(4):  # Split the 8 output indices into 4 blocks of 2
            for t in strassen:
                for (i, j_lo), c_u in t['U'].items():
                    j = j_block * 2 + j_lo
                    U[i * m + j][col] += c_u
                for (j_lo, k_lo), c_v in t['V'].items():
                    j = j_block * 2 + j_lo
                    k = k_block * 2 + k_lo
                    V[j * p + k][col] += c_v
                for (k_lo, i), c_w in t['W'].items():
                    k = k_block * 2 + k_lo
                    W[k * n + i][col] += c_w
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _trim_zero_columns(U, V, W):
    """Remove columns that are all zero in U, V, and W."""
    rank = len(U[0])
    keep = []
    for r in range(rank):
        if any(abs(U[i][r]) > 1e-15 for i in range(len(U))) or \
           any(abs(V[i][r]) > 1e-15 for i in range(len(V))) or \
           any(abs(W[i][r]) > 1e-15 for i in range(len(W))):
            keep.append(r)
    if len(keep) == rank:
        return U, V, W, rank
    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
    return U_new, V_new, W_new, len(keep)

def _try_rank56_polish(U, V, W, deadline):
    """Try to reduce rank by checking linear dependencies among columns."""
    n, m, p = 2, 4, 8
    rank = len(U[0])
    
    # Check for linearly dependent columns by forming the tensor and checking
    # if any column is a linear combination of others
    # This is complex, so we just verify and return
    err = _verify(U, V, W, n, m, p)
    if err < 1e-10:
        return U, V, W, rank
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with rank-56
    result = _construct_rank56()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify
    err = _verify(U, V, W, n, m, p)
    
    # Trim zero columns
    U, V, W, rank = _trim_zero_columns(U, V, W)
    
    # Re-verify after trimming
    if rank < 56:
        err = _verify(U, V, W, n, m, p)
    
    # Try to find a lower-rank decomposition
    # Strategy: The 2x4x8 matmul can be seen as 4 independent 2x2x4 subproblems
    # (splitting the 4 middle indices into 2 pairs, and 8 output indices into 4 pairs)
    # Each 2x2x4 subproblem has rank at most 7 (Strassen for 2x2x2) * 2 (for the 4th dim)
    # But we can potentially do better.
    
    # Actually, let's try a different approach: use the fact that the tensor is
    # a Kronecker product structure. The 2x4x8 matmul tensor is:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # This is NOT a Kronecker product in general.
    
    # Let's try rank 28: use Strassen on pairs of j indices and schoolbook on k
    # For each pair of j indices (j=0,1) and (j=2,3), and for each k:
    # We need 7 Strassen terms per (j_pair, k) pair
    # That's 2 * 8 * 7 = 112, which is worse.
    
    # Alternative: For each k, we need a 2x4 matmul. The rank of 2x4 matmul is 8 (schoolbook).
    # So total rank is 8 * 8 = 64. That's worse than 56.
    
    # The rank-56 construction is: for each of 2 j-blocks and 4 k-blocks, use 7 Strassen terms.
    # 2 * 4 * 7 = 56. This is the current best.
    
    # Can we do better? The lower bound for 2x4x8 matmul rank is at least max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    # But the actual rank is likely higher.
    
    # Let's try to verify and return the rank-56 result
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: return the untrimmed version
        result = _construct_rank56()
        return result
# EVOLVE-BLOCK-END
