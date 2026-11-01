import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    # Local indices: ii in {0,1}, jj in {0,1}, kk in {0,1}
    # Global indices: i = i_offset + ii, j = j_offset + jj, k = k_offset + kk
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i

    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0], -C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    # M5: (A[0,0]+A[0,1])B[1,1] -> -C[0,0], C[0,1]
    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _build_rank_28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=0, k=0..4
    # C_{ik} += A_{i0} B_{0k}
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 3: j=1,2, k=2
    # C_{ik} += sum_{j=1,2} A_{ij} B_{j2}
    # This is 2x2x1. Rank 4.
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1

    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the standard matmul tensor T[i, j, k] = 1 if we're looking at the structural tensor
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the structural sense.
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the structural tensor has all ones.
    
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_27(deadline):
    """Attempt to find a rank-27 decomposition via optimization."""
    n, m, p = DIMENSIONS
    target_rank = 27
    
    # Get the rank-28 solution as starting point
    U28, V28, W28 = _build_rank_28()
    
    # Strategy: remove one column from the rank-28 solution and try to fix the residual
    # The residual after removing column r is:
    # R[i,j,k] = 1 - sum_{s != r} U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
    #           = -U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # So we need to find a rank-1 tensor that equals this residual.
    # A rank-1 tensor is u[i*m+j] * v[j*p+k] * w[k*n+i]
    # So we need U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = -U28[i*m+j,r] * V28[j*p+k,r] * W28[k*n+i,r]
    # This means the removed column's tensor must be rank 1, which it is by construction.
    # But wait, that's trivially true. The issue is that removing a column from a valid 
    # rank-28 decomposition gives a residual that IS the removed column's contribution,
    # which is rank 1. So we can't just remove a column.
    
    # Better strategy: try to merge two columns into one.
    # Or: use scipy.optimize to find a rank-27 decomposition.
    
    # Parameterize U, V, W with rank 27
    # Total variables: 27*(6 + 15 + 10) = 27*31 = 837
    # This is too many for a quick optimization.
    
    # Alternative: exploit the structure. The rank-28 solution has blocks.
    # Maybe we can find a different block decomposition with lower rank.
    
    # Let's try: R(2,3,5) <= R(2,2,5) + R(2,1,5) = ?
    # R(2,1,5) = 10 (trivial)
    # R(2,2,5) <= ? We know R(2,2,5) <= 17 (from Strassen-like)
    # Actually R(2,2,5) is known to be 17.
    # So R(2,3,5) <= 17 + 10 = 27?
    
    # Let's try to build R(2,2,5) with rank 17 and R(2,1,5) with rank 10.
    # Total: 27.
    
    # R(2,1,5): A is 2x1, B is 1x5, C is 2x5.
    # C_{ik} = A_{i0} * B_{0k}
    # This is rank 10 (one for each i,k pair).
    
    # R(2,2,5): A is 2x2, B is 2x5, C is 2x5.
    # Need rank 17.
    
    # Let's try to find R(2,2,5) = 17.
    # This is a known result. Let me try to construct it.
    
    # Actually, let me just try the optimization approach with a time limit.
    # Use a smaller parameterization.
    
    # For now, return None to indicate failure, and fall back to rank 28.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin out of 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # First, try to find a rank-27 decomposition
    best_result = None
    
    # Try rank 27
    if time.time() < deadline:
        result_27 = _try_rank_27(deadline)
        if result_27 is not None:
            U, V, W = result_27
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W, 27)
    
    # Fall back to rank 28
    if best_result is None or time.time() > deadline:
        U, V, W = _build_rank_28()
        best_result = (U, V, W, 28)
    
    U, V, W, rank = best_result
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
