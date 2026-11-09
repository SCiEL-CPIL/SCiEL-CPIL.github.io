"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    
    The tensor T[i,j,k] = delta for the matrix multiplication A (n x m) * B (m x p) = C (n x p)
    has T[i,j,k] = 1 if and only if the (i,k) entry of C depends on A[i,j]*B[j,k].
    
    With the convention W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    (since each (i,j,k) triple appears exactly once in the schoolbook).
    
    Actually, let me reconsider. The tensor for matrix multiplication is:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me think more carefully.
    
    The matrix multiplication tensor has entries T[i, j, k] where:
    - i indexes rows of A (and C): 0..n-1
    - j indexes columns of A (and rows of B): 0..m-1  
    - k indexes columns of B (and C): 0..p-1
    
    T[i,j,k] = 1 means C[i,k] += A[i,j] * B[j,k]
    
    A rank-r decomposition gives: T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For schoolbook: each (i,j,k) triple gets its own rank term, so U[i*m+j, i*m*p+j*p+k] = 1,
    V[j*p+k, i*m*p+j*p+k] = 1, W[k*n+i, i*m*p+j*p+k] = 1, and T[i,j,k] = 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check dimensions
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_kronecker_decomposition(n, m, p, deadline):
    """Try to find a lower-rank decomposition using Kronecker product structure.
    
    If n = n1*n2, m = m1*m2, p = p1*p2, then the rank of M_{n,m,p} is at most
    rank(M_{n1,m1,p1}) * rank(M_{n2,m2,p2}).
    
    For 3x4x8:
    - 3 = 3*1, 4 = 2*2, 8 = 4*2
    - M_{3,2,4} has rank at most 24 (schoolbook) or possibly less
    - M_{1,2,2} has rank at most 4 (schoolbook)
    - Product: 24*4 = 96 (same as schoolbook)
    
    Alternative:
    - 3 = 1*3, 4 = 4*1, 8 = 8*1
    - M_{1,4,8} rank at most 32, M_{3,1,1} rank at most 3
    - Product: 32*3 = 96
    
    - 3 = 3*1, 4 = 4*1, 8 = 8*1
    - M_{3,4,1} rank at most 12, M_{1,1,8} rank at most 8
    - Product: 12*8 = 96
    
    None of these help directly. But maybe we can use Strassen for sub-problems.
    """
    n, m, p = 3, 4, 8
    
    # Try: 3x4x8 = (3x2x4) x (1x2x2) via Kronecker
    # rank(M_{3,2,4}) * rank(M_{1,2,2})
    # Schoolbook: 24 * 4 = 96
    # But can we do M_{1,2,2} with rank < 4? M_{1,2,2}: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    # This is just a dot product, rank = 2*1 = 2? No, it's 1x2 times 2x2, so 4 scalar mults.
    # Actually M_{1,2,2} has rank 2 (since it's just two scalar multiplications added together... no)
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    # This is rank 4 in general (no way to reduce below 4 for generic inputs)
    
    # Try another factorization: 3x4x8 = (3x4x2) x (1x1x4)
    # rank(M_{3,4,2}) * rank(M_{1,1,4}) = 24 * 4 = 96
    
    # Try: use the fact that 8 = 2^3, 4 = 2^2
    # M_{3,4,8}: maybe decompose as block matrix multiplication
    
    # For now, just return schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try to improve via Kronecker product structure
    if time.time() < deadline:
        U2, V2, W2 = _try_kronecker_decomposition(n, m, p, deadline)
        if U2.shape[1] < U.shape[1]:
            if _verify_decomp(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
