"""Matrix multiplication 4x4 by 4x7: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition as (U, V, W) numpy int arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition: C[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] for all i,j,k."""
    n, m, p = DIMENSIONS
    # Reconstruct the multiplication tensor
    # T[i,j,k] = 1 means we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1
    # and T[i,j,k] = 0 for non-matching (but in matmul, every (i,j,k) is a valid index)
    # Actually the tensor for matmul C[i,k] = sum_j A[i,j]*B[j,k] has entries T[i,j,k] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :])
                s = s * W[k*n+i, :]
                s = np.sum(s)
                if s != 1:
                    return False
    return True

def _try_random_search(n, m, p, max_iter, deadline):
    """Try random low-rank constructions and verify."""
    best_rank = None
    best_result = None
    for r in range(n*m*p - 1, max(n*m, m*p, p*n), -1):
        if time.time() > deadline:
            break
        if best_rank is not None and r >= best_rank:
            break
        for trial in range(max_iter):
            if time.time() > deadline:
                break
            # Random integer entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
            if _verify(U, V, W):
                best_rank = r
                best_result = (U, V, W)
                break
        if best_rank is not None:
            break
    return best_rank, best_result

def _try_structure_exploitation(n, m, p, deadline):
    """
    Try to exploit the fact that m=4 divides p=7? No.
    Try Strassen-style: if one dimension is a power of 2, we can use Strassen recursively.
    Here m=4=2^2. We can split A into 2x2 blocks (each 2x4) and B into 2x2 blocks (each 4x4 and 4x3... no).
    
    Actually, let's think about bilinear complexity more carefully.
    The known lower bound for 4x4 * 4x7 is at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28.
    Upper bound: schoolbook = 112.
    
    One approach: use the fact that we can compute 4x4 * 4x4 with known minimal rank.
    The 4x4 * 4x4 matrix multiplication has known minimal rank.
    Actually, for 2x2*2x2, rank = 7 (Strassen).
    For 4x4*4x4, we can use 2x2 Strassen twice: 7*7 = 49? No, that's not how it works.
    
    Actually, the bilinear complexity of matrix multiplication M_{n,m,p} satisfies:
    M_{n,m,p} <= M_{n,m,4} * M_{4,1,p} (block decomposition)
    
    For 4x4 * 4x7:
    Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    
    We can compute A*B1 (4x4 * 4x4) and A*B2 (4x4 * 4x3) separately.
    
    M_{4,4,4}: We can use Strassen's algorithm adapted. The minimal rank for 2x2*2x2 is 7.
    For 4x4*4x4, one approach: treat as 2x2 blocks of 2x2 matrices.
    Using Strassen: 7 multiplications of 2x2*2x2, each of rank 7, so total rank = 7*7 = 49.
    But this isn't necessarily optimal.
    
    Actually, a simpler bound: M_{4,4,4} <= M_{4,4,2} * M_{2,1,2}... this is getting complicated.
    
    Let me try a different approach: directly search for a decomposition with rank < 112.
    """
    # Try to find a decomposition with rank around 50-70 using structured random search
    best_rank = None
    best_result = None
    
    for r in [64, 56, 50, 48, 44, 40, 36, 32, 28]:
        if time.time() > deadline:
            break
        # Multiple random trials with different structures
        for trial in range(500):
            if time.time() > deadline:
                break
            # Use entries from {-1, 0, 1, 2, 3}
            U = np.random.randint(-2, 4, size=(n*m, r))
            V = np.random.randint(-2, 4, size=(m*p, r))
            W = np.random.randint(-2, 4, size=(p*n, r))
            
            # Quick check: verify a subset first
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                best_rank = r
                best_result = (U, V, W)
                break
        if best_rank is not None:
            break
    
    return best_rank, best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, leaving 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p  # 112
    
    # Try random search for lower rank
    best_rank, best_result = _try_random_search(n, m, p, max_iter=100, deadline=deadline)
    if best_rank is not None and best_rank < current_rank:
        U, V, W = best_result
        current_rank = best_rank
    
    # Try structure exploitation
    best_rank2, best_result2 = _try_structure_exploitation(n, m, p, deadline)
    if best_rank2 is not None and best_rank2 < current_rank:
        U, V, W = best_result2
        current_rank = best_rank2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
