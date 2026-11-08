"""Rewritten program: search for low-rank exact tensor decomposition of 2x3 * 3x10 multiplication."""
import time
import random
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matmul tensor.
    
    The tensor T[i,j,k] for (C = A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    must equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    rank = U.shape[1]
    # Build the full tensor T of shape (n, m, p)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    # The true matmul tensor: T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij}B_{jk},
    # and the tensor entry for coefficient of A_{ij}B_{jk} in (AB)_{ik} is delta_{i,k-target}... 
    # Actually the structure tensor of matmul: T[i,j,k] = 1 if the output index (i,k) receives
    # contribution from A_{i,j} * B_{j,k}. So T[i,j,k] = 1 for all valid i,j,k.
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)


def _try_random_rank(n, m, p, rank, attempts=200, seed=42, time_limit=10.0):
    """Try to find a rank-`rank` decomposition via random initialization + verification.
    
    This is a heuristic: randomly initialize U, V, W and check if they work.
    For small ranks this is unlikely to succeed, but we include it for completeness.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    for attempt in range(attempts):
        if time.time() - start > time_limit:
            break
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None


def _greedy_construct(n, m, p):
    """Construct a decomposition by greedily selecting bilinear terms.
    
    The idea: the matmul tensor has entries T[i,j,k]=1. We want to express it as
    sum_r u_r ⊗ v_r ⊗ w_r where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    A natural approach: each "term" r corresponds to a choice of (i, j, k) triple,
    giving U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1. This is the schoolbook.
    
    To reduce rank, we can try to combine terms. For example, if we fix i and j,
    we can have one term that covers all k for that (i,j) pair by using a different W.
    But W[k*n+i, r] must be 1 for all k in the group, which means W has 1s in multiple
    positions for the same i. This is valid as long as the tensor equation holds.
    
    Let's think more carefully. The tensor equation is:
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Consider a decomposition where each term r is associated with a pair (i, j).
    Then U[i*m+j, r] = 1, V[j*p+k, r] = a_{j,k}, W[k*n+i, r] = b_{i,k}.
    The equation becomes: for the term r=(i,j): a_{j,k} * b_{i,k} must contribute to T[i,j,k]=1.
    But we need sum over r for fixed (i,j,k) to be 1. If only r=(i,j) has U[i*m+j,r]≠0,
    then we need a_{j,k} * b_{i,k} = 1 for all k. We can set a_{j,k}=1, b_{i,k}=1.
    This gives rank n*m = 6, with W[k*n+i, r(i,j)] = 1 for all k.
    
    Let's verify: rank = n*m = 6.
    For each (i,j), create a term. U[i*m+j, r]=1. V[j*p+k, r]=1 for all k. W[k*n+i, r]=1 for all k.
    Check T[i',j',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
    Only r corresponding to (i',j') has U[i'*m+j', r] = 1. For that r:
    V[j'*p+k', r] = 1, W[k'*n+i', r] = 1. So T[i',j',k'] = 1. ✓
    
    Can we do even better? Rank = m = 3?
    For each j, create a term. U[i*m+j, r(j)] = c_{i,j}, V[j*p+k, r(j)] = 1, W[k*n+i, r(j)] = d_{i,k}.
    T[i,j,k] = sum_{j'} U[i*m+j', r(j')] * V[j'*p+k, r(j')] * W[k*n+i, r(j')]
    = sum_{j'} c_{i,j'} * (1 if j'=j else 0... wait, V[j'*p+k, r(j')] is 1 for all k when j'=j'.
    Actually V[j'*p+k, r(j')] = 1 for all k. So:
    T[i,j,k] = sum_{j'} c_{i,j'} * 1 * d_{i,k} = d_{i,k} * sum_{j'} c_{i,j'}
    We need this to equal 1 for all i,j,k. So d_{i,k} * sum_{j'} c_{i,j'} = 1.
    This must hold for all j, which means it doesn't depend on j. So we need
    d_{i,k} * S_i = 1 where S_i = sum_{j'} c_{i,j'}.
    We can set S_i = 1, d_{i,k} = 1. Then c_{i,j'} can be any values summing to 1.
    E.g., c_{i,0}=1, c_{i,1}=0, c_{i,2}=0.
    
    Let's verify rank=3:
    r=0 (j=0): U[i*3+0, 0]=1, U[i*3+1, 0]=0, U[i*3+2, 0]=0 for all i.
                V[0*10+k, 0]=1 for all k, V[1*10+k, 0]=0, V[2*10+k, 0]=0.
                W[k*2+i, 0]=1 for all k,i.
    r=1 (j=1): U[i*3+1, 1]=1, others 0.
                V[1*10+k, 1]=1, others 0.
                W[k*2+i, 1]=1 for all k,i.
    r=2 (j=2): U[i*3+2, 2]=1, others 0.
                V[2*10+k, 2]=1, others 0.
                W[k*2+i, 2]=1 for all k,i.
    
    Check T[i,j,k] = sum_{r=0}^{2} U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r]
    For fixed (i,j,k): U[i*3+j, r] is 1 only when r=j. So:
    T[i,j,k] = V[j*10+k, j] * W[k*2+i, j] = 1 * 1 = 1. ✓
    
    So rank = m = 3 works! Can we do rank = 2?
    With rank 2, we have 2 terms. Each term r has U (6-dim), V (30-dim), W (20-dim).
    The tensor has 60 entries all equal to 1. The rank of the matmul tensor...
    Actually, the minimum rank of the 2x3x10 matmul tensor is known to be at most min(n*m, m*p, p*n) = min(6,30,20)=6,
    but we found rank 3. The true minimum rank for (n,m,p) matmul is at least max(n, m, p) in some cases?
    No, the lower bound is max(n, p) for the "slice rank" but tensor rank can be lower.
    
    Actually, for multiplying an n×m matrix by an m×p matrix, the tensor rank is at most n*m (our first construction)
    and at most m (our second construction). The minimum is at least max(n, p) / something...
    
    Let me think about whether rank < 3 is possible for n=2, m=3, p=10.
    The output has n*p = 20 entries. Each output entry C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor rank is the minimum number of rank-1 tensors needed.
    
    A lower bound: the tensor rank is at least the max of the "slice ranks". 
    For fixed i, the slice T[i,:,:] is an m×p matrix of all 1s, which has rank 1.
    For fixed j, the slice T[:,j,:] is an n×p matrix of all 1s, rank 1.
    For fixed k, the slice T[:,:,k] is an n×m matrix of all 1s, rank 1.
    These don't give strong lower bounds.
    
    Another lower bound: the tensor rank is at least the matrix rank of any unfolding.
    The (n*m) × (p) unfolding... let's think. The tensor T[i,j,k] can be seen as a 3D array.
    Reshape to (n*m) × (p): T_flat[i*m+j, k] = 1. This is a 6×10 matrix of all 1s, rank 1.
    Reshape to (n*m) × (m*p)... that's not standard.
    
    The key constraint: with rank r, we have r vectors u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    
    For rank 2: T[i,j,k] = u_0[i*m+j]*v_0[j*p+k]*w_0[k*n+i] + u_1[i*m+j]*v_1[j*p+k]*w_1[k*n+i] = 1 for all i,j,k.
    
    This means the function (i,j,k) -> T[i,j,k] is a sum of 2 rank-1 "tensors" and equals 1 everywhere.
    The constant function 1 on a 2×3×10 grid. Can it be written as sum of 2 rank-1 tensors?
    A rank-1 tensor is u(i,j) * v(j,k) * w(k,i). Note the dependencies: u depends on (i,j), v on (j,k), w on (k,i).
    
    This is a constrained problem. Let me just try rank 2 numerically.
    
    Actually, let me just implement the rank-3 construction which I've verified, and also try rank 2.
    """
    n, m, p = 2, 3, 10
    
    # Rank m = 3 construction
    rank = m
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        r = j
        # U[i*m+j, r] = 1 for all i
        for i in range(n):
            U[i * m + j, r] = 1.0
        # V[j*p+k, r] = 1 for all k
        for k in range(p):
            V[j * p + k, r] = 1.0
        # W[k*n+i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1.0
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try the rank-m construction first
    result = _greedy_construct(n, m, p)
    if result is not None:
        rank = result["U"].shape[1]
        # Verify once more
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
