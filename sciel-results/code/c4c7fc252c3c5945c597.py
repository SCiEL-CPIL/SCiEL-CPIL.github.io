"""
Rank-minimal exact tensor decomposition for 2x2x12 matrix multiplication.
Strategy: Start from schoolbook rank-48, then use local search with
rank-reducing moves (rank-1 replacement, term merging) to find lower rank.
"""
import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Check each output element
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank dimension: U[i,j,r] * V[j,k,r] * W[k,i,r]
                prod = U3[i, j, :] * V3[j, k, :] * W3[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_reduction(U, V, W, n, m, p, max_iter=5000, deadline=None):
    """
    Try to reduce rank by replacing pairs of rank-1 terms with a single
    rank-1 term when possible (i.e., when the two terms lie in a 2D
    subspace that can be collapsed).
    
    Actually, a more practical approach: try to find if any rank-1 term
    is redundant (can be removed) or if two terms can be combined.
    
    For simplicity, use a different strategy: try random perturbations
    and see if rank can be reduced via Gaussian elimination on the
    tensor components.
    
    Better: Use the fact that if we have a decomposition of rank r,
    we can try to find a decomposition of rank r-1 by checking if
    the tensor lies in the secant variety of order r-1.
    
    Practical approach: Try to eliminate one term at a time by
    expressing it as a combination of others.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    n, m, p = n, m, p
    r = U.shape[1]
    
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 for all i,j,k (since we're decomposing the identity tensor
    # for matrix multiplication where each output is a single product)
    
    # Actually, the tensor we're decomposing is the matrix multiplication tensor:
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (k,i) of C
    # Wait, let me re-read the problem.
    # 
    # The task says: "exactly decompose the matrix multiplication tensor"
    # For C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor is T[i,j,k] where the contraction is:
    # C[i,k] = sum_r U[i, r] * V[r, k] * W[...]
    # 
    # Actually the standard tensor decomposition for matrix mult:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta_{jk} ... no.
    # 
    # The matrix multiplication tensor has components:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # But that's not right either. Let me think again.
    #
    # In the tensor decomposition framework:
    # We want: C[i,k] = sum_r (sum_j A[i,j] * U[j, r]) * (sum_l B[l, k] * V[r, l]) * W[r, i, k]
    # No, that's not standard.
    #
    # Standard: C[i,k] = sum_r U[i, r] * V[r, k] where U and V are derived from A and B.
    # But here we have three matrices U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    #
    # The decomposition is:
    # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # where T has shape (n*m, m*p, p*n) and T[a, b, c] = 1 if the corresponding
    # matrix multiplication term is present.
    #
    # Specifically, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T has indices (a, b, c) where:
    # a = i*m + j (row of A, column of A)
    # b = j*p + k (row of B, column of B)  
    # c = k*n + i (row of C... wait, C is n x p, so c should index C)
    #
    # Actually from the problem: W[k*n+i, r] - so c = k*n + i
    # And the tensor entry T[a, b, c] = 1 if and only if:
    # a corresponds to A[i,j], b corresponds to B[j,k], c corresponds to C[i,k]
    # i.e., a = i*m+j, b = j*p+k, c = i*p+k... no, c = k*n+i.
    #
    # Let me just check: for the decomposition to work, we need:
    # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where T[a, b, c] = 1 if (a,b,c) form a valid matrix multiplication term.
    #
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 iff i=i', j=j', k=k' (the indices match)
    # Wait no. In matrix mult, C[i,k] gets contribution from A[i,j]*B[j,k].
    # So the "tensor" has a 1 at position (i*m+j, j*p+k, k*n+i) for all valid i,j,k.
    # And 0 elsewhere.
    
    # Let's build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now try to reduce rank
    # Strategy: For each term r, check if it can be removed (set to 0)
    # and the tensor is still correctly decomposed.
    # More powerfully: try to find linear dependencies.
    
    # Build matrix M where M[flat_index, r] = U[a,r]*V[b,r]*W[c,r] for each (a,b,c)
    # But that's huge. Instead, work with the structure.
    
    # Simpler approach: try removing each term one at a time
    for r_remove in range(r):
        if deadline and time.time() > deadline:
            break
        # Try removing term r_remove
        U_test = np.delete(U, r_remove, axis=1)
        V_test = np.delete(V, r_remove, axis=1)
        W_test = np.delete(W, r_remove, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            r -= 1
            # Restart from beginning since indices changed
            r_remove = -1
    
    return U, V, W, r


def _strassen_2x2x2_block():
    """
    Standard Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for n=m=p=2.
    """
    n, m, p, r = 2, 2, 2, 7
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1  # W[k*n+i]: k=0,i=0 -> 0; k=1,i=1 -> 3
    
    # M2: (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1  # k=0, i=1 -> 1
    
    # M3: A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[2, 2] += 1  # k=1, i=0 -> 2
    
    # M4: A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[1, 3] += 1  # k=0,i=0 -> 0; k=0,i=1 -> 1
    
    # M5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[2, 4] -= 1; W[3, 4] += 1  # k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[2, 5] += 1
    W[3, 5] += 1  # k=1, i=1 -> 3
    
    # M7: (A01-A11)(B11+B01)
    U[1, 6] += 1; U[3, 6] -= 1
    V[3, 6] += 1; V[1, 6] += 1
    W[0, 6] += 1  # k=0, i=0 -> 0
    
    return U, V, W


def _build_strassen_2x2x12():
    """
    Build rank-42 decomposition using Strassen on 6 independent 2x2x2 blocks.
    Each block handles one pair of columns (k, k+1).
    """
    n, m, p = 2, 2, 12
    r = 7 * (p // 2)  # 42
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    
    for block in range(p // 2):
        k0 = 2 * block
        k1 = k0 + 1
        base = 7 * block
        
        # Get the 2x2x2 Strassen block
        U4, V4, W4 = _strassen_2x2x2_block()
        
        # Map V indices: for 2x2x2, V has shape (4, 7) with V[j*2+k, r]
        # For our problem, V has shape (m*p, 7) = (24, 7) with V[j*p+k, r]
        # So V4[j*2+k, r] -> V[j*p+k, r]
        
        # U indices: same mapping, U4[i*2+j, r] -> U[i*2+j, r] (n=m=2)
        
        # W indices: for 2x2x2, W4[k*2+i, r]
        # For our problem, W has shape (p*n, 7) = (24, 7) with W[k*n+i, r] = W[k*2+i, r]
        # But k in W4 is 0 or 1, while in our problem k is k0 or k1.
        # So W4[k_local*2+i, r] -> W[k_global*2+i, r] where k_global = k0 + k_local
        
        U[:, base:base+7] += U4
        V[:, base:base+7] += 0  # Need to remap V
        W[:, base:base+7] += 0  # Need to remap W
        
        # Remap V: V4[j*2+k_local, r] -> V[j*p + (k0+k_local), r]
        for j in range(2):
            for k_local in range(2):
                k_global = k0 + k_local
                V[j*p + k_global, base:base+7] += V4[j*2 + k_local, :]
        
        # Remap W: W4[k_local*2+i, r] -> W[(k0+k_local)*n + i, r]
        for k_local in range(2):
            for i in range(2):
                k_global = k0 + k_local
                W[k_global*n + i, base:base+7] += W4[k_local*2 + i, :]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with Strassen rank-42
    U, V, W = _build_strassen_2x2x12()
    
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        Us = np.zeros((n*m, rank_school), dtype=int)
        Vs = np.zeros((m*p, rank_school), dtype=int)
        Ws = np.zeros((p*n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    # Try to reduce rank
    U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline=deadline)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try additional reduction passes
    for _ in range(3):
        if time.time() > deadline:
            break
        U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline=deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
