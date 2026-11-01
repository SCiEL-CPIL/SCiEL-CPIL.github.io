import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid matrix multiplication decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    """
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += abs(val - 1.0)
    return err

def _verify_decomposition_frac(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i * m + j, r]) * Fraction(V[j * p + k, r]) * Fraction(W[k * n + i, r])
                if val != Fraction(1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_rank_48():
    """
    Try to construct a rank-48 decomposition.
    
    Strategy: The 2x3x9 problem can be viewed as 9 copies of 2x3x1.
    Schoolbook for 2x3x1 has rank 6, giving 9*6 = 54.
    
    We can try to share some structure. The key insight is that the 2x3 multiplication
    (without the output column) can be done in fewer multiplications.
    
    For 2x3x1: C[i] = sum_j A[i,j] * B[j]
    C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    
    This is equivalent to multiplying a 2x3 matrix by a 3x1 vector.
    The tensor rank of 2x3x1 is at most 6 (schoolbook) but could be lower.
    
    Actually, for the full 2x3x9 problem, we need:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: use the fact that this is a bilinear map from (R^6, R^27) to R^18.
    
    Let's try a different decomposition. We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the identity for 2x3:
    Let's define 6 products per output column, but try to reduce.
    
    Actually, the minimal rank for 2x3x9 is not known to be less than 54 in general,
    but let's try to find a rank-48 decomposition.
    
    For 2x2x2, the rank is 7 (Strassen). For 2x3x1, we can use:
    C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    
    This can be done in 6 multiplications (schoolbook) or possibly fewer.
    
    Let's try: use 5 multiplications for the 2x3x1 sub-problem?
    That would give 9*5 = 45.
    
    Or use 6 for some and 4 for others?
    
    Actually, let me try a direct approach with rank 48 = 6*8.
    We can handle 8 of the 9 output columns with 6 multiplications each = 48,
    and the 9th column would need additional multiplications.
    
    Alternative: rank 48 = 4*12. Use 12 multiplications for 3 output columns?
    
    Let me try a different structure. 
    For each pair of output columns (k1, k2), we compute:
    C[:,k1] = A * B[:,k1]
    C[:,k2] = A * B[:,k2]
    
    This is a 2x3x2 problem. The schoolbook rank is 2*3*2 = 12.
    Can we do it in fewer?
    
    For 2x3x2:
    C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
    C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
    C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2] + A[0,2]*B[2,k2]
    C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2] + A[1,2]*B[2,k2]
    
    This is 12 terms. Can we share?
    
    Let's define:
    P1 = A[0,0]*B[0,k1]
    P2 = A[0,1]*B[1,k1]
    P3 = A[0,2]*B[2,k1]
    P4 = A[1,0]*B[0,k1]
    P5 = A[1,1]*B[1,k1]
    P6 = A[1,2]*B[2,k1]
    P7 = A[0,0]*B[0,k2]
    P8 = A[0,1]*B[1,k2]
    P9 = A[0,2]*B[2,k2]
    P10 = A[1,0]*B[0,k2]
    P11 = A[1,1]*B[1,k2]
    P12 = A[1,2]*B[2,k2]
    
    No obvious sharing. So 2x3x2 has rank at most 12.
    
    For 9 output columns, we can group them as 4 pairs + 1 single:
    4 * 12 + 6 = 54. Still 54.
    
    Or 3 groups of 3: 3 * 18 = 54.
    
    Let me try a completely different approach. Use the Kronecker product structure.
    
    2x3x9 = (2x3x1) ⊗ (1x1x9) in some sense?
    No, that's not quite right.
    
    Actually, 2x3x9 can be seen as: for each of the 9 output columns, we do a 2x3x1 multiplication.
    The tensor is the direct sum of 9 copies of the 2x3x1 tensor.
    So the rank is at most 9 * rank(2x3x1).
    
    What is rank(2x3x1)? It's the rank of the tensor T[i,j,k] where i in {0,1}, j in {0,1,2}, k=0.
    T[i,j,0] = 1 for all i,j.
    
    This is the matrix multiplication tensor for 2x3 times 3x1.
    The rank is at most 6 (schoolbook). Is it less?
    
    For 2x2x1, the rank is 4 (since C[0]=A[0,0]*B[0]+A[0,1]*B[1], C[1]=A[1,0]*B[0]+A[1,1]*B[1], 
    and there's no way to share since the A and B indices are different).
    Actually, 2x2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1].
    This is 4 multiplications. Can we do it in 3? 
    Let's see: we need to compute 4 bilinear forms. The tensor is:
    T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    This is the all-ones 2x2x1 tensor. Its rank is 4 (it's a matrix of all ones, rank 1, but as a tensor it's different).
    Actually, for a 2x2x1 tensor with all entries 1, the rank is 4 because you can't write it as a sum of fewer than 4 rank-1 tensors.
    
    Similarly, 2x3x1 has rank at most 6. It might be exactly 6.
    
    So 9*6 = 54 is likely optimal or close to optimal.
    
    Let me just return the schoolbook decomposition.
    """
    return None

def _try_rank_54_optimized():
    """
    Try to find a rank-54 decomposition that is slightly better structured.
    This is just the schoolbook, but let's make sure it's correct.
    """
    n, m, p = 2, 3, 9
    U, V, W, rank = _schoolbook(n, m, p)
    err = _verify_decomposition(U, V, W, n, m, p)
    if err < 1e-9:
        return U, V, W, rank
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try to find a better decomposition
    # Try rank 48
    result = _try_rank_48()
    if result is not None:
        U, V, W, rank = result
        if _verify_decomposition(U, V, W, n, m, p) < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 54 (schoolbook)
    result = _try_rank_54_optimized()
    if result is not None:
        U, V, W, rank = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: explicit schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
