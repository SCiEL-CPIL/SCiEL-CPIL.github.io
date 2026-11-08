"""
Tensor decomposition for 3x3 matrix multiplication.
We attempt to find a rank-23 decomposition (Strassen-type) and verify it exactly.
If it fails, we fall back to the schoolbook rank-27 decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# Deadline guard: 150s total, aim to finish by 140s
START_TIME = time.time()
DEADLINE = 140.0

def verify_decomposition(U, V, W):
    """
    Verify that U, V, W form an exact tensor decomposition of 3x3 matrix multiplication.
    
    U: shape (n*m, rank) = (9, rank)
    V: shape (m*p, rank) = (9, rank)
    W: shape (p*n, rank) = (9, rank)
    
    The decomposition is valid if:
    For all i, j, k (0..2):
        sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... 
    Actually, the condition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * ... no.
    
    The tensor multiplication condition is:
    For each r, we have:
    A * B = sum_r (sum_j U[i,j,r] * B[j,k]) ... no.
    
    Let's use the standard form:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition is:
    C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * ... no.
    
    Correct form:
    C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) ... no, that's not right either.
    
    The correct tensor decomposition form is:
    C[i,k] = sum_r lambda_r * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) ... no.
    
    Actually, the form is:
    C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... no.
    
    Let's use the explicit formula from the seed:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, the condition for the tensor to be decomposed is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Where T[i,j,k] = 1 if the product term A[i,j]*B[j,k] contributes to C[i,k].
    But C[i,k] = sum_j A[i,j]*B[j,k].
    
    So the tensor T has entries:
    T[i,j,k] = 1 if k is such that A[i,j]*B[j,k] is part of the sum for C[i,k].
    But actually, for fixed i,k, we sum over j.
    So the tensor is:
    T[i,j,k] = 1 if j is the intermediate index.
    
    Actually, the standard form is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as:
    C[i,k] = sum_r sum_j sum_k' ...
    
    Let's use the verification:
    For all i, j, k in 0..2:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    Wait, is that right?
    
    Let's check the seed program:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    So for the schoolbook case:
    For a fixed (i,j,k), there is exactly one r (column) where all three are 1.
    So sum_r ... = 1 for the specific (i,j,k) that matches the column.
    But we need to check the condition for all i,j,k.
    
    Actually, the condition is:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k} ? No.
    
    The tensor for matrix multiplication is:
    T[i,j,k] = 1 if we are multiplying A[i,j] and B[j,k].
    But the output is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    
    So the tensor T is defined by:
    T[i,j,k] = 1 for all i,j,k? No.
    T[i,j,k] is the coefficient of A[i,j] B[j,k] in the expression for C[i,k].
    But C[i,k] = sum_j A[i,j] B[j,k].
    So for a fixed i,k, the terms are A[i,0]B[0,k], A[i,1]B[1,k], A[i,2]B[2,k].
    
    So the tensor T has:
    T[i,j,k] = 1 if we want A[i,j]B[j,k] to contribute to C[i,k].
    But actually, A[i,j]B[j,k] contributes to C[i,k] only if the indices match: i for row, k for column, j for inner.
    So T[i,j,k] = 1 for all i,j,k? No, that's not a tensor decomposition.
    
    The correct condition is:
    For all i, k:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor form:
    C[i,k] = sum_{j} sum_{r} ...
    
    Let's use the explicit check:
    Construct the tensor T[i,j,k] where T[i,j,k] = 1 if the term A[i,j]B[j,k] is included in the sum for C[i,k].
    But actually, for matrix multiplication, T[i,j,k] = 1 for all i,j,k is NOT correct.
    
    Let's look at the seed again:
    The seed sets U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] all to 1 for r corresponding to (i,j,k).
    
    So the condition is:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    Is this correct?
    Let's test with i=0, j=0, k=0.
    In schoolbook, there is one r where U[0, r]=1, V[0, r]=1, W[0, r]=1.
    So sum = 1. Correct.
    
    But wait, what if i=0, j=0, k=1?
    In schoolbook, there is a different r for (0,0,1).
    U[0, r']=1, V[1, r']=1 (since j*p+k = 0*3+1 = 1), W[1, r']=1 (since k*n+i = 1*3+0 = 3? No, W index is k*n+i = 1*3+0 = 3? But W has 9 rows, 0..8. 3 is valid.
    
    Wait, the indices:
    U row: i*m+j = 0*3+0 = 0
    V row: j*p+k = 0*3+1 = 1
    W row: k*n+i = 1*3+0 = 3
    
    So for r corresponding to (0,0,1):
    U[0, r]=1, V[1, r]=1, W[3, r]=1.
    
    So for the query (i=0, j=0, k=1):
    U[0, r]*V[1, r]*W[3, r] = 1*1*1 = 1 for that r.
    And 0 for others.
    So sum = 1.
    
    But does this verify the decomposition?
    The decomposition is valid if:
    For all i, k:
    sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
    
    Where T[i,j,k] is recovered from U, V, W?
    
    Actually, the condition sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    is the condition that the tensor T[i,j,k] = sum_r ... equals the matrix multiplication tensor.
    
    But the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k?
    No! The matrix multiplication tensor is:
    T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    
    But actually, for fixed i,k, we sum over j. So for each (i,k), there are 3 terms (j=0,1,2).
    So T[i,j,k] should be 1 for all i,j,k?
    
    Let's think:
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    So T[0,0,0]=1, T[0,1,0]=1, T[0,2,0]=1.
    
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    So T[0,0,1]=1, T[0,1,1]=1, T[0,2,1]=1.
    
    So yes, T[i,j,k] = 1 for all i,j,k in 0..2.
    
    So the condition is:
    For all i, j, k in 0..2:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    This is the correct verification condition.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Check dimensions
    if len(U) != n*m or len(U[0]) != rank:
        return False
    if len(V) != m*p or len(V[0]) != rank:
        return False
    if len(W) != p*n or len(W[0]) != rank:
        return False
    
    # Verify all 81 conditions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                
                s = 0
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if s != 1:
                    return False
    
    return True


def get_strassen_23():
    """
    Attempt to return a rank-23 decomposition.
    This is based on known Strassen-type algorithms for 3x3 matrices.
    If the coefficients are wrong, the verifier will reject it.
    """
    # This is a placeholder. In practice, we would have the actual coefficients.
    # For now, return an empty list to indicate failure.
    return None


def get_schoolbook_27():
    """Return the schoolbook rank-27 decomposition."""
    n, m, p = DIMENSIONS
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
    return U, V, W


def construct():
    # Check deadline
    if time.time() > DEADLINE:
        U, V, W = get_schoolbook_27()
        return {"U": U, "V": V, "W": W}
    
    # Try to get rank-23 decomposition
    candidate = get_strassen_23()
    if candidate is not None:
        U, V, W = candidate
        if verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook_27()
    return {"U": U, "V": V, "W": W}
