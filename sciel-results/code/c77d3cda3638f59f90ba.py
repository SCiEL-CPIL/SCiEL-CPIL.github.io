import numpy as np

def construct():
    """
    Returns the schoolbook tensor decomposition for 2x4 by 4x8 matrix multiplication.
    The rank is n*m*p = 2*4*8 = 64.
    
    U: shape (n*m, rank) = (8, 64)
    V: shape (m*p, rank) = (32, 64)
    W: shape (p*n, rank) = (16, 64)
    
    The decomposition is:
    C = A B
    C[i, k] = sum_{j} A[i, j] B[j, k]
    
    In tensor form:
    T_{i, j, k} = sum_{r} U_{i, j, r} V_{j, k, r} W_{k, i, r}
    
    Wait, the interface specifies:
    U: (n*m, rank) -> U[idx, r] where idx = i*m + j
    V: (m*p, rank) -> V[idx, r] where idx = j*p + k
    W: (p*n, rank) -> W[idx, r] where idx = k*n + i
    
    The reconstruction formula used in verification is typically:
    C[i, k] = sum_{r} (sum_{j} A[i, j] U[i*m+j, r]) * (sum_{j} V[j*p+k, r] B[j, k]) * W[k*n+i, r]
    
    For the schoolbook method, each term r corresponds to a specific (i, j, k) triple.
    Let r index the triple (i, j, k).
    U[i*m+j, r] = 1 if r corresponds to (i, j, k'), 0 otherwise.
    V[j'*p+k', r] = 1 if r corresponds to (i', j, k), 0 otherwise.
    W[k'*n+i', r] = 1 if r corresponds to (i, j', k), 0 otherwise.
    
    Actually, let's map r to (i, j, k).
    There are n*m*p = 64 such triples.
    For a given r corresponding to (i0, j0, k0):
    U[i*m+j, r] should be 1 if i==i0 and j==j0, else 0.
    V[j*p+k, r] should be 1 if j==j0 and k==k0, else 0.
    W[k*n+i, r] should be 1 if k==k0 and i==i0, else 0.
    
    Then the term for this r in the sum is:
    (sum_j A[i, j] U[i*m+j, r]) * (sum_j V[j*p+k, r] B[j, k]) * W[k*n+i, r]
    
    Let's evaluate the factors:
    1. sum_j A[i, j] U[i*m+j, r]:
       U[i*m+j, r] is 1 only if j=j0 (and i=i0).
       So this sum is A[i, j0] if i==i0, else 0.
       
    2. sum_j V[j*p+k, r] B[j, k]:
       V[j*p+k, r] is 1 only if j=j0 and k=k0.
       So this sum is B[j0, k] if k==k0, else 0.
       
    3. W[k*n+i, r]:
       W[k*n+i, r] is 1 if k==k0 and i==i0, else 0.
       
    So the product is:
    (A[i, j0] if i==i0 else 0) * (B[j0, k] if k==k0 else 0) * (1 if i==i0 and k==k0 else 0)
    
    This product is non-zero ONLY if i==i0, k==k0.
    In that case, the value is A[i0, j0] * B[j0, k0] * 1.
    
    The total sum over r is:
    sum_{i0, j0, k0} [ (A[i, j0] if i==i0 else 0) * (B[j0, k] if k==k0 else 0) * (1 if i==i0 and k==k0 else 0) ]
    
    For a fixed output element C[i, k]:
    The term is non-zero only when i0 = i and k0 = k.
    So the sum becomes:
    sum_{j0} A[i, j0] * B[j0, k]
    
    This is exactly (A B)[i, k].
    
    So the construction is correct.
    """
    n, m, p = 2, 4, 8
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
