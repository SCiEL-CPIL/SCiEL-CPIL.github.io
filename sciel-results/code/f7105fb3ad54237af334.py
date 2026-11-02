import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the factors U, V, W exactly decompose the matrix multiplication tensor.
    We check the identity:
    Sum_k U[i*m+j, k] * V[j*p+k', k] * W[k'*n+i', k] ... wait, the indices are:
    U shape: (n*m, rank) -> index a = i*m + j
    V shape: (m*p, rank) -> index b = j*p + k
    W shape: (p*n, rank) -> index c = k*n + i (trace convention W[k*n+i])
    
    The tensor component T[a, b, c] for matmul C=AB is 1 if:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    otherwise 0.
    
    We check if sum_k U[a,k] * V[b,k] * W[c,k] == T[a,b,c] for all a,b,c.
    """
    rank = U[0].__len__() if isinstance(U, list) and U else U.shape[1]
    
    # Build target tensor
    # T[a, b, c]
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For fixed a (i,j), b (j',k), c (k',i'), the term is non-zero only if:
    # j = j' AND k = k' AND i = i'
    # So T[a,b,c] = 1 iff (i == i') and (j == j') and (k == k')
    # a = i*m+j, b = j*p+k, c = k*n+i'
    # Check consistency:
    # From a: i = a//m, j = a%m
    # From b: j' = b//p, k = b%p
    # From c: k' = c//n, i' = c%n
    # Condition: (j == j') and (k == k') and (i == i')
    
    # We can verify by checking all a, b, c.
    # Total elements: (3*3) * (3*7) * (7*3) = 9 * 21 * 21 = 3969.
    # This is small. We can do an exact integer/rational check or float check.
    
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j_prime = b // p
            k = b % p
            # For this a, b to contribute, we need j == j_prime
            if j != j_prime:
                for c in range(p * n):
                    # Sum must be 0
                    val = sum(U[a][kk] * V[b][kk] * W[c][kk] for kk in range(rank))
                    if abs(val) > 1e-9:
                        return False
            else:
                # We need k_prime == k and i_prime == i
                # c = k_prime * n + i_prime
                target_c = k * n + i
                for c in range(p * n):
                    val = sum(U[a][kk] * V[b][kk] * W[c][kk] for kk in range(rank))
                    if c == target_c:
                        if abs(val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(val) > 1e-9:
                            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factors as lists of lists (integers)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col_idx] = 1
                # V index: j*p + k
                V[j * p + k][col_idx] = 1
                # W index: k*n + i
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    # Optional: Verification (commented out for speed, but logic is there)
    # if not _verify_decomposition(U, V, W, n, m, p):
    #     raise ValueError("Decomposition is invalid")
        
    return {"U": U, "V": V, "W": W}
