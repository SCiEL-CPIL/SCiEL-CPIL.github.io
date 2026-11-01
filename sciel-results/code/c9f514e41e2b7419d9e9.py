import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # Convert to numpy for speed
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over rank
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # Check if s is close to 1
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    
    The tensor T has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3,4}.
    We seek rank r such that T = sum_{c=0}^{r-1} u_c v_c w_c
    where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    
    The trivial schoolbook decomposition has rank n*m*p = 2*2*5 = 20.
    We can do better by exploiting the structure.
    
    Strategy:
    1. Use Strassen-like decomposition for the 2x2x2 sub-tensors (k=0,1) to reduce rank.
    2. Handle remaining k=2,3,4 with minimal additional rank.
    
    For a 2x2x2 tensor with all entries 1, the rank is 7 (Strassen).
    So for k=0 and k=1, if we could decompose them independently, we'd need 7+7=14.
    But the tensor is coupled across k.
    
    Actually, the full tensor T[i,j,k] = 1 for all i,j,k can be written as:
    T = sum_{k=0}^{4} E_k where E_k is the 2x2x2 tensor with all entries 1 at fixed k.
    
    But E_k is not independent; we can share factors.
    
    Let's think differently. The tensor is:
    T[i,j,k] = 1
    
    This is the product of three all-ones vectors:
    T = a * b * c where a[i] = 1, b[j] = 1, c[k] = 1.
    Wait, that's not a rank-1 decomposition of the tensor in the way we need.
    
    Actually, T[i,j,k] = 1 = (1)*(1)*(1), so it IS rank 1!
    But wait, that would mean U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    Let's check: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1*1*1 = 1. Yes!
    
    So rank 1 should work? Let me verify the index conventions.
    
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (10, rank)
    W has shape (p*n, rank) = (10, rank)
    
    U[i*m+j, c] = 1 for all i,j,c
    V[j*p+k, c] = 1 for all j,k,c
    W[k*n+i, c] = 1 for all k,i,c
    
    Then for each (i,j,k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_c 1*1*1 = rank
    
    So if rank=1, the sum is 1. That works!
    
    But wait, is this correct? Let me re-read the task.
    
    "Matrix multiplication 2,2,5"
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 5 matrix."
    
    The matrix multiplication tensor for A (2x2) times B (2x5) gives C (2x5).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k.
    
    Yes! So T is the all-ones tensor of shape 2x2x5.
    
    And as I showed, rank 1 works:
    U[i*m+j, 0] = 1 for all i,j
    V[j*p+k, 0] = 1 for all j,k
    W[k*n+i, 0] = 1 for all k,i
    
    Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1 for each (i,j,k).
    
    So the minimal rank is 1!
    
    Let me verify with the original code's _verify function.
    """
    n, m, p = 2, 2, 5
    
    # Try rank 1
    rank = 1
    U = [[1] for _ in range(n * m)]  # 4 rows
    V = [[1] for _ in range(m * p)]  # 10 rows
    W = [[1] for _ in range(p * n)]  # 10 rows
    
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 20
    rank = 20
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    print(f"Verified: {_verify(result['U'], result['V'], result['W'], 2, 2, 5, rank)}")
