import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=3, p=7.
    Rank: n*m*p = 63.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (Trace convention: C[i,k] = sum_j A[i,j] B[j,k])
                # The tensor term is A[i,j] * B[j,k] * C[i,k]
                # So W corresponds to C[i,k]
                # The prompt says W[k*n+i, r].
                # Let's check the convention: W[k*n+i, r]
                # If W is indexed by k*n+i, then row index is k*n+i.
                # This corresponds to C[k, i]? Or C[i, k]?
                # Standard trace: tr(A B C^T) = sum A[i,j] B[j,k] C[k,i].
                # Here C[k,i] is the element at row k, col i.
                # The prompt says "output factor follows the official trace convention W[k*n+i,r]".
                # This usually means the linear index for C is k*n + i, where k is row, i is col?
                # Or k is col, i is row?
                # In the seed program: W[k * n + i][column] = 1.
                # Loop: for i in range(n), for j in range(m), for k in range(p).
                # U[i*m+j], V[j*p+k], W[k*n+i].
                # This matches the seed. I will keep this.
                W[k * n + i][column] = 1
                column += 1
                
                # Check deadline
                if time.time() > deadline:
                    # This shouldn't happen for such a small problem, but good practice
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    if time.time() > deadline:
        # Incomplete decomposition is invalid. 
        # But since it's a simple loop, it won't break early.
        pass

    return {"U": U, "V": V, "W": W}
