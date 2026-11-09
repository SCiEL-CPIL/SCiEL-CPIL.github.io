"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication 3x5 * 5x5.
    
    Uses a block decomposition:
    - Split A (3x5) into 3 blocks of size 1x5
    - Split B (5x5) into 5 blocks of size 5x1
    
    For each pair (i, j), we compute a 1x5 times 5x1 product, which is a scalar.
    This is just a dot product, which has rank 1 in the tensor decomposition sense
    (it's a single multiplication).
    
    Wait, that's not right. The tensor M_{n,m,p} has entries T[i,j,k] = delta[i+k, j].
    Actually, the matrix multiplication tensor is defined such that:
    C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}.
    More precisely, the bilinear form is: sum_{i,j,k} C_{i,k} * T[i,j,k] * A_{i,j} * B_{j,k}
    where T[i,j,k] = 1 if the indices match appropriately.
    
    Actually, the standard definition: the matrix multiplication tensor M_{n,m,p} is in R^{n*m} x R^{m*p} x R^{p*n}
    with M_{n,m,p} = sum_{i,j,k} e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)} ⊗ e_{k,i}^{(C)}
    
    So the rank is the minimum number of rank-1 tensors needed to express this.
    
    For the schoolbook method, rank = n*m*p.
    
    Let me think about this differently. We want to decompose:
    C = A * B
    
    where A is 3x5, B is 5x5, C is 3x5.
    
    We can use Strassen's algorithm for 2x2 blocks. Let's partition:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 1x2, A22 is 1x3
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    Hmm, this is getting complicated. Let me just use the schoolbook method for now,
    but try to find a better decomposition.
    
    Actually, for M_{3,5,5}, I don't know of a known decomposition better than schoolbook.
    Let me just return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all indices i, j, k
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
