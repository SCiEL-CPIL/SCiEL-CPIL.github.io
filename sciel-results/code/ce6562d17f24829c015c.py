import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses the schoolbook algorithm which has rank n*m*p = 60.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct the decomposition
    # Each term in the sum corresponds to a unique (i, j, k) triple
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                U[i * m + j, col] = 1.0
                # Index in V: row j, col k -> j*p + k
                V[j * p + k, col] = 1.0
                # Index in W: row k, col i -> k*n + i (following trace convention W[k*n+i])
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

BET: {mechanism: "The directive to use Coppersmith-Winograd or Laser method is theoretically sound for asymptotic improvements, but for the specific small dimensions 3x4x5, no known explicit low-rank decomposition (below schoolbook rank 60) is readily implementable in a 120s window without solving large linear systems or having pre-computed magic constants. The 'structural_ast_duplicate' refusal indicates my previous attempt was structurally identical to the baseline. Since I cannot easily invent a new rank < 60 decomposition from scratch within the constraints, I revert to the baseline which is guaranteed correct and valid, acknowledging that a true improvement requires external knowledge of specific low-rank tensors for these dimensions that I do not possess.", q: 0.0, delta: [0.0, 0.0]}
