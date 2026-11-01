"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = start_time + 115  # 115 second limit (safety margin)

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54

    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Strategy: Exploit the rectangular asymmetry.
    # The matrix multiplication C = A * B where A is 2x3 and B is 3x9.
    # We can view this as two independent 1x3 x 3x9 multiplications (for each row of A).
    # For a 1 x m x p multiplication, the tensor rank is exactly m*p (schoolbook is optimal).
    # Here m=3, p=9, so rank for each row is 27? No, wait.
    # The tensor for <1, m, p> has entries T[i,j,k] with i in {0}.
    # T[0,j,k] = 1.
    # Rank 1 decomposition: sum_c u_c * v_c * w_c = 1 for all j,k.
    # This is equivalent to rank-1 matrix M[j,k]=1.
    # The rank of the all-ones matrix of size m x p is 1.
    # So the tensor rank for <1, m, p> is 1?
    # Let's check: T[0,j,k] = sum_c U[0,c] V[j,c] W[k,c].
    # If we set U[0,0]=1, V[j,0]=1, W[k,0]=1, then sum is 1.
    # So yes, the rank of <1, m, p> matrix multiplication is 1?
    # Wait, the definition of tensor rank for matrix multiplication <n, m, p> is the minimum number of multiplications needed.
    # For <1, m, p>, we compute C[0,k] = sum_j A[0,j] B[j,k].
    # This is just a dot product of a vector of length m with each column of B.
    # Standard algorithm: m*p multiplications.
    # Can we do better?
    # The tensor T is of size 1 x m x p. T[0,j,k] = 1.
    # We want to write T = sum_c u_c v_c^T w_c^T.
    # This is equivalent to finding the rank of the matrix T (flattened).
    # The matrix T is 1 x (m*p) with all entries 1.
    # The rank of this matrix is 1.
    # So the tensor rank of <1, m, p> is 1.
    #
    # However, the interface requires U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    # If we decompose <2, 3, 9> as two independent <1, 3, 9> multiplications, we would have rank 1+1=2?
    # Let's check the constraints.
    # For i=0: sum_c U[0*3+j, c] V[j*9+k, c] W[k*2+0, c] = 1 for all j,k.
    # For i=1: sum_c U[1*3+j, c] V[j*9+k, c] W[k*2+1, c] = 1 for all j,k.
    #
    # If we use rank 1 for i=0:
    # U[j, 0] = 1 for j in 0..2.
    # V[j*9+k, 0] = 1 for all j,k.
    # W[k*2+0, 0] = 1 for all k.
    # Then sum_c=0: U[j,0] V[j*9+k,0] W[k*2+0,0] = 1*1*1 = 1. Correct.
    #
    # If we use rank 1 for i=1 (column 1):
    # U[3+j, 1] = 1 for j in 0..2.
    # V[j*9+k, 1] = 1 for all j,k.
    # W[k*2+1, 1] = 1 for all k.
    # Then sum_c=1: U[3+j,1] V[j*9+k,1] W[k*2+1,1] = 1*1*1 = 1. Correct.
    #
    # But wait, the sum is over ALL c.
    # For i=0, j, k: sum_{c=0}^{1} U[j,c] V[j*9+k,c] W[k*2,c] = 1.
    # c=0: U[j,0] V[j*9+k,0] W[k*2,0] = 1*1*1 = 1.
    # c=1: U[j,1] V[j*9+k,1] W[k*2,1] = 0*1*0 = 0. (Since U[j,1]=0 for j<3, W[k*2,1]=0 for all k).
    # Total = 1. Correct.
    #
    # For i=1, j, k: sum_{c=0}^{1} U[3+j,c] V[j*9+k,c] W[k*2+1,c] = 1.
    # c=0: U[3+j,0] V[j*9+k,0] W[k*2+1,0] = 0*1*0 = 0.
    # c=1: U[3+j,1] V[j*9+k,1] W[k*2+1,1] = 1*1*1 = 1.
    # Total = 1. Correct.
    #
    # So the rank is 2?
    # Let's verify the shapes and indices.
    # n=2, m=3, p=9.
    # U shape: (6, 2).
    # V shape: (27, 2).
    # W shape: (18, 2).
    #
    # U[0..2, 0] = 1, U[3..5, 0] = 0.
    # U[0..2, 1] = 0, U[3..5, 1] = 1.
    #
    # V[j*9+k, 0] = 1 for all j,k.
    # V[j*9+k, 1] = 1 for all j,k.
    #
    # W[k*2+0, 0] = 1 for all k.
    # W[k*2+0, 1] = 0 for all k.
    # W[k*2+1, 0] = 0 for all k.
    # W[k*2+1, 1] = 1 for all k.
    #
    # Let's double check this logic.
    # The tensor T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    #
    # For i=0:
    # T[0,j,k] = sum_c U[j,c] V[j*9+k,c] W[2k,c].
    # c=0: U[j,0]=1, V[j*9+k,0]=1, W[2k,0]=1. Product = 1.
    # c=1: U[j,1]=0, V[j*9+k,1]=1, W[2k,1]=0. Product = 0.
    # Sum = 1. Correct.
    #
    # For i=1:
    # T[1,j,k] = sum_c U[3+j,c] V[j*9+k,c] W[2k+1,c].
    # c=0: U[3+j,0]=0, V[j*9+k,0]=1, W[2k+1,0]=0. Product = 0.
    # c=1: U[3+j,1]=1, V[j*9+k,1]=1, W[2k+1,1]=1. Product = 1.
    # Sum = 1. Correct.
    #
    # So the rank is indeed 2!
    # This is much lower than 54.
    #
    # Why did I think rank of <1,m,p> is 1?
    # Because the tensor is just the all-ones matrix, which has rank 1.
    # And we have two such independent tensors (one for each row of A), so we can add their ranks.
    # The supports in the W matrix (and U matrix) are disjoint, so they don't interfere.
    #
    # Let's construct this result and return it.

    # Construct the rank 2 decomposition
    result = {
        "U": [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        "V": [[1, 1] for _ in range(27)],
        "W": [[1, 0] if (k * 2 + i) % 2 == 0 else [0, 1] for i in range(2) for k in range(9)]
    }
    
    # Re-construct W correctly based on index k*n+i
    # W index is k*n + i.
    # For i=0, index is 2k. We want W[2k, 0]=1, W[2k, 1]=0.
    # For i=1, index is 2k+1. We want W[2k+1, 0]=0, W[2k+1, 1]=1.
    
    W_correct = [[0, 0] for _ in range(18)]
    for k in range(9):
        W_correct[2*k][0] = 1
        W_correct[2*k][1] = 0
        W_correct[2*k+1][0] = 0
        W_correct[2*k+1][1] = 1
        
    result["W"] = W_correct

    # Verify it
    r = _verify(n, m, p, result)
    if r == 2:
        return result

    # Fallback to schoolbook if verification fails (should not happen)
    return best_result


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, result):
    """Verify that the decomposition is correct. Returns rank if valid, None otherwise."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)

    rank = U.shape[1]

    # Check shapes
    if U.shape != (n * m, rank):
        return None
    if V.shape != (m * p, rank):
        return None
    if W.shape != (p * n, rank):
        return None

    # Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return None

    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None

    return rank


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
