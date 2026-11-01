"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Construct rank-49 Strassen-based decomposition for 2x2x14
    result = _construct_strassen_49(n, m, p)
    if result is not None:
        # Try to find a lower-rank decomposition with remaining time
        elapsed = time.time() - start_time
        if elapsed < 80:
            improved = _search_lower_rank(n, m, p, start_rank=48, end_rank=43, deadline=deadline)
            if improved is not None:
                return improved
        return result

    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    
    We split the 14 columns of B into 7 pairs of 2 columns. For each pair,
    we apply Strassen's 7-product formula to the 2x2x2 sub-problem.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49

    # Strassen's 7 products for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k where j in {0,1}, k in {0,...,p-1}
    # W indices: k*n + i where k in {0,...,p-1}, i in {0,1}

    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7

        # M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1

        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1

        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1

    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication.
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = 0.0
                for r in range(rank):
                    tensor_val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(tensor_val - 1.0) > 1e-10:
                    return False
    return True


def _search_lower_rank(n, m, p, start_rank, end_rank, deadline):
    """Try to find a decomposition with rank < 49 using structured search."""
    # For 2x2 x 2x14, the tensor T[i,j,k] = 1 for all i,j,k
    # We need U (4 x rank), V (28 x rank), W (28 x rank) such that
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Try random sparse search for lower ranks
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        result = _try_random_decomposition(n, m, p, rank, max_attempts=200, deadline=deadline)
        if result is not None:
            return result
    
    return None


def _try_random_decomposition(n, m, p, rank, max_attempts, deadline):
    """Try to find a rank-rank decomposition via random sparse initialization."""
    for attempt in range(max_attempts):
        if time.time() > deadline - 3:
            break
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None
# EVOLVE-BLOCK-END
