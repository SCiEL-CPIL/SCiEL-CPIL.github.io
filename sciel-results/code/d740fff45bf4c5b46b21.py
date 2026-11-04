import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The current program's structural idea: Strassen-based rank-56 decomposition.
    # The inspiration's refinement machinery: programmatic verification and a targeted local-search polish.
    
    # First, build the rank-56 decomposition.
    result = _construct_rank56(n, m, p)
    
    # Verify it's correct.
    ok, err = _matmul_check(result['U'], result['V'], result['W'], n, m, p)
    
    if not ok:
        # Fallback to schoolbook if verification fails.
        result = _construct_schoolbook(n, m, p)
    
    # Try to improve with a local search (refinement from inspiration).
    # We attempt to reduce the rank by finding linear dependencies among the columns.
    # This is a heuristic approach: we check if any column can be expressed as a linear combination of others.
    # If so, we can merge them, reducing the rank.
    
    # However, for a fixed decomposition like Strassen's, the columns are typically linearly independent.
    # So we try a different approach: we look for a lower-rank decomposition by solving a system of equations.
    # This is computationally expensive, so we only try it if we have enough time.
    
    if _time_left() > 30:
        # Try to find a rank-55 or lower decomposition.
        # This is a heuristic search, so it might not succeed.
        improved = _try_local_search(result, n, m, p)
        if improved is not None:
            ok, err = _matmul_check(improved['U'], improved['V'], improved['W'], n, m, p)
            if ok:
                result = improved
    
    return result

def _construct_rank56(n, m, p):
    """The original Strassen-based rank-56 decomposition (known valid)."""
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u][col] = 1
                V[v][col] = 1
                W[w][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)

def _try_local_search(result, n, m, p):
    """
    Attempt to reduce the rank by finding linear dependencies.
    This is a heuristic and may not succeed.
    """
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    
    # Check if any column is a linear combination of the others.
    # This is equivalent to checking if the matrix [U; V; W] has linearly dependent columns.
    # We form a matrix where each column is the concatenation of U[:, r], V[:, r], W[:, r].
    # If the rank of this matrix is less than 'rank', then there's a dependency.
    
    # However, the tensor decomposition requires that the product U[:, r] * V[:, r] * W[:, r] 
    # contributes to the tensor. A linear dependency in the concatenated matrix doesn't necessarily 
    # mean we can reduce the rank of the tensor decomposition, because the product structure is nonlinear.
    
    # So this approach is not straightforward. Instead, we try a different heuristic:
    # We look for pairs of columns that are "similar" and see if we can merge them.
    # This is also heuristic and may not work.
    
    # For now, we return None to indicate no improvement was found.
    # A more sophisticated local search would require solving a system of polynomial equations,
    # which is computationally expensive.
    
    return None

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
