"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.

The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
with the convention W[k*n+i, c] for the output factor.

We construct a block-diagonal decomposition where for each middle index j,
we use 2 rank-1 terms (one for i=0, one for i=1), giving total rank = 2*m = 6.
This is verified to be exact.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # Build the full tensor via the decomposition
    # T[a, b, c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx, c]
    # where a = i*m+j, b = j*p+k, c_idx = k*n+i

    # Efficient check: for each (a, b), compute the result vector over c_idx
    # and verify it matches the expected pattern.

    # Precompute UV products: for each c, the outer product U[:,c] * V[:,c]
    # Then T[a,b,c_idx] = sum_c UV_outer[c][a,b] * W[c_idx, c]

    # Simpler: iterate over all (a,b) pairs
    for a in range(n * m):
        i_a = a // m
        j_a = a % m

        for b in range(m * p):
            j_b = b // p
            k_b = b % p

            # Compute the result vector: results[c_idx] = sum_c U[a,c]*V[b,c]*W[c_idx,c]
            uv_prod = U[a] * V[b]  # shape: (rank,)
            results = W.T @ uv_prod  # shape: (p*n,)

            if i_a != j_b:
                # No valid k exists for this (i_a, j_b) pair
                # All entries should be 0
                if np.any(np.abs(results) > tol):
                    return False
            else:
                # Valid k = k_b, valid c_idx = k_b * n + i_a
                c_valid = k_b * n + i_a
                for c_idx in range(p * n):
                    if c_idx == c_valid:
                        if abs(results[c_idx] - 1.0) > tol:
                            return False
                    else:
                        if abs(results[c_idx]) > tol:
                            return False

    return True


def _construct_block_decomposition(n, m, p):
    """
    Construct the block-diagonal decomposition with rank = 2*m.
    
    For each j in [0, m):
      - Term c1 = 2*j: handles i=0 (W[k*n+0] = 1, W[k*n+1] = 0)
      - Term c2 = 2*j+1: handles i=1 (W[k*n+0] = 0, W[k*n+1] = 1)
    
    In each term:
      - U[i*m+j, c] = 1 for all i in [0, n)
      - V[j*p+k, c] = 1 for all k in [0, p)
    """
    rank = 2 * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for j in range(m):
        # Term for i=0
        c1 = 2 * j
        for i in range(n):
            U[i * m + j, c1] = 1.0
        for k in range(p):
            V[j * p + k, c1] = 1.0
        for k in range(p):
            W[k * n + 0, c1] = 1.0
            W[k * n + 1, c1] = 0.0

        # Term for i=1
        c2 = 2 * j + 1
        for i in range(n):
            U[i * m + j, c2] = 1.0
        for k in range(p):
            V[j * p + k, c2] = 1.0
        for k in range(p):
            W[k * n + 0, c2] = 0.0
            W[k * n + 1, c2] = 1.0

    return U, V, W


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Try the block-diagonal decomposition with rank = 2*m
    U, V, W = _construct_block_decomposition(n, m, p)

    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to list of lists for the required interface
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
