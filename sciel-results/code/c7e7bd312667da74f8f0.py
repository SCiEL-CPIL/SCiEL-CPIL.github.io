import numpy as np
from itertools import product

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check shapes
    assert len(U) == n*m
    assert len(V) == m*p
    assert len(W) == p*n
    assert all(len(row) == rank for row in U)
    assert all(len(row) == rank for row in V)
    assert all(len(row) == rank for row in W)
    
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if val != int(val) and 2*val != int(2*val):
                    return False
    
    # Verify tensor equation
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    # Actually: sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_j A[i,j]*B[j,k]
    # For this to hold for all A, B, we need:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j,j']
    # for all i, j, j', k
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] should be delta[j,jp]
                    s = np.sum(U[i*m+j] * V[jp*p+k] * W[k*n+i])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True
