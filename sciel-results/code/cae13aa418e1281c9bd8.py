import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def build_target_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[a,b,c] where
    a = i*m + j, b = j*p + k, c = k*n + i, with the consistency constraints.
    T is 1 on consistent (i,j,k) triples, 0 otherwise."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] (outer) == T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def is_half_integer(M, tol=1e-9):
    M = np.array(M, dtype=float)
    doubled = M * 2
    rounded = np.round(doubled)
    return np.max(np.abs(doubled - rounded)) < tol

def to_half_int(M):
    M = np.array(M, dtype=float)
    return (np.round(M*2) / 2.0)

def schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
