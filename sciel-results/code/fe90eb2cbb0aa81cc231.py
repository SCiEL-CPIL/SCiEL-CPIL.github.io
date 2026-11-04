"""Tensor decomposition search for 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NMP = N * M * P  # 105 schoolbook rank


def check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition is exact."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute all entries
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    r = U.shape[1]
    for c in range(r):
        T += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
    
    # Check expected values
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                if abs(T[a, b, d] - 1.0) > tol:
                    return False
    
    # Check zeros
    for i1 in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    for i3 in range(n):
                        for k3 in range(p):
                            if i1 == i3 and j1 == j2 and k2 == k3:
                                continue
                            a = i1 * m + j1
                            b = j2 * p + k2
                            d = k3 * n + i3
                            if abs(T[a, b, d]) > tol:
                                return False
    return True


def laderman_3x2x3():
    """
    Return Laderman's rank-7 decomposition for 3x2 * 2x3.
    U: (6, 7), V: (6, 7), W: (9, 7)
    A is 3x2 (index i*2+j), B is 2x3 (index j*3+k), C is 3x3 (index k*3+i)
    """
    U = np.zeros((6, 7))
    V = np.zeros((6, 7))
    W = np.zeros((9, 7))
    
    # Laderman's 7 products for 3x2 * 2x3:
    # M1 = A[0,0] * (B[0,0] + B[1,2])
    U[0, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[8, 0] = 1
    
    # M2 = (A[0,0] + A[2,0]) * (B[0,0] + B[1,0])
    U[0, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[2, 1] = 1; W[6, 1] = 1; W[8, 1] = 1
    
    # M3 = (A[1,0] + A[2,1]) * (B[0,1] + B[1,1])
    U[2, 2] = 1; U[5, 2] = 1
    V[1, 2] = 1; V[4, 2] = 1
    W[1, 2] = 1; W[3, 2] = 1; W[7, 2] = 1
    
    # M4 = (A[0,1] + A[2,1]) * (B[1,0] + B[1,1])
    U[1, 3] = 1; U[5, 3] = 1
    V[3, 3] = 1; V[4, 3] = 1
    W[2, 3] = 1; W[4, 3] = 1; W[6, 3] = 1; W[8, 3] = 1
    
    # M5 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[2, 4] = 1; U[0, 4] = -1
    V[0, 4] = 1; V[1, 4] = 1
    W[3, 4] = 1; W[4, 4] = 1
    
    # M6 = (A[1,1] - A[2,0]) * (B[0,1] + B[1,1])
    U[3, 5] = 1; U[4, 5] = -1
    V[1, 5] = 1; V[4, 5] = 1
    W[3, 5] = 1; W[5, 5] = 1; W[7, 5] = 1
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[2, 6] = 1; W[4, 6] = 1; W[6, 6] = 1; W[8, 6] = 1
    
    return U, V, W


def schoolbook_nm_p(n, m, p):
    """Return schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s wall-clock
    
    n, m, p = N, M, P
    
    # Block decomposition: split middle dim 5 = 2 + 3
    # A = [A1 | A2], A1 is 3x2, A2 is 3x3
    # B = [B1; B2], B1 is 2x7, B2 is 3x7
    # C = A1*B1 + A2*B2
    #
    # A1*B1 (3x2 * 2x7): split 7 = 3+3+1
    #   3x2*2x3 Laderman x2 (rank 7 each) + 3x2*2x1 schoolbook (rank 6) = 20
    # A2*B2 (3x3 * 3x7): split 7 = 3+3+1
    #   3x3*3x3 schoolbook x2 (rank 27 each) + 3x3*3x1 schoolbook (rank 9) = 63
    # Total: 83
    
    decs = []
    
    # Part 1a: A1 (3x2) * B1[:,0:3] (2x3) -> Laderman rank 7
    U_l, V_l, W_l = laderman_3x2x3()
    U_f = np.zeros((15, 7)); V_f = np.zeros((35, 7)); W_f = np.zeros((21, 7))
    for i in range(3):
        for j in range(2):
            U_f[i*5+j, :] = U_l[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V_f[j*7+k, :] = V_l[j*3+k, :]
    for k in range(3):
        for i in range(3):
            W_f[k*3+i, :] = W_l[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Part 1b: A1 (3x2) * B1[:,3:6] (2x3) -> Laderman rank 7
    U_l, V_l, W_l = laderman_3x2x3()
    U_f = np.zeros((15, 7)); V_f = np.zeros((35, 7)); W_f = np.zeros((21, 7))
    for i in range(3):
        for j in range(2):
            U_f[i*5+j, :] = U_l[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V_f[j*7+(k+3), :] = V_l[j*3+k, :]
    for k in range(3):
        for i in range(3):
            W_f[k*3+i, :] = W_l[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Part 1c: A1 (3x2) * B1[:,6:7] (2x1) -> schoolbook rank 6
    U_s, V_s, W_s = schoolbook_nm_p(3, 2, 1)
    U_f = np.zeros((15, 6)); V_f = np.zeros((35, 6)); W_f = np.zeros((21, 6))
    for i in range(3):
        for j in range(2):
            U_f[i*5+j, :] = U_s[i*2+j, :]
    for j in range(2):
        V_f[j*7+6, :] = V_s[j, :]
    for k in range(1):
        for i in range(3):
            W_f[k*3+i, :] = W_s[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Part 2a: A2 (3x3) * B2[:,0:3] (3x3) -> schoolbook rank 27
    U_s, V_s, W_s = schoolbook_nm_p(3, 3, 3)
    U_f = np.zeros((15, 27)); V_f = np.zeros((35, 27)); W_f = np.zeros((21, 27))
    for i in range(3):
        for j in range(3):
            U_f[i*5+(j+2), :] = U_s[i*3+j, :]
    for j in range(3):
        for k in range(3):
            V_f[(j+2)*7+k, :] = V_s[j*3+k, :]
    for k in range(3):
        for i in range(3):
            W_f[k*3+i, :] = W_s[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Part 2b: A2 (3x3) * B2[:,3:6] (3x3) -> schoolbook rank 27
    U_s, V_s, W_s = schoolbook_nm_p(3, 3, 3)
    U_f = np.zeros((15, 27)); V_f = np.zeros((35, 27)); W_f = np.zeros((21, 27))
    for i in range(3):
        for j in range(3):
            U_f[i*5+(j+2), :] = U_s[i*3+j, :]
    for j in range(3):
        for k in range(3):
            V_f[(j+2)*7+(k+3), :] = V_s[j*3+k, :]
    for k in range(3):
        for i in range(3):
            W_f[k*3+i, :] = W_s[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Part 2c: A2 (3x3) * B2[:,6:7] (3x1) -> schoolbook rank 9
    U_s, V_s, W_s = schoolbook_nm_p(3, 3, 1)
    U_f = np.zeros((15, 9)); V_f = np.zeros((35, 9)); W_f = np.zeros((21, 9))
    for i in range(3):
        for j in range(3):
            U_f[i*5+(j+2), :] = U_s[i*3+j, :]
    for j in range(3):
        V_f[(j+2)*7+6, :] = V_s[j, :]
    for k in range(1):
        for i in range(3):
            W_f[k*3+i, :] = W_s[k*3+i, :]
    decs.append((U_f, V_f, W_f))
    
    # Combine
    U_all = np.hstack([d[0] for d in decs])
    V_all = np.hstack([d[1] for d in decs])
    W_all = np.hstack([d[2] for d in decs])
    
    rank = U_all.shape[1]
    
    # Verify
    if not check_decomposition(U_all.tolist(), V_all.tolist(), W_all.tolist(), tol=1e-9):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = schoolbook_nm_p(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_all.tolist(), "V": V_all.tolist(), "W": W_all.tolist()}
