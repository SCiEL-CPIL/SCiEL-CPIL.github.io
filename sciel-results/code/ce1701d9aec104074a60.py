"""Matrix multiplication 4x4 * 4x7: search for rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = DIMENSIONS
RANK = N * M * P          # schoolbook rank = 112
DEADLINE = 130.0          # safety margin before the 150s kill

def _schoolbook():
    U = np.zeros((N * M, RANK)); V = np.zeros((M * P, RANK)); W = np.zeros((P * N, RANK))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1.0
                V[j * P + k, c] = 1.0
                W[k * N + i, c] = 1.0
                c += 1
    return U, V, W

def _tensor_mat(U, V, W, r):
    """T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]"""
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = float(np.dot(U[i * M + j, :r], V[j * P + k, :r]) * W[k * N + i, :r][:r].dot(np.ones(r)) if False else np.einsum('c,c,c->', U[i*M+j,:r], V[j*P+k,:r], W[k*N+i,:r]))
    return T

def _tensor_mat_fast(U, V, W, r):
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            u = U[i * M + j, :r]
            for k in range(P):
                T[i, j, k] = np.einsum('c,c,c->', u, V[j * P + k, :r], W[k * N + i, :r])
    return T

def _target():
    return np.ones((N, M, P))

def _verify(U, V, W, r, tol=1e-6):
    T = _tensor_mat_fast(U, V, W, r)
    return np.max(np.abs(T - 1.0)) < tol

def _als(U, V, W, r, iters, lr=1.0, seed=0):
    """Randomized ALS on the multiplication tensor with fixed W structure."""
    rng = np.random.default_rng(seed)
    U = U.copy(); V = V.copy()
    T = _target()
    # Precompute W slices as rows for speed
    best_err = None
    for it in range(iters):
        # update U: for each (i,j), U[i,j,:] ~ V[j,:,:]^T * (sum_k W[k,i,:]*T[i,j,k])
        for i in range(N):
            for j in range(M):
                num = np.zeros(r)
                for k in range(P):
                    num += T[i, j, k] * W[k * N + i, :r] * V[j * P + k, :r]
                U[i * M + j, :r] = num
        # update V
        for j in range(M):
            for k in range(P):
                num = np.zeros(r)
                for i in range(N):
                    num += T[i, j, k] * U[i * M + j, :r] * W[k * N + i, :r]
                V[j * P + k, :r] = num
        err = np.max(np.abs(_tensor_mat_fast(U, V, W, r) - 1.0))
        if best_err is None or err < best_err:
            best_err = err
    return U, V, best_err

def _to_half_int(mat, r):
    """Round to half-integers and verify; return (rank, U,V,W) if valid else None."""
    Ur = np.round(mat[0] * 2) / 2.0
    Vr = np.round(mat[1] * 2) / 2.0
    Wr = np.round(mat[2] * 2) / 2.0
    if _verify(Ur, Vr, Wr, r):
        return r, Ur, Vr, Wr
    return None

def construct():
    start = time.time()
    U0, V0, W0 = _schoolbook()
    best = (RANK, U0.copy(), V0.copy(), W0.copy())
    target = _target()

    # Try decreasing ranks with randomized ALS
    for r in range(100, 50, -1):
        if time.time() - start > DEADLINE:
            break
        for trial in range(30):
            if time.time() - start > DEADLINE:
                break
            rng = np.random.default_rng(1000 + trial)
            U = U0[:, :r] + rng.normal(0, 0.5, size=(N * M, r))
            V = V0[:, :r] + rng.normal(0, 0.5, size=(M * P, r))
            W = W0[:, :r] + rng.normal(0, 0.5, size=(P * N, r))
            U, V, err = _als(U, V, W, r, iters=60, seed=trial)
            # also try updating W
            for it in range(60):
                for k in range(P):
                    for i in range(N):
                        num = np.zeros(r)
                        for j in range(M):
                            num += target[i, j, k] * U[i * M + j, :r] * V[j * P + k, :r]
                        W[k * N + i, :r] = num
                err = np.max(np.abs(_tensor_mat_fast(U, V, W, r) - 1.0))
            res = _to_half_int((U, V, W), r)
            if res is not None:
                rank, Ur, Vr, Wr = res
                if rank < best[0]:
                    best = (rank, Ur, Vr, Wr)
                    # found a better rank; keep searching lower
                    continue
    # Final: verify best and return as lists
    rank, U, V, W = best
    def to_list(A, r):
        return [[float(A[i, j]) for j in range(r)] for i in range(A.shape[0])]
    return {"U": to_list(U, rank), "V": to_list(V, rank), "W": to_list(W, rank)}
