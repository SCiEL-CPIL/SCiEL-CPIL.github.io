"""Tensor rank decomposition for 2x4 x 4x6 matrix multiplication."""
import time
import sys
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6

DEADLINE = time.time() + 118  # safety margin


def _verify(U, V, W):
    r = U.shape[1]
    UF = [[Fraction(U[i, t]) for t in range(r)] for i in range(N * M)]
    VF = [[Fraction(V[i, t]) for t in range(r)] for i in range(M * P)]
    WF = [[Fraction(W[i, t]) for t in range(r)] for i in range(P * N)]
    for i in range(N):
        for k in range(P):
            for j1 in range(M):
                for j2 in range(M):
                    target = 1 if j1 == j2 else 0
                    s = Fraction(0)
                    for t in range(r):
                        s += UF[i * M + j1][t] * VF[j2 * P + k][t] * WF[k * N + i][t]
                    if s != target:
                        return False
    return True


def _strassen_baseline():
    """Rank-42 Strassen composition."""
    # 2x2x2 Strassen terms (U_t, V_t, W_t) each length 4
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    rank = 42
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    for a_block in range(2):
        for b_block in range(3):
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = terms[t]
                a_indices = [0, 1, 4, 5] if a_block == 0 else [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i], col] = U_t[i]
                b_row_start = 0 if a_block == 0 else 2
                b_col_start = b_block * 2
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1,
                ]
                for i in range(4):
                    V[b_indices[i], col] = V_t[i]
                w_indices = [
                    2 * b_block,
                    2 * b_block + 1,
                    6 + 2 * b_block,
                    6 + 2 * b_block + 1,
                ]
                for i in range(4):
                    W[w_indices[i], col] = W_t[i]
    return U, V, W


def _als_step(U, V, W):
    """One ALS iteration: fix two, solve third by least squares."""
    r = U.shape[1]
    # Solve U: for each (i, j1), linear system over (j2, k)
    U_new = U.copy()
    for i in range(N):
        for j1 in range(M):
            # A_mat[(j2,k), t] = V[j2*P+k, t] * W[k*N+i, t]
            A_mat = np.zeros((M * P, r))
            b = np.zeros(M * P)
            for j2 in range(M):
                for k in range(P):
                    A_mat[j2 * P + k, :] = V[j2 * P + k, :] * W[k * N + i, :]
                    b[j2 * P + k] = 1.0 if j2 == j1 else 0.0
            sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
            U_new[i * M + j1, :] = sol
    V_new = V.copy()
    for j2 in range(M):
        for k in range(P):
            A_mat = np.zeros((N * M, r))
            b = np.zeros(N * M)
            for i in range(N):
                for j1 in range(M):
                    A_mat[i * M + j1, :] = U_new[i * M + j1, :] * W[k * N + i, :]
                    b[i * M + j1] = 1.0 if j1 == j2 else 0.0
            sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
            V_new[j2 * P + k, :] = sol
    W_new = W.copy()
    for i in range(N):
        for k in range(P):
            A_mat = np.zeros((N * M, r))
            b = np.zeros(N * M)
            for j1 in range(M):
                for j2 in range(M):
                    A_mat[i * M + j1, :] = U_new[i * M + j1, :] * V_new[j2 * P + k, :]
                    b[i * M + j1] = 1.0 if j1 == j2 else 0.0
            sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
            W_new[k * N + i, :] = sol
    return U_new, V_new, W_new


def _residual(U, V, W):
    r = U.shape[1]
    err = 0.0
    for i in range(N):
        for k in range(P):
            for j1 in range(M):
                for j2 in range(M):
                    target = 1.0 if j1 == j2 else 0.0
                    s = 0.0
                    for t in range(r):
                        s += U[i * M + j1, t] * V[j2 * P + k, t] * W[k * N + i, t]
                    err += (s - target) ** 2
    return err


def _round_halfint(x):
    return np.round(x * 2) / 2


def _try_rank(r, max_iters=200, n_restarts=3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    best = None
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        U = np.random.randn(N * M, r)
        V = np.random.randn(M * P, r)
        W = np.random.randn(P * N, r)
        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            U, V, W = _als_step(U, V, W)
            # round and check
            Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
            if _verify(Ur, Vr, Wr):
                return Ur, Vr, Wr
        # final check
        Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
        if _verify(Ur, Vr, Wr):
            return Ur, Vr, Wr
    return None


def construct():
    start = time.time()
    # Baseline
    U, V, W = _strassen_baseline()
    if not _verify(U, V, W):
        # schoolbook fallback
        rank = N * M * P
        U = np.zeros((N * M, rank))
        V = np.zeros((M * P, rank))
        W = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1
                    V[j * P + k, col] = 1
                    W[k * N + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    best_rank = U.shape[1]
    best = (U, V, W)

    # Try to reduce rank
    for r in range(best_rank - 1, 20, -1):
        if time.time() > DEADLINE - 20:
            break
        res = _try_rank(r, max_iters=150, n_restarts=2)
        if res is not None:
            U, V, W = res
            best_rank = r
            best = (U, V, W)
            break  # found lower rank, stop (could continue but time)

    U, V, W = best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
