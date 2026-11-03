"""Tensor rank decomposition for 2x4 x 4x6 matrix multiplication.

Strategy:
1. Construct a rank-42 Strassen-based baseline (deterministic, fast).
2. Verify it exactly with integer arithmetic.
3. Attempt rank reduction via:
   a. Randomized ALS with half-integer rounding and exact verification.
   b. Greedy column-pruning of the Strassen basis (structural change).
4. Return the lowest-rank exact decomposition found within the time budget.
"""
import time
import sys
import random
from fractions import Fraction
from itertools import combinations

import numpy as np

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6

DEADLINE = time.time() + 145  # safety margin before 150s kill


def _verify(U, V, W):
    """Exact verification using Fraction arithmetic."""
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
    """Rank-42 Strassen composition using 2x2x2 Strassen terms."""
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
    U_new = U.copy()
    for i in range(N):
        for j1 in range(M):
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


def _round_halfint(x):
    return np.round(x * 2) / 2


def _try_rank_als(r, max_iters=100, n_restarts=4, seed=None):
    """Try to find a rank-r decomposition via randomized ALS."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 15:
            break
        U = np.random.randn(N * M, r)
        V = np.random.randn(M * P, r)
        W = np.random.randn(P * N, r)
        for it in range(max_iters):
            if time.time() > DEADLINE - 15:
                break
            U, V, W = _als_step(U, V, W)
            Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
            if _verify(Ur, Vr, Wr):
                return Ur, Vr, Wr
        Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
        if _verify(Ur, Vr, Wr):
            return Ur, Vr, Wr
    return None


def _greedy_prune(U, V, W, target_rank, max_time=30):
    """
    Structural change: Greedy column pruning from the Strassen basis.
    Try removing columns one at a time, keeping the decomposition exact.
    This exploits the fact that the rank-42 basis may have linear dependencies
    that allow exact removal of some terms.
    """
    start_time = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    
    while r > target_rank:
        if time.time() > start_time + max_time or time.time() > DEADLINE - 10:
            break
        
        # Try removing each column
        removed = False
        for col in range(r):
            if time.time() > start_time + max_time or time.time() > DEADLINE - 10:
                break
            
            # Temporarily remove column col
            mask = [i for i in range(r) if i != col]
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            if _verify(U_sub, V_sub, W_sub):
                U, V, W = U_sub, V_sub, W_sub
                r = r - 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W


def _try_row_exploitation(U, V, W, max_time=20):
    """
    Exploit n=2 structure: try to combine the two output rows' contributions.
    For each pair of terms, check if they can be merged or if one row's
    contribution is linearly dependent on the other.
    This is a heuristic search over the 42 Strassen terms.
    """
    start_time = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    
    # Try to find pairs of columns that are "redundant" in a specific sense
    # If column c1 and c2 have W values that are proportional for one row
    # but not the other, they might be combinable.
    
    # Simpler approach: try all single-column removals (already done in greedy_prune)
    # Here, try 2-column removals
    while r > 10:
        if time.time() > start_time + max_time or time.time() > DEADLINE - 10:
            break
        
        removed = False
        for c1 in range(r):
            if time.time() > start_time + max_time or time.time() > DEADLINE - 10:
                break
            for c2 in range(c1 + 1, r):
                if time.time() > start_time + max_time or time.time() > DEADLINE - 10:
                    break
                
                mask = [i for i in range(r) if i != c1 and i != c2]
                U_sub = U[:, mask]
                V_sub = V[:, mask]
                W_sub = W[:, mask]
                
                if _verify(U_sub, V_sub, W_sub):
                    U, V, W = U_sub, V_sub, W_sub
                    r = r - 2
                    removed = True
                    break
            if removed:
                break
        
        if not removed:
            break
    
    return U, V, W


def construct():
    start = time.time()
    
    # Step 1: Construct and verify the rank-42 Strassen baseline
    U, V, W = _strassen_baseline()
    if not _verify(U, V, W):
        # Fallback: schoolbook rank-24
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
    best = (U.copy(), V.copy(), W.copy())
    
    # Step 2: Greedy column pruning (structural change)
    U_pruned, V_pruned, W_pruned = _greedy_prune(U, V, W, target_rank=10, max_time=20)
    if U_pruned.shape[1] < best_rank:
        best_rank = U_pruned.shape[1]
        best = (U_pruned, V_pruned, W_pruned)
    
    # Step 3: Try 2-column removals
    U2, V2, W2 = _try_row_exploitation(best[0], best[1], best[2], max_time=15)
    if U2.shape[1] < best_rank:
        best_rank = U2.shape[1]
        best = (U2, V2, W2)
    
    # Step 4: Try ALS for lower ranks
    for r in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > DEADLINE - 20:
            break
        res = _try_rank_als(r, max_iters=80, n_restarts=3, seed=r * 1000)
        if res is not None:
            U, V, W = res
            best_rank = r
            best = (U, V, W)
            break
    
    U, V, W = best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
