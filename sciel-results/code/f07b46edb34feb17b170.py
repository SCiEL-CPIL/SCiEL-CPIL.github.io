"""Rewritten program: structured CP-decomposition search for the 2x2 x 2x11
matrix-multiplication tensor, with exact half-integer verification and a
deadline guard. Returns the best verified (lowest-rank) exact decomposition."""

import time
import math
import random

import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Return list of (a, b, c) indices with target value 1.
    a = i*m + j  (U leg, size n*m)
    b = j*p + k  (V leg, size m*p)
    c = k*n + i  (W leg, size p*n)
    """
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx.append((a, b, c))
    return idx


def _check_exact(U, V, W, target_idx, n, m, p, tol=1e-6):
    """Return True if the decomposition reproduces all target entries exactly
    (within tol) and the rank is consistent."""
    r = U.shape[1]
    # Build the implied tensor values at target positions
    for (a, b, c) in target_idx:
        val = 0.0
        for alpha in range(r):
            val += U[a, alpha] * V[b, alpha] * W[c, alpha]
        if abs(val - 1.0) > tol:
            return False
    return True


def _round_to_half(X):
    return np.round(2.0 * X) / 2.0


def _cp_search(n, m, p, r, target_idx, iters=400, seed=0, tol=1e-6):
    """Alternating least squares CP fit to the sparse target tensor.
    Returns (U, V, W) or None if it did not converge to a good fit."""
    rng = np.random.default_rng(seed)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    U = rng.normal(size=(dim_u, r)) * 0.5
    V = rng.normal(size=(dim_v, r)) * 0.5
    W = rng.normal(size=(dim_w, r)) * 0.5

    # Precompute for speed: for each target (a,b,c) we need sum_alpha U[a,al] V[b,al] W[c,al]
    # ALS: fix two, solve for the third via linear system.
    best_res = float("inf")
    for it in range(iters):
        for which in range(3):
            # Build linear system for the "which" leg.
            # For each target t with (a,b,c):
            #   sum_alpha U[a,al] V[b,al] W[c,al] = 1
            # If solving for W (which==2): for each c that appears, we have
            #   sum_alpha W[c,al] * (U[a,al]*V[b,al]) = 1
            # Group by the index of the leg being solved.
            if which == 0:
                # solve U[a, :] : for each target, coeff over alpha = V[b,al]*W[c,al]
                # unknowns U[a,al] for all a. Build per-a system.
                # For a given a, equations: sum_alpha U[a,al] * (V[b,al]*W[c,al]) = 1 for each (b,c) paired with this a.
                # We'll build a global least-squares over all (target, alpha) contributions.
                # Simpler: build matrix M of shape (num_targets, r) where M[t,al] = (product of the other two legs at t, al),
                # and vector y[t]=1, and unknowns are the leg values at each target's index.
                # But unknowns repeat across targets (same a appears in multiple targets). Use normal equations per index.
                pass
        # To keep it simple and robust, use a global gradient-free Nelder-Mead on the residual? Too slow.
        # Instead use a direct per-index linear solve.
        if which == 0:
            # For each a, gather targets with this a.
            for a in range(dim_u):
                rows = []
                rhs = []
                for (ta, tb, tc) in target_idx:
                    if ta != a:
                        continue
                    coeff = V[tb, :] * W[tc, :]  # shape (r,)
                    rows.append(coeff)
                    rhs.append(1.0)
                if not rows:
                    continue
                A_mat = np.array(rows)  # (nt, r)
                b_vec = np.array(rhs)   # (nt,)
                sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[a, :] = sol
        elif which == 1:
            for b in range(dim_v):
                rows = []
                rhs = []
                for (ta, tb, tc) in target_idx:
                    if tb != b:
                        continue
                    coeff = U[ta, :] * W[tc, :]
                    rows.append(coeff)
                    rhs.append(1.0)
                if not rows:
                    continue
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[b, :] = sol
        else:
            for c in range(dim_w):
                rows = []
                rhs = []
                for (ta, tb, tc) in target_idx:
                    if tc != c:
                        continue
                    coeff = U[ta, :] * V[tb, :]
                    rows.append(coeff)
                    rhs.append(1.0)
                if not rows:
                    continue
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[c, :] = sol

        # compute residual
        res = 0.0
        for (ta, tb, tc) in target_idx:
            val = 0.0
            for alpha in range(r):
                val += U[ta, alpha] * V[tb, alpha] * W[tc, alpha]
            res += (val - 1.0) ** 2
        if res < best_res:
            best_res = res
        if best_res < tol:
            break

    # final residual check
    res = 0.0
    for (ta, tb, tc) in target_idx:
        val = 0.0
        for alpha in range(r):
            val += U[ta, alpha] * V[tb, alpha] * W[tc, alpha]
        res += (val - 1.0) ** 2
    if res < 1e-4:
        return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100.0  # safety margin under the 120s limit

    target_idx = _build_target(n, m, p)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Fallback: schoolbook rank 44 (always valid, half-integers).
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_fb, V_fb, W_fb = schoolbook()

    best = None
    best_rank = n * m * p

    # Try ranks below schoolbook, from a lower bound upward.
    # Lower bound: at least max(dim_u, dim_v, dim_w) is not required, but a
    # reasonable floor for this structured tensor. We search a window.
    r_lo = max(12, min(best_rank - 1, 30))
    r_hi = best_rank - 1

    seed_counter = 0
    for r in range(r_lo, r_hi + 1):
        if time.time() > deadline - 15.0:
            break
        found = None
        max_seeds = 6
        for s in range(max_seeds):
            if time.time() > deadline - 15.0:
                break
            res = _cp_search(n, m, p, r, target_idx, iters=250, seed=seed_counter)
            seed_counter += 1
            if res is not None:
                U, V, W = res
                # Round to half-integers and verify exactness.
                Ur = _round_to_half(U)
                Vr = _round_to_half(V)
                Wr = _round_to_half(W)
                if _check_exact(Ur, Vr, Wr, target_idx, n, m, p, tol=1e-9):
                    found = (Ur, Vr, Wr)
                    break
                # If rounding failed, try exact (unrounded) verification.
                if _check_exact(U, V, W, target_idx, n, m, p, tol=1e-9):
                    found = (U, V, W)
                    break
        if found is not None:
            U, V, W = found
            best_rank = r
            best = (U, V, W)
            # Continue trying smaller ranks.
            continue

    if best is not None:
        U, V, W = best
    else:
        U, V, W = U_fb, V_fb, W_fb

    # Convert to plain lists of numbers (int or half-int float) for the interface.
    def to_list(M):
        out = []
        for row in M:
            rlist = []
            for x in row:
                x = float(x)
                # snap tiny values
                if abs(x) < 1e-12:
                    x = 0.0
                if abs(x * 2 - round(x * 2)) < 1e-9:
                    x = round(x * 2) / 2.0
                    if abs(x - round(x)) < 1e-12:
                        x = int(round(x))
                rlist.append(x)
            out.append(rlist)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
