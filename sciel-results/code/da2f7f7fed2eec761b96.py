import numpy as np
import time
import random

def _build_3x2x2_factors():
    """
    Build the rank-11 decomposition for 3x2 * 2x2 -> 3x2.
    """
    nU, nV, nW = 6, 4, 6
    rk = 11
    U = np.zeros((nU, rk), dtype=object)
    V = np.zeros((nV, rk), dtype=object)
    W = np.zeros((nW, rk), dtype=object)

    # Strassen for 2x2 * 2x2 (rows 0,1 of A)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1

    # Row 2 direct terms
    U[4,7]=1; V[0,7]=1; W[4,7]=1
    U[5,8]=1; V[2,8]=1; W[4,8]=1
    U[4,9]=1; V[1,9]=1; W[5,9]=1
    U[5,10]=1; V[3,10]=1; W[5,10]=1

    return U, V, W


def _check_decomposition(U, V, W, T):
    """Check if U,V,W decompose T exactly using integer arithmetic."""
    # Use float64 for speed, but values are small integers so exact
    T_check = np.einsum('il,jl,kl->ijk', U, V, W)
    return np.allclose(T_check, T, atol=1e-10)


def _try_reduce_rank(U, V, W, target_rank, timeout=110):
    """
    Attempt to reduce the rank below the current value using multiple strategies:
    1. Greedy single-column elimination
    2. Pairwise column elimination
    3. Random subset testing
    """
    start = time.time()
    nU, cur_rank = U.shape
    nV, _ = V.shape
    nW, _ = W.shape

    # Build the full tensor T from the decomposition
    T = np.einsum('il,jl,kl->ijk', U, V, W)

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rk = cur_rank

    def time_left():
        return timeout - (time.time() - start)

    # Strategy 1: Greedy single-column elimination (multiple passes)
    for _pass in range(5):
        if time_left() < 5:
            break
        progress = False
        for col_idx in range(best_rk):
            if time_left() < 3:
                break
            if best_rk <= target_rank:
                break
            keep = [i for i in range(best_rk) if i != col_idx]
            U_sub = best_U[:, keep]
            V_sub = best_V[:, keep]
            W_sub = best_W[:, keep]
            if _check_decomposition(U_sub, V_sub, W_sub, T):
                best_U = U_sub.copy()
                best_V = V_sub.copy()
                best_W = W_sub.copy()
                best_rk -= 1
                progress = True
                if best_rk <= target_rank:
                    break
        if not progress:
            break

    # Strategy 2: Pairwise column elimination
    if time_left() > 10 and best_rk > target_rank:
        for i in range(best_rk):
            if time_left() < 5:
                break
            if best_rk <= target_rank:
                break
            for j in range(i+1, best_rk):
                if time_left() < 3:
                    break
                if best_rk <= target_rank:
                    break
                keep = [k for k in range(best_rk) if k != i and k != j]
                if len(keep) < target_rank:
                    continue
                U_sub = best_U[:, keep]
                V_sub = best_V[:, keep]
                W_sub = best_W[:, keep]
                if _check_decomposition(U_sub, V_sub, W_sub, T):
                    best_U = U_sub.copy()
                    best_V = V_sub.copy()
                    best_W = W_sub.copy()
                    best_rk -= 2
                    if best_rk <= target_rank:
                        break

    # Strategy 3: Random subset testing
    if time_left() > 5 and best_rk > target_rank:
        random.seed(42)
        for _ in range(200):
            if time_left() < 2:
                break
            if best_rk <= target_rank:
                break
            # Try removing 1-3 random columns
            n_remove = random.randint(1, min(3, best_rk - target_rank))
            if n_remove <= 0:
                break
            removed = random.sample(range(best_rk), n_remove)
            keep = [i for i in range(best_rk) if i not in removed]
            U_sub = best_U[:, keep]
            V_sub = best_V[:, keep]
            W_sub = best_W[:, keep]
            if _check_decomposition(U_sub, V_sub, W_sub, T):
                best_U = U_sub.copy()
                best_V = V_sub.copy()
                best_W = W_sub.copy()
                best_rk -= n_remove

    return best_U, best_V, best_W, best_rk


def construct():
    n, m, p = 3, 4, 4
    N = n * m  # 12
    M = m * p  # 16
    P = p * n  # 12

    deadline = time.time() + 140  # Safety margin before 150s kill

    # Build local factors
    U_loc, V_loc, W_loc = _build_3x2x2_factors()
    local_rank = 11

    # Assemble the 3x4*4x4 decomposition via 4 blocks
    rank = 4 * local_rank
    U = np.zeros((N, rank), dtype=np.float64)
    V = np.zeros((M, rank), dtype=np.float64)
    W = np.zeros((P, rank), dtype=np.float64)

    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        off = block_idx * local_rank
        for r in range(3):
            for c in range(2):
                la = r * 2 + c
                ga = r * m + (a_col_start + c)
                U[ga, off:off+local_rank] += U_loc[la, :].astype(np.float64)
        for r in range(2):
            for c in range(2):
                lb = r * 2 + c
                gb = (b_row_start + r) * p + (b_col_start + c)
                V[gb, off:off+local_rank] += V_loc[lb, :].astype(np.float64)
        for r in range(3):
            for c in range(2):
                lc = r * 2 + c
                gc = (c_col_start + c) * n + r
                W[gc, off:off+local_rank] += W_loc[lc, :].astype(np.float64)

    # 4 blocks
    add_block(0, 0, 0, 0, 0)  # A_L * B_11 -> C_L
    add_block(1, 2, 2, 0, 0)  # A_R * B_21 -> C_L
    add_block(2, 0, 0, 2, 2)  # A_L * B_12 -> C_R
    add_block(3, 2, 2, 2, 2)  # A_R * B_22 -> C_R

    # Try to reduce rank with time budget
    remaining = deadline - time.time()
    if remaining > 10:
        U_red, V_red, W_red, rk = _try_reduce_rank(U, V, W, target_rank=35, timeout=min(remaining - 5, 120))
        if rk < rank:
            U, V, W, rank = U_red, V_red, W_red, rk

    # Convert to list
    def to_list(arr):
        out = []
        for row in arr:
            out.append([float(x) for x in row])
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
