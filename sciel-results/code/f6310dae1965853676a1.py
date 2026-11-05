"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_rank16(deadline):
    """
    Attempt to find a rank-16 decomposition via a focused local search.
    Start from rank-17, drop one column, and perform a limited number of
    coordinate-wise improvements on a small candidate set.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16

    # Precompute non-zero indices
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz.append((i*m+j, j*p+k, k*n+i))

    def err(U, V, W):
        e = 0.0
        for a,b,c in nz:
            s = 0.0
            for r in range(rank):
                s += U[a,r]*V[b,r]*W[c,r]
            e += (s-1.0)**2
        # sample a few zero positions
        for _ in range(12):
            a = random.randint(0, n_a-1)
            b = random.randint(0, n_b-1)
            c = random.randint(0, n_c-1)
            is_nz = False
            for (aa,bb,cc) in nz:
                if a==aa and b==bb and c==cc:
                    is_nz = True
                    break
            if not is_nz:
                s = 0.0
                for r in range(rank):
                    s += U[a,r]*V[b,r]*W[c,r]
                e += s*s
        return e

    # Try dropping each of the 17 columns and do a small local search
    for drop in range(17):
        if time.time() > deadline - 15:
            break
        U0, V0, W0 = _make_rank17()
        U = np.delete(U0, drop, axis=1)
        V = np.delete(V0, drop, axis=1)
        W = np.delete(W0, drop, axis=1)

        best_err = err(U, V, W)
        # 150 iterations of coordinate moves
        for it in range(150):
            if time.time() > deadline - 10:
                break
            which = random.randint(0, 2)
            if which == 0:
                a = random.randint(0, n_a-1)
                r = random.randint(0, rank-1)
                old = U[a,r]
            elif which == 1:
                b = random.randint(0, n_b-1)
                r = random.randint(0, rank-1)
                old = V[b,r]
            else:
                c = random.randint(0, n_c-1)
                r = random.randint(0, rank-1)
                old = W[c,r]

            # candidate values: small grid around old plus fixed small set
            cands = set()
            cands.update([old + d for d in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)])
            cands.update([-1.0, -0.5, 0.0, 0.5, 1.0])

            best_val = old
            best_local = best_err
            for nv in cands:
                if which == 0:
                    U[a,r] = nv
                elif which == 1:
                    V[b,r] = nv
                else:
                    W[c,r] = nv
                e = err(U, V, W)
                if e < best_local - 1e-12:
                    best_local = e
                    best_val = nv
                if which == 0:
                    U[a,r] = old
                elif which == 1:
                    V[b,r] = old
                else:
                    W[c,r] = old

            if which == 0:
                U[a,r] = best_val
            elif which == 1:
                V[b,r] = best_val
            else:
                W[c,r] = best_val
            best_err = best_local
            if best_err < 1e-10:
                break

        if best_err < 1e-10:
            return U, V, W
    return None

def construct():
    start = time.time()
    deadline = start + 115

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # First try a bounded search for rank 16
    res16 = _try_rank16(deadline)
    if res16 is not None:
        U, V, W = res16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final schoolbook fallback
    rank_fb = n*m*p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
