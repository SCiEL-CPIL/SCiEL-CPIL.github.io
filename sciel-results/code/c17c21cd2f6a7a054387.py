"""
Rank-minimal tensor decomposition for 3x4x8 matrix multiplication.

Approach:
  1. Construct the multiplication tensor T of shape (n*m, m*p, p*n).
     T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise.
  2. For each candidate rank R (starting from the theoretical lower bound 41),
     attempt to find matrices U (n*m x R), V (m*p x R), W (p*n x R) such that
     sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c).
  3. Use a two-phase approach:
     Phase 1: Fix W to be the identity-like structure (or random), solve for U,V
              via linear least squares on the flattened tensor.
     Phase 2: Iterate - fix two matrices, solve for the third, alternating.
     Phase 3: Round to nearest half-integers and verify.
  4. If no exact solution is found within time budget, fall back to schoolbook (rank 96).

The key insight for efficiency: for a fixed W, the problem becomes linear in U and V
jointly if we vectorize. Specifically, for each r, the contribution is
outer(U[:,r], V[:,r]) contracted with W[:,r]. We can set up a linear system.

Actually, the cleanest linear approach: fix W. Then for each pair (a,b), we need:
  sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]  for all c.
This is:  (U * diag(V[:,r]) ... ) -- not quite linear in both U and V simultaneously.

Better: fix V and W. Then for each (a, r):
  sum_b V[b,r] * ... no.

Let me restructure. The tensor equation is:
  T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

If we fix V and W, then for each a:
  T[a,:,c] = sum_r U[a,r] * V[:,r] * W[c,r]
  => T[a,:,c] = sum_r U[a,r] * (V[:,r] * W[c,r])
  
This is linear in U[a,:]. For each a and c:
  T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

So if V and W are fixed, U is determined by solving a linear system.
Similarly for V (fix U, W) and W (fix U, V).

We can do alternating linear solves:
  - Initialize U, V, W randomly (or with structure).
  - Fix V, W: solve for U. For each a, the equation over all (b,c) is:
    sum_r U[a,r] * (V[:,r] ⊙ W[c,r]) ... 
    Actually, for fixed a:
      T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    This is a linear system in U[a, :] with variables U[a,0..R-1].
    The "features" for variable r at position (b,c) is V[b,r]*W[c,r].
    So we have a matrix M_a of shape (m*p, R) where M_a[b,c,r] = V[b,r]*W[c,r].
    Then T[a] (flattened to m*p) = M_a @ U[a,:].
    Solve U[a,:] = lstsq(M_a, T[a]).
  - Similarly fix U, W: solve for V. M_b of shape (n*m, R) where M_b[a,c,r] = U[a,r]*W[c,r].
    T[:,b,:] flattened = M_b @ V[b,:].
  - Fix U, V: solve for W. M_c of shape (n*m, R) where M_c[a,b,r] = U[a,r]*V[b,r].
    T[:,:,c] flattened = M_c @ W[c,:].

This is the ALS (Alternating Least Squares) approach for tensor decomposition.

For exact solutions with integer/half-integer entries, we need the ALS to converge
to a solution that is (or can be rounded to) half-integers. This is not guaranteed,
but for small problems it may work.

Given the 120s time limit, we'll try ranks from 41 upward, with a few ALS iterations
per rank, and check for exactness (or near-exactness that rounds cleanly).
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _als_decompose(T, R, n, m, p, max_iter=50, tol=1e-10, seed=42):
    """
    Attempt to decompose tensor T into rank-R factors U, V, W using ALS.
    T has shape (n*m, m*p, p*n).
    Returns (U, V, W, error) or None if failed.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = T.shape

    # Initialize
    U = rng.randn(nm, R) * 0.5
    V = rng.randn(mp, R) * 0.5
    W = rng.randn(pn, R) * 0.5

    for it in range(max_iter):
        # Solve for U: fix V, W
        U_new = np.zeros_like(U)
        for a in range(nm):
            # M_a[b,c,r] = V[b,r] * W[c,r], shape (mp, pn, R) -> flatten to (mp*pn, R)
            M_a = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, R)
            t_a = T[a].reshape(mp * pn)
            # Solve M_a @ u_a = t_a
            u_a, _, _, _ = np.linalg.lstsq(M_a, t_a, rcond=None)
            U_new[a] = u_a

        # Solve for V: fix U, W
        V_new = np.zeros_like(V)
        for b in range(mp):
            # M_b[a,c,r] = U[a,r] * W[c,r], shape (nm, pn, R) -> flatten to (nm*pn, R)
            M_b = np.einsum('ar,cr->acr', U_new, W).reshape(nm * pn, R)
            t_b = T[:, b, :].reshape(nm * pn)
            v_b, _, _, _ = np.linalg.lstsq(M_b, t_b, rcond=None)
            V_new[b] = v_b

        # Solve for W: fix U, V
        W_new = np.zeros_like(W)
        for c in range(pn):
            # M_c[a,b,r] = U[a,r] * V[b,r], shape (nm, mp, R) -> flatten to (nm*mp, R)
            M_c = np.einsum('ar,br->abr', U_new, V_new).reshape(nm * mp, R)
            t_c = T[:, :, c].reshape(nm * mp)
            w_c, _, _, _ = np.linalg.lstsq(M_c, t_c, rcond=None)
            W_new[c] = w_c

        # Check convergence
        err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U_new, V_new, W_new))
        if err < tol:
            return U_new, V_new, W_new, err

        U, V, W = U_new, V_new, W_new

    # Final error
    err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
    return U, V, W, err


def _verify_and_round(U, V, W, T, n, m, p, tol=1e-6):
    """
    Try to round U, V, W to half-integers and verify exactness.
    Returns (U_r, V_r, W_r) as lists if valid, else None.
    """
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0

    # Verify
    max_err = 0.0
    for a in range(T.shape[0]):
        for b in range(T.shape[1]):
            for c in range(T.shape[2]):
                val = np.dot(U_r[a], V_r[b] * W_r[c])
                target = T[a, b, c]
                err = abs(val - target)
                if err > max_err:
                    max_err = err

    if max_err < tol:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None


def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    start_time = time.time()
    time_budget = 110  # leave 10s for overhead

    # Try ranks from theoretical lower bound upward
    lower_bound = 41
    max_rank = n * m * p  # schoolbook

    for R in range(lower_bound, max_rank + 1):
        if time.time() - start_time > time_budget:
            break

        # Try multiple random initializations
        found = False
        for seed in range(10):
            if time.time() - start_time > time_budget:
                break
            result = _als_decompose(T, R, n, m, p, max_iter=80, seed=seed)
            if result is not None:
                U, V, W, err = result
                if err < 1e-8:
                    rounded = _verify_and_round(U, V, W, T, n, m, p)
                    if rounded is not None:
                        return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
                    # If rounding fails, try with smaller tolerance or different rounding
                    # Try rounding to integers only
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    max_err = 0.0
                    for a in range(T.shape[0]):
                        for b in range(T.shape[1]):
                            for c in range(T.shape[2]):
                                val = np.dot(U_int[a], V_int[b] * W_int[c])
                                target = T[a, b, c]
                                err = abs(val - target)
                                if err > max_err:
                                    max_err = err
                    if max_err < 1e-6:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            found = True

    # Fallback: schoolbook decomposition (rank = n*m*p = 96)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
