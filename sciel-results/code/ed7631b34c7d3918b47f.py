"""Improved rank-44 decomposition for 3x4 by 4x4 matrix multiplication.

The design keeps the Strassen+schoolbook block structure (rank 44) but:
1. Uses a cleaner, verified construction.
2. Attempts targeted rank reduction by checking for linear dependencies
   among the 48 tensor slices, with a time-bounded search.
"""
import time
import numpy as np

N, M, P = 3, 4, 4
TENSOR_DIM = N * M * P  # 48


def _verify(U, V, W, tol=1e-9):
    """Verify that sum_r U_r ⊗ V_r ⊗ W_r = ones(N,M,P)."""
    rank = U.shape[1]
    T = np.zeros((N, M, P), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(N, M)
        v = V[:, r].reshape(M, P)
        w = W[:, r].reshape(P, N)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)


def _build_rank44():
    """Build a rank-44 decomposition:
    - 4 Strassen blocks (2x2x2) for rows 0-1, covering cols 0-1 and 2-3,
      with B-blocks (0,0),(2,0),(0,2),(2,2)
    - 4 schoolbook blocks (1x2x2) for row 2, same B-blocks
    Total: 4*7 + 4*4 = 28 + 16 = 44
    """
    rank = 44
    U = np.zeros((N * M, rank), dtype=np.float64)
    V = np.zeros((M * P, rank), dtype=np.float64)
    W = np.zeros((P * N, rank), dtype=np.float64)
    col = 0

    def strassen(a_r, a_c, b_r, b_c):
        """Add 7 Strassen terms for A[a_r:a_r+2, a_c:a_c+2] * B[b_r:b_r+2, b_c:b_c+2]."""
        nonlocal col
        # m1 = (A11 - A22)(B21 + B22)
        U[a_r * M + a_c, col] = 1
        U[(a_r + 1) * M + (a_c + 1), col] = -1
        V[(b_r) * P + (b_c + 1), col] = 1
        V[(b_r + 1) * P + (b_c + 1), col] = 1
        W[(b_c + 1) * N + a_r, col] = 1
        W[(b_c + 1) * N + (a_r + 1), col] = 1
        col += 1
        # m2 = (A21 + A22) B11
        U[(a_r + 1) * M + a_c, col] = 1
        U[(a_r + 1) * M + (a_c + 1), col] = 1
        V[(b_r) * P + b_c, col] = 1
        W[b_c * N + a_r, col] = 1
        W[b_c * N + (a_r + 1), col] = 1
        col += 1
        # m3 = A11 (B12 - B22)
        U[a_r * M + a_c, col] = 1
        V[(b_r) * P + (b_c + 1), col] = 1
        V[(b_r + 1) * P + (b_c + 1), col] = -1
        W[(b_c + 1) * N + a_r, col] = 1
        col += 1
        # m4 = A22 (B21 - B11)
        U[(a_r + 1) * M + (a_c + 1), col] = 1
        V[(b_r + 1) * P + (b_c + 1), col] = 1
        V[(b_r) * P + b_c, col] = -1
        W[(b_c + 1) * N + (a_r + 1), col] = 1
        col += 1
        # m5 = (A11 + A21) B12
        U[a_r * M + a_c, col] = 1
        U[(a_r + 1) * M + a_c, col] = 1
        V[(b_r) * P + (b_c + 1), col] = 1
        W[(b_c + 1) * N + a_r, col] = 1
        W[(b_c + 1) * N + (a_r + 1), col] = 1
        col += 1
        # m6 = (A22 - A11) (B11 + B21)
        U[(a_r + 1) * M + (a_c + 1), col] = 1
        U[a_r * M + a_c, col] = -1
        V[(b_r) * P + b_c, col] = 1
        V[(b_r + 1) * P + (b_c + 1), col] = 1
        W[b_c * N + (a_r + 1), col] = 1
        col += 1
        # m7 = (A11 - A22) (B22 + B11)  -- actually standard Strassen m7
        U[a_r * M + a_c, col] = 1
        U[(a_r + 1) * M + (a_c + 1), col] = -1
        V[(b_r + 1) * P + (b_c + 1), col] = 1
        V[(b_r) * P + b_c, col] = 1
        W[(b_c + 1) * N + (a_r + 1), col] = 1
        col += 1

    def schoolbook_1x2(a_r, a_c, b_r, b_c):
        """Add 4 schoolbook terms for A[a_r, a_c:a_c+2] * B[b_r:b_r+2, b_c:b_c+2]."""
        nonlocal col
        for dr in range(2):
            for dc in range(2):
                U[a_r * M + a_c + dc, col] = 1
                V[(b_r + dr) * P + (b_c + dc), col] = 1
                W[(b_c + dc) * N + a_r, col] = 1
                col += 1

    # Strassen blocks for rows 0-1
    strassen(0, 0, 0, 0)
    strassen(0, 0, 2, 0)
    strassen(0, 2, 0, 0)
    strassen(0, 2, 2, 0)

    # Schoolbook blocks for row 2
    schoolbook_1x2(2, 0, 0, 0)
    schoolbook_1x2(2, 0, 2, 0)
    schoolbook_1x2(2, 2, 0, 0)
    schoolbook_1x2(2, 2, 2, 0)

    assert col == rank, f"Expected {rank} terms, got {col}"
    return U, V, W


def _try_reduce_rank(U, V, W, deadline):
    """Try to find a lower-rank decomposition by checking dependencies.
    
    Strategy: For each pair of terms, check if one can be eliminated
    by adjusting the other. Also try removing a single term and solving
    for the deficit in the span of remaining terms.
    """
    rank = U.shape[1]
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    best_rank = rank
    best_U, best_V, best_W = U_cur, V_cur, W_cur

    def term_outer(r):
        """Flatten the outer product of term r."""
        u = U_cur[:, r].reshape(N, M)
        v = V_cur[:, r].reshape(M, P)
        w = W_cur[:, r].reshape(P, N)
        return np.einsum('ij,jk,ki->ijk', u, v, w).flatten()

    # Check if removing term r leaves the tensor representable
    # by the remaining terms (with adjusted coefficients)
    for r_try in range(rank):
        if time.time() > deadline - 5:
            break
        if best_rank <= rank - 1:
            # Already found a reduction, try to go further
            if best_rank <= 40:
                break

        # Remove term r_try
        U_new = np.delete(U_cur, r_try, axis=1)
        V_new = np.delete(V_cur, r_try, axis=1)
        W_new = np.delete(W_cur, r_try, axis=1)
        new_rank = rank - 1

        # The deficit is term_outer(r_try)
        deficit = term_outer(r_try)

        # Build matrix of remaining terms
        A_mat = np.zeros((TENSOR_DIM, new_rank), dtype=np.float64)
        for nr in range(new_rank):
            orig = nr if nr < r_try else nr + 1
            A_mat[:, nr] = term_outer(orig)

        # Solve A_mat @ delta = deficit
        try:
            delta, _, _, _ = np.linalg.lstsq(A_mat, deficit, rcond=1e-10)
            resid = np.linalg.norm(A_mat @ delta - deficit)
            if resid < 1e-8:
                # Check if delta entries are "nice" (half-integers or small rationals)
                # Scale by 2 and check integrality
                if np.allclose(delta * 2, np.round(delta * 2), atol=1e-6):
                    # Apply: new_term_r = old_term_r + delta_r * old_term_r
                    # Actually we need: sum_{r != r_try} (1 + delta_r) * T_r = ones
                    # Wait, the equation is: sum_{r != r_try} delta_r * T_r = T_{r_try}
                    # So the new decomposition uses coefficients (1+delta_r) for r != r_try?
                    # No: original is sum_all T_r = ones. If we remove T_{r_try},
                    # we need sum_{r != r_try} T_r + deficit = ones, where deficit = T_{r_try}.
                    # If deficit = sum_{r != r_try} delta_r T_r, then
                    # ones = sum_{r != r_try} (1 + delta_r) T_r.
                    # But that changes the U,V,W matrices.
                    
                    # Actually, the standard approach: the tensor is sum_r U_r⊗V_r⊗W_r.
                    # If we can write T_{r_try} = sum_{r != r_try} c_r T_r,
                    # then ones = sum_{r != r_try} (U_r⊗V_r⊗W_r) + sum_{r != r_try} c_r (U_r⊗V_r⊗W_r)
                    # = sum_{r != r_try} (1+c_r) (U_r⊗V_r⊗W_r).
                    # But (1+c_r)(U_r⊗V_r⊗W_r) = ((1+c_r)U_r)⊗V_r⊗W_r, which is valid.
                    
                    # However, we need the new U,V,W to have the same structure.
                    # Let's just scale U_new columns by (1+delta).
                    for nr in range(new_rank):
                        U_new[:, nr] *= (1 + delta[nr])
                    
                    # Verify
                    if _verify(U_new, V_new, W_new, tol=1e-6):
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        # Continue with the reduced decomposition
                        U_cur, V_cur, W_cur = U_new, V_new, W_new
                        rank = new_rank
                        break
        except:
            pass

    return best_U, best_V, best_W, best_rank


def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Build rank-44 decomposition
    U, V, W = _build_rank44()

    # Verify
    assert _verify(U, V, W, tol=1e-8), "Rank-44 construction failed verification"

    best_U, best_V, best_W = U, V, W
    best_rank = 44

    # Try to reduce rank
    if time.time() < deadline - 10:
        U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, deadline)
        if rank_red < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = rank_red

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
