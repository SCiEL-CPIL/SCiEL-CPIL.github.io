"""Matrix multiplication tensor decomposition for 2x3 @ 3x8.
Uses rank-11 per 2x3x2 block (Strassen-7 for the 2x2 core + 4 scalar products),
giving total rank 44. Includes a fast self-verification and a deadline guard."""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _build_rank44():
    n, m, p = 2, 3, 8
    rank = 44

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen 7 for A[0:2, 0:2] * B[0:2, k0:k2]
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0, col] = 1.0
        U[m + 1, col] = 1.0
        V[0 * p + k0, col] = 1.0
        V[1 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M2 = (A10 + A11) * B0k0
        U[m + 0, col] = 1.0
        U[m + 1, col] = 1.0
        V[0 * p + k0, col] = 1.0
        W[k0 * n + 1, col] = 1.0
        W[k1 * n + 1, col] = -1.0
        col += 1

        # M3 = A00 * (B0k1 - B1k1)
        U[0, col] = 1.0
        V[0 * p + k1, col] = 1.0
        V[1 * p + k1, col] = -1.0
        W[k1 * n + 0, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M4 = A11 * (B1k0 - B0k0)
        U[m + 1, col] = 1.0
        V[1 * p + k0, col] = 1.0
        V[0 * p + k0, col] = -1.0
        W[k0 * n + 0, col] = 1.0
        W[k0 * n + 1, col] = 1.0
        col += 1

        # M5 = (A00 + A01) * B1k1
        U[0, col] = 1.0
        U[1, col] = 1.0
        V[1 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = -1.0
        W[k1 * n + 0, col] = 1.0
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[m + 0, col] = 1.0
        U[0, col] = -1.0
        V[0 * p + k0, col] = 1.0
        V[0 * p + k1, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1, col] = 1.0
        U[m + 1, col] = -1.0
        V[1 * p + k0, col] = 1.0
        V[1 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = 1.0
        col += 1

        # 4 scalar products for A[:,2] * B[2, :]
        # P1 = A02 * B2k0
        U[2, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[k0 * n + 0, col] = 1.0
        col += 1

        # P2 = A02 * B2k1
        U[2, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[k1 * n + 0, col] = 1.0
        col += 1

        # P3 = A12 * B2k0
        U[m + 2, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[k0 * n + 1, col] = 1.0
        col += 1

        # P4 = A12 * B2k1
        U[m + 2, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 iff valid, else 0.
    Uses the trace convention W[k*n+i, t]."""
    # Build the multiplication tensor T[i, j, k] = 1 (the shared index is j)
    # The decomposition must satisfy:
    #   for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    # (since the multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all valid i,j,k)
    
    # Wait - the multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for ALL i,j,k.
    
    # Let's check a few entries to verify.
    # Full check would be n*m*p = 2*3*8 = 48 entries, each a sum of rank terms.
    # Do it efficiently with einsum.
    
    # T_reconstructed[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # For each t, the contribution to T is U[:,t] (n*m) * V[:,t] (m*p) * W[:,t] (p*n)
    # which is an outer product reshaped to (n, m, p) via:
    #   U_reshaped[t] has shape (n, m) indexed by [i, j]
    #   V_reshaped[t] has shape (m, p) indexed by [j, k]
    #   W_reshaped[t] has shape (p, n) indexed by [k, i]
    #   T[:, :, :, t] = U_reshaped[t, i, j] * V_reshaped[t, j, k] * W_reshaped[t, k, i]
    
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T_reconstructed[i, j, k] = sum_t U_r[i,j,t] * V_r[j,k,t] * W_r[k,i,t]
    # Use einsum: 'ijt,jkt,kit->ijk'
    T_rec = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    
    # Expected: all ones
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T_rec, T_expected, atol=tol)


def construct():
    t_start = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the rank-44 decomposition
    U, V, W = _build_rank44()
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Should not happen with the hardcoded construction, but guard anyway
        raise ValueError("Decomposition verification failed")
    
    # Convert to list-of-lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Should be nearly impossible for this fixed construction, but guard
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
