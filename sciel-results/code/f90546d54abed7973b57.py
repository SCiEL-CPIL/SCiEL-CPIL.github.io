"""Rank-17 decomposition for 2x3x3 matrix multiplication with targeted rank-16 search."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            ja2, kb = divmod(j, P)
            if ja != ja2:
                continue
            for k in range(W_DIM):
                kb2, ic = divmod(k, N)
                if ia == ic and kb == kb2:
                    T[i, j, k] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    # Strassen columns 0..6
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    # Schoolbook columns 7..16
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _als_solve(U, V, W, T, max_iters=200, deadline=None, tol=1e-12):
    """Alternating least squares to refine U, V, W to match T."""
    rank = U.shape[1]
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Solve for U: T[i,j,k] = sum_c U[i,c] * (V[j,c]*W[k,c])
        # For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
        # Reshape: T0[i, j, k] -> for fixed i, we need U[i,:] such that
        # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
        # This is: for each i, T[i] = sum_c U[i,c] * outer(V[:,c], W[:,c])
        # Vectorize: T0[i, j*k + l] = sum_c U[i,c] * V[j,c] * W[l,c]
        # Build matrix A_U where A_U[i, c, j, l] = V[j,c]*W[l,c]
        # Actually: for each i, we solve: sum_c U[i,c] * (V[:,c] ⊗ W[:,c]) = T[i,:,:].ravel()
        # So: B_U[c, j, l] = V[j,c] * W[l,c], shape (rank, V_DIM, W_DIM)
        # T_flat[i, j, l] = T[i, j, l]
        # We need: for each i, T[i].ravel() = sum_c U[i,c] * B_U[c].ravel()
        # So: T_flat (U_DIM, V_DIM*W_DIM) = U (U_DIM, rank) @ B_U_flat (rank, V_DIM*W_DIM)
        B_U = np.einsum('jc,lc->jcl', V, W).reshape(rank, V_DIM * W_DIM)
        T_flat = T.reshape(U_DIM, V_DIM * W_DIM)
        # U = T_flat @ B_U^T (B_U^T @ B_U)^{-1}
        try:
            U_new = np.linalg.lstsq(B_U.T, T_flat.T, rcond=None)[0].T
        except:
            break
        # Solve for V: T[i,j,k] = sum_c (U[i,c]*W[k,c]) * V[j,c]
        B_V = np.einsum('ic,kc->ikc', U_new, W).reshape(rank, U_DIM * W_DIM)
        T_flat_v = T.reshape(V_DIM, U_DIM * W_DIM).T  # (U_DIM*W_DIM, V_DIM)
        try:
            V_new = np.linalg.lstsq(B_V.T, T_flat_v.T, rcond=None)[0].T
        except:
            break
        # Solve for W: T[i,j,k] = sum_c (U[i,c]*V[j,c]) * W[k,c]
        B_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(rank, U_DIM * V_DIM)
        T_flat_w = T.reshape(W_DIM, U_DIM * V_DIM).T  # (U_DIM*V_DIM, W_DIM)
        try:
            W_new = np.linalg.lstsq(B_W.T, T_flat_w.T, rcond=None)[0].T
        except:
            break
        U, V, W = U_new, V_new, W_new
        # Check convergence
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    return U, V, W, err


def _try_rank16(deadline):
    """Try to find rank-16 by zeroing each column of rank-17 and repairing with ALS."""
    T = _build_tensor()
    U17, V17, W17 = _rank17()
    
    # Try zeroing each of the 17 columns
    for col_to_zero in range(17):
        if time.time() > deadline:
            break
        # Create rank-16 matrices by removing column col_to_zero
        cols = [c for c in range(17) if c != col_to_zero]
        U16 = U17[:, cols].copy()
        V16 = V17[:, cols].copy()
        W16 = W17[:, cols].copy()
        
        # Add small random perturbation to help escape local minima
        rng = np.random.RandomState(42 + col_to_zero)
        U16 += 0.01 * rng.randn(U_DIM, 16)
        V16 += 0.01 * rng.randn(V_DIM, 16)
        W16 += 0.01 * rng.randn(W_DIM, 16)
        
        # Run ALS to try to recover
        U_new, V_new, W_new, err = _als_solve(U16, V16, W16, T, max_iters=300, deadline=deadline)
        
        if err < 1e-9:
            # Verify
            if _validate(U_new, V_new, W_new, T):
                return U_new, V_new, W_new
    
    # Also try pure random initialization with ALS
    for trial in range(20):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(1000 + trial)
        U = rng.randn(U_DIM, 16)
        V = rng.randn(V_DIM, 16)
        W = rng.randn(W_DIM, 16)
        
        U_new, V_new, W_new, err = _als_solve(U, V, W, T, max_iters=300, deadline=deadline)
        
        if err < 1e-9:
            if _validate(U_new, V_new, W_new, T):
                return U_new, V_new, W_new
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Try to find rank-16
    result = _try_rank16(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-17
    U, V, W = _rank17()
    T = _build_tensor()
    if not _validate(U, V, W, T):
        # Schoolbook fallback
        rank_fb = N * M * P
        U_fb = np.zeros((U_DIM, rank_fb))
        V_fb = np.zeros((V_DIM, rank_fb))
        W_fb = np.zeros((W_DIM, rank_fb))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_fb[i*M+j, col] = 1
                    V_fb[j*P+k, col] = 1
                    W_fb[k*N+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
