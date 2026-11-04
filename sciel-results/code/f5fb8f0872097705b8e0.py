"""Block-structured rank-17 decomposition for 2x3 * 3x3 matmul,
with structured search for rank-16 using row-block sharing."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    """Full matmul tensor T[i,j,k]=1 iff (i//M)==(k%N) and (i%M)==(j//P)."""
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


def _validate(U, V, W, T, atol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    """Valid rank-17 decomposition: Strassen(2x2x2) for (j in {0,1}, k in {0,1})
    plus schoolbook for the remaining 10 terms."""
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


def _structured_rank16_search(deadline):
    """
    Structured search for rank-16 using the observation that the 2x3x3 tensor
    consists of two 1x3x3 blocks (one for each row of the 2x3 matrix).
    
    Strategy: Assume that most rank-1 terms are shared between the two blocks,
    with the second block being a linear combination of the first block's terms
    plus a small number of additional terms.
    
    We parameterize:
    - U[0:3, :] = A (first row block, 3x16)
    - U[3:6, :] = B (second row block, 3x16)
    - V and W are 9x16 and 6x16 respectively
    
    The constraint is that the tensor factors correctly for both blocks.
    We use a gradient-free local search starting from the rank-17 solution
    with one column merged.
    """
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16
    
    # Start from rank-17 and try to merge pairs of columns
    U17, V17, W17 = _rank17()
    
    best_err = np.inf
    best_params = None
    
    for trial in range(30):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(42 + trial)
        
        # Initialize by taking rank-17 and zeroing out one column,
        # then adding small perturbations to encourage merging
        col_to_drop = trial % 17
        U = np.zeros((U_DIM, rank))
        V = np.zeros((V_DIM, rank))
        W = np.zeros((W_DIM, rank))
        
        col_idx = 0
        for c in range(17):
            if c == col_to_drop:
                continue
            U[:, col_idx] = U17[:, c]
            V[:, col_idx] = V17[:, c]
            W[:, col_idx] = W17[:, c]
            col_idx += 1
        
        # Add small random perturbations to all entries
        U += rng.randn(U_DIM, rank) * 0.01
        V += rng.randn(V_DIM, rank) * 0.01
        W += rng.randn(W_DIM, rank) * 0.01
        
        # Alternating least squares with convergence check
        prev_err = np.inf
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Solve for U
            B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
            U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
            
            # Solve for V
            B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
            V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
            
            # Solve for W
            B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
            W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
            
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(recon - T)
            
            if err < 1e-10:
                return U, V, W
            if err < best_err:
                best_err = err
                best_params = (U.copy(), V.copy(), W.copy())
            
            if abs(prev_err - err) < 1e-14 * max(1.0, prev_err):
                break
            prev_err = err
    
    # Try a second phase: start from random initializations with structure
    for trial in range(20):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(1000 + trial)
        
        # Initialize with a bias towards the known structure
        U = rng.randn(U_DIM, rank) * 0.3
        V = rng.randn(V_DIM, rank) * 0.3
        W = rng.randn(W_DIM, rank) * 0.3
        
        # Add structured components based on nonzero tensor entries
        nz = np.argwhere(T != 0)
        for _ in range(20):
            i, j, k = nz[rng.randint(len(nz))]
            c = rng.randint(rank)
            U[i, c] += rng.randn()
            V[j, c] += rng.randn()
            W[k, c] += rng.randn()
        
        prev_err = np.inf
        for it in range(100):
            if time.time() > deadline:
                break
            
            B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
            U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
            
            B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
            V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
            
            B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
            W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
            
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(recon - T)
            
            if err < 1e-10:
                return U, V, W
            if err < best_err:
                best_err = err
                best_params = (U.copy(), V.copy(), W.copy())
            
            if abs(prev_err - err) < 1e-14 * max(1.0, prev_err):
                break
            prev_err = err
    
    # If we found a near-solution, try to polish it
    if best_params is not None and best_err < 1e-4:
        U, V, W = best_params
        # High-precision polish
        for it in range(50):
            if time.time() > deadline:
                break
            B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
            U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
            B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
            V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
            B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
            W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-10:
                return U, V, W
    
    return None


def construct():
    start = time.time()
    deadline = start + 130  # Safety margin before 150s kill

    # Baseline: valid rank-17 (verified).
    U17, V17, W17 = _rank17()
    T = _build_tensor()
    base = (U17, V17, W17)
    if not _validate(U17, V17, W17, T):
        # Fallback to guaranteed-correct schoolbook rank-18.
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        base = (U, V, W)

    # Attempt rank-16 with structured search
    try:
        r16 = _structured_rank16_search(deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
