"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Design (kept from seed family):
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: A[0:2,:] (2x2) * B[0:2,:] (2x10). Split the 10 output columns into
    five 2-wide blocks; apply classic rank-7 Strassen to each 2x2 x 2x2 block.
    Subtotal: 5 * 7 = 35.
  - Part 2: A[2,:] (2x1) * B[2,:] (1x10). Schoolbook: 2*1*10 = 20.
  Total rank: 55.

Refinements:
  1. Multi-trial exact verification.
  2. Graceful fallback to schoolbook (rank 60).
  3. Deadline guard.
  4. Attempt to reduce rank below 55 via ALS-based rank minimization,
     starting from the rank-55 solution and trying to project out terms.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before the 150 s kill


def _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234):
    """Verify the decomposition with random integer inputs (exact arithmetic).

    Returns (ok, max_err). Uses vectorized ops for speed.
    """
    rng = np.random.RandomState(seed)
    max_err = 0.0
    for _ in range(num_trials):
        A = rng.randint(-20, 21, size=(n, m)).astype(float)
        B = rng.randint(-20, 21, size=(m, p)).astype(float)
        C_direct = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat          # (rank,)
        v_sums = V.T @ B_flat          # (rank,)
        scalars = u_sums * v_sums      # (rank,)
        C_flat = W @ scalars           # (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # (n, p)

        err = float(np.max(np.abs(C_decomp - C_direct)))
        if err > max_err:
            max_err = err
        if err > 1e-9:
            return False, max_err
    return True, max_err


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _als_reduce_rank(U, V, W, target_rank, max_iters=200, tol=1e-12, deadline=None):
    """Try to reduce the rank of a decomposition to target_rank using ALS.
    
    Starts from the current (U,V,W) and attempts to find a valid rank-target_rank
    decomposition. Returns (success, U_new, V_new, W_new).
    """
    n, m, p = DIMENSIONS
    
    if U.shape[1] <= target_rank:
        return False, None, None, None
    
    U0 = U[:, :target_rank].copy()
    V0 = V[:, :target_rank].copy()
    W0 = W[:, :target_rank].copy()
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_err = np.inf
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U_mat = U0.reshape(n, m, target_rank)
        V_mat = V0.reshape(m, p, target_rank)
        W_mat = W0.reshape(p, n, target_rank)
        
        # Compute full sum S[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        S = np.zeros((n, m, p))
        for r2 in range(target_rank):
            term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
            S += term_3d
        
        T = _build_tensor(n, m, p)
        
        # Update W
        for r in range(target_rank):
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r
            num = np.einsum('ijk,ij,jk->ki', R, U_mat[:, :, r], V_mat[:, :, r])
            prod_3d = np.einsum('ij,jk->ijk', U_mat[:, :, r], V_mat[:, :, r])
            den = np.einsum('ijk,ijk->ki', prod_3d, prod_3d)
            mask = den > 1e-15
            W_mat[:, :, r][mask] = num[mask] / den[mask]
            W_mat[:, :, r][~mask] = 0.0
        
        # Update U
        for r in range(target_rank):
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r
            num = np.einsum('ijk,jk,ki->ij', R, V_mat[:, :, r], W_mat[:, :, r])
            prod_3d = np.einsum('jk,ki->ijk', V_mat[:, :, r], W_mat[:, :, r])
            den = np.einsum('ijk,ijk->ij', prod_3d, prod_3d)
            mask = den > 1e-15
            U_mat[:, :, r][mask] = num[mask] / den[mask]
            U_mat[:, :, r][~mask] = 0.0
        
        # Update V
        for r in range(target_rank):
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r
            num = np.einsum('ijk,ij,ki->jk', R, U_mat[:, :, r], W_mat[:, :, r])
            prod_3d = np.einsum('ij,ki->ijk', U_mat[:, :, r], W_mat[:, :, r])
            den = np.einsum('ijk,ijk->jk', prod_3d, prod_3d)
            mask = den > 1e-15
            V_mat[:, :, r][mask] = num[mask] / den[mask]
            V_mat[:, :, r][~mask] = 0.0
        
        U0 = U_mat.reshape(-1, target_rank)
        V0 = V_mat.reshape(-1, target_rank)
        W0 = W_mat.reshape(-1, target_rank)
        
        # Recompute S and check error
        S = np.zeros((n, m, p))
        for r2 in range(target_rank):
            term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
            S += term_3d
        err = np.max(np.abs(S - T))
        if err < best_err:
            best_err = err
            best_U = U0.copy()
            best_V = V0.copy()
            best_W = W0.copy()
        
        if err < tol:
            break
    
    return best_err < 1e-9, best_U, best_V, best_W


def _schoolbook(n, m, p):
    """Guaranteed-correct schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def _strassen_decomposition(n, m, p):
    """Build the rank-55 Strassen-on-2x2-blocks + schoolbook decomposition."""
    rank = 35 + 20  # 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (five 2x2x2 blocks)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        #     -> C[0,k0] += M1, C[1,k1] += M1
        U[0 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        #     -> C[0,k1] += M2, C[1,k1] -= M2
        U[1 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        #     -> C[0,k1] += M3, C[1,k1] += M3
        U[0 * 3 + 0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        V[1 * 10 + k1, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        #     -> C[0,k0] += M4, C[1,k0] += M4
        U[1 * 3 + 1, idx] = 1.0
        V[1 * 10 + k0, idx] = 1.0
        V[0 * 10 + k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        #     -> C[0,k0] -= M5, C[0,k1] += M5
        U[0 * 3 + 0, idx] = 1.0
        U[0 * 3 + 1, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        #     -> C[1,k1] += M6
        U[1 * 3 + 0, idx] = 1.0
        U[0 * 3 + 0, idx] = -1.0
        V[0 * 10 + k0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        #     -> C[0,k0] += M7
        U[0 * 3 + 1, idx] = 1.0
        U[1 * 3 + 1, idx] = -1.0
        V[1 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Build the rank-55 decomposition.
    U, V, W = _strassen_decomposition(n, m, p)

    # Rigorous verification with several seeds; fall back to schoolbook on failure.
    ok, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234)
    if not ok:
        ok2, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=9999)
        if not ok2:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction via ALS (only if time permits)
    if time.time() - start_time < 100:
        for target in [54, 53, 52]:
            if time.time() - start_time > 120:
                break
            success, U_new, V_new, W_new = _als_reduce_rank(
                U, V, W, target, max_iters=50, deadline=start_time + 125
            )
            if success:
                ok, _ = _verify_decomposition(U_new, V_new, W_new, n, m, p, num_trials=8, seed=5555)
                if ok:
                    U, V, W = U_new, V_new, W_new
                    break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    A_flat = A.flatten()
    B_flat = B.flatten()
    u_sums = U.T @ A_flat
    v_sums = V.T @ B_flat
    scalars = u_sums * v_sums
    C_flat = W @ scalars
    C_decomp = C_flat.reshape(p, n).T

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
