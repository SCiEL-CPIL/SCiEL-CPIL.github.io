import time
import numpy as np
from scipy.linalg import lstsq, solve
import random

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill


def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)


def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _snap_and_verify(U, V, W, T, tol=1e-6):
    """
    Try to snap U, V, W to nearest half-integers.
    If the snapped solution has error < tol, return the snapped solution.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2

    err = _tensor_error(U_snap, V_snap, W_snap, T)
    if err < tol:
        return U_snap, V_snap, W_snap, err
    return None


def _polish_solution(U, V, W, T, max_iter=100, tol=1e-14):
    """
    High-precision polish: alternate between solving for each matrix
    with very tight tolerance, then snap to half-integers.
    """
    n = U.shape[0] // 5 if U.shape[0] % 5 == 0 else 2
    m = U.shape[0] // n if n > 0 else 5
    p = W.shape[0] // n if n > 0 else 5

    err = np.inf
    for it in range(max_iter):
        # Update U
        A_u = np.einsum('jr,kr->jkr', V, W).reshape(V.shape[0] * W.shape[0], W.shape[1])
        b_u = T.reshape(U.shape[0], -1).T
        U_new, _, _, _ = lstsq(A_u, b_u, cond=None)
        U = U_new.T

        # Update V
        A_v = np.einsum('ir,kr->ikr', U, W).reshape(U.shape[0] * W.shape[0], W.shape[1])
        b_v = T.transpose(1, 0, 2).reshape(V.shape[0], -1).T
        V_new, _, _, _ = lstsq(A_v, b_v, cond=None)
        V = V_new.T

        # Update W
        A_w = np.einsum('ir,jr->ijr', U, V).reshape(U.shape[0] * V.shape[0], V.shape[1])
        b_w = T.transpose(2, 0, 1).reshape(W.shape[0], -1).T
        W_new, _, _, _ = lstsq(A_w, b_w, cond=None)
        W = W_new.T

        err = _tensor_error(U, V, W, T)
        if err < tol:
            break

    # Try snapping
    snapped = _snap_and_verify(U, V, W, T)
    if snapped:
        return snapped

    if err < 1e-10:
        return U, V, W, err

    return U, V, W, err


def _als_decompose(n, m, p, rank, T, seed=0, time_limit=5.0, max_iter=500):
    """
    Alternating Least Squares for tensor decomposition.
    Returns (U, V, W, error) or None.
    """
    rng = np.random.RandomState(seed)

    # Random initialization
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    # Normalize columns
    for mat in (U, V, W):
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    start_time = time.time()
    iterations = 0
    prev_err = np.inf
    patience = 0

    while iterations < max_iter:
        if time.time() - start_time > time_limit:
            break

        # Update U: for each i, solve (V W^T) U_i = T_i
        # T_i is shape (m*p, p*n), V W^T is (m*p, p*n, rank)
        # Reshape: T_flat[i] = T[i].reshape(-1), A = (V*W^T).reshape(m*p*p*n, rank)
        A_u = np.einsum('jr,kr->jkr', V, W).reshape(m * p * p * n, rank)
        b_u = T.reshape(n * m, -1).T  # (n*m, m*p*p*n)
        U_new, _, _, _ = lstsq(A_u, b_u, cond=None)
        U = U_new.T

        # Update V
        A_v = np.einsum('ir,kr->ikr', U, W).reshape(n * m * p * n, rank)
        b_v = T.transpose(1, 0, 2).reshape(m * p, -1).T
        V_new, _, _, _ = lstsq(A_v, b_v, cond=None)
        V = V_new.T

        # Update W
        A_w = np.einsum('ir,jr->ijr', U, V).reshape(n * m * m * p, rank)
        b_w = T.transpose(2, 0, 1).reshape(p * n, -1).T
        W_new, _, _, _ = lstsq(A_w, b_w, cond=None)
        W = W_new.T

        iterations += 1
        err = _tensor_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1

        if best_err < 1e-12:
            break
        if patience > 50:
            break
        if abs(prev_err - best_err) < 1e-15 and iterations > 20:
            break
        prev_err = best_err

    return best_U, best_V, best_W, best_err


def _try_rank_linear(n, m, p, rank, T, seed=0, time_limit=5.0):
    """
    Try to find rank-rank decomposition using linear system approach.
    Fix U and V, solve for W linearly.
    Returns (U, V, W, error) or None.
    """
    rng = np.random.RandomState(seed)

    # Random U and V
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)

    # Normalize
    norms_U = np.linalg.norm(U, axis=0, keepdims=True)
    norms_U[norms_U == 0] = 1.0
    U /= norms_U
    norms_V = np.linalg.norm(V, axis=0, keepdims=True)
    norms_V[norms_V == 0] = 1.0
    V /= norms_V

    # Build linear system for W:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For fixed U,V: sum_r (U[i,r]*V[j,r]) W[k,r] = T[i,j,k]
    # This is a linear system in W.
    # A[k, (i,j), r] = U[i,r] * V[j,r]
    # A is (p*n, n*m*m*p, rank)
    # Reshape: A_flat is (p*n, n*m*m*p*rank)
    # b is (p*n, n*m*m*p)

    A = np.einsum('ir,jr->ijkr', U, V)  # (n*m, m*p, rank) -> need to reshape
    # Actually: A[k, i, j, r] = U[i,r] * V[j,r]
    # We want: for each k, sum over (i,j,r): A[k,i,j,r] * W[k,r] ... 
    # Wait, W[k,r] is shared across all (i,j) for fixed k.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For fixed k: T[:, :, k] = sum_r (U[:,r] * V[:,r]) * W[k,r]
    # Let M_r = U[:,r] * V[:,r] (outer product, shape (n*m, m*p))
    # Then T[:,:,k] = sum_r M_r * W[k,r]
    # So for each k, we have: T[:,:,k].reshape(-1) = [M_0, M_1, ..., M_{rank-1}] @ W[k,:]
    # This is a (n*m*m*p) x rank linear system for W[k,:]

    # Build M: shape (n*m, m*p, rank)
    M = np.einsum('ir,jr->ijr', U, V)  # (n*m, m*p, rank)
    M_flat = M.reshape(n * m * m * p, rank)  # (n*m*m*p, rank)

    # T reshaped: T[i,j,k] -> for each k, T[:,:,k].reshape(-1)
    T_reshaped = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)  # (p*n, n*m*m*p)

    # Solve: M_flat @ W[k,:] = T_reshaped[k,:]
    # W is (p*n, rank)
    W, _, _, _ = lstsq(M_flat, T_reshaped.T, cond=None)
    W = W.T  # (p*n, rank)

    err = _tensor_error(U, V, W, T)

    # Now do ALS to refine
    start_time = time.time()
    iterations = 0
    max_iter = 200

    while iterations < max_iter:
        if time.time() - start_time > time_limit:
            break

        # Update U
        A_u = np.einsum('jr,kr->jkr', V, W).reshape(m * p * p * n, rank)
        b_u = T.reshape(n * m, -1).T
        U_new, _, _, _ = lstsq(A_u, b_u, cond=None)
        U = U_new.T

        # Update V
        A_v = np.einsum('ir,kr->ikr', U, W).reshape(n * m * p * n, rank)
        b_v = T.transpose(1, 0, 2).reshape(m * p, -1).T
        V_new, _, _, _ = lstsq(A_v, b_v, cond=None)
        V = V_new.T

        # Update W
        A_w = np.einsum('ir,jr->ijr', U, V).reshape(n * m * m * p, rank)
        b_w = T.transpose(2, 0, 1).reshape(p * n, -1).T
        W_new, _, _, _ = lstsq(A_w, b_w, cond=None)
        W = W_new.T

        iterations += 1
        err = _tensor_error(U, V, W, T)
        if err < 1e-12:
            break

    return U, V, W, err


def _structured_init(n, m, p, rank, seed=0):
    """
    Generate structured random initializations that might be closer to solutions.
    """
    rng = np.random.RandomState(seed)

    # Strategy 1: Block structure based on matrix multiplication
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Try to assign some structure
    for r in range(rank):
        # Random but with some sparsity
        if rng.random() < 0.5:
            # Dense random
            U[:, r] = rng.randn(n * m)
            V[:, r] = rng.randn(m * p)
            W[:, r] = rng.randn(p * n)
        else:
            # Sparse random
            U[:, r] = rng.randn(n * m)
            U[rng.randint(0, n * m, size=n * m // 2), r] = 0
            V[:, r] = rng.randn(m * p)
            V[rng.randint(0, m * p, size=m * p // 2), r] = 0
            W[:, r] = rng.randn(p * n)
            W[rng.randint(0, p * n, size=p * n // 2), r] = 0

    # Normalize
    for mat in (U, V, W):
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

    return U, V, W


def _try_rank(n, m, p, rank, T, seed=0, time_limit=5.0):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    rng = np.random.RandomState(seed)

    # Try multiple initialization strategies
    best_result = None
    best_err = np.inf

    # Strategy 1: Random initialization + ALS
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    for mat in (U, V, W):
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

    result = _als_decompose(n, m, p, rank, T, seed=seed, time_limit=time_limit * 0.4, max_iter=300)
    if result[3] < best_err:
        best_err = result[3]
        best_result = result

    # Strategy 2: Linear system approach (fix U,V, solve for W)
    U2 = rng.randn(n * m, rank)
    V2 = rng.randn(m * p, rank)
    for mat in (U2, V2):
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

    result2 = _try_rank_linear(n, m, p, rank, T, seed=seed + 1000, time_limit=time_limit * 0.4)
    if result2[3] < best_err:
        best_err = result2[3]
        best_result = result2

    # Strategy 3: Structured initialization
    U3, V3, W3 = _structured_init(n, m, p, rank, seed=seed + 2000)
    result3 = _als_decompose(n, m, p, rank, T, seed=seed + 3000, time_limit=time_limit * 0.3, max_iter=200)
    if result3[3] < best_err:
        best_err = result3[3]
        best_result = result3

    if best_result is None:
        return None

    U, V, W, err = best_result

    # Try polishing if we got reasonably close
    if err < 1e-4:
        polished = _polish_solution(U, V, W, T, max_iter=50, tol=1e-14)
        if polished[3] < err:
            U, V, W, err = polished

    # Try snapping
    snapped = _snap_and_verify(U, V, W, T)
    if snapped:
        return snapped

    # If we got very close, return it
    if err < 1e-8:
        return U, V, W, err

    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Start with schoolbook as fallback
    U, V, W = _schoolbook_solution(n, m, p)
    T = _build_tensor(n, m, p)

    # Known lower bound is 25.
    # We try from 25 upwards.
    candidate_ranks = list(range(25, 51))

    best_solution = (U, V, W, 50)

    for rank in candidate_ranks:
        elapsed = time.time() - start_time
        remaining = DEADLINE_SECONDS - elapsed
        if remaining < 3:
            break

        success = False
        # Allocate more time to lower ranks
        if rank <= 27:
            num_seeds = 20
            time_per_seed = max(2.0, remaining / (num_seeds * 3))
        elif rank <= 35:
            num_seeds = 12
            time_per_seed = max(1.5, remaining / (num_seeds * 3))
        else:
            num_seeds = 8
            time_per_seed = max(1.0, remaining / (num_seeds * 3))

        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE_SECONDS - 2:
                break

            try:
                res = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_per_seed)
                if res is not None:
                    U_new, V_new, W_new, err = res
                    # Verify
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-10:
                        U, V, W = U_new, V_new, W_new
                        success = True
                        break
            except Exception:
                continue

        if success:
            break

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
