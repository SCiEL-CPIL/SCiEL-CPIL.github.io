"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True


def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=100, tol=1e-10, seed=0):
    """Alternating Least Squares for tensor decomposition.
    T has shape (n*m, m*p, p*n). Find U(n*m x rank), V(m*p x rank), W(p*n x rank)
    such that T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    # Initialize
    U = rng.randn(dim1, rank)
    V = rng.randn(dim2, rank)
    W = rng.randn(dim3, rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 5.0:
            break
        
        # Update U: fix V, W
        # T_(1) is dim1 x (dim2*dim3)
        # T_(1)[i, j*dim3+k] = T[i,j,k]
        # We want: T_(1) = U @ (V ⊙ W)^T
        # where (V ⊙ W) is the Khatri-Rao product: column c is V[:,c] ⊗ W[:,c]
        V_kr_W = np.kron(V.T, W.T).T  # shape (dim2*dim3, rank) - actually let me be careful
        # V is (dim2, rank), W is (dim3, rank)
        # Khatri-Rao: for each c, column is V[:,c] ⊗ W[:,c] which has length dim2*dim3
        # np.kron(V[:,c], W[:,c]) gives this
        # More efficiently: V_kr_W = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        V_kr_W = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        
        T1 = T.reshape(dim1, -1)  # dim1 x (dim2*dim3)
        # Solve: T1 = U @ V_kr_W^T  =>  U = T1 @ V_kr_W^T @ (V_kr_W @ V_kr_W^T)^{-1}
        try:
            U_new, _, _, _ = np.linalg.lstsq(V_kr_W, T1.T, rcond=None)
            U = U_new.T  # back to (dim1, rank)
        except np.linalg.LinAlgError:
            pass
        
        # Update V: fix U, W
        # T_(2) is dim2 x (dim1*dim3)
        # T_(2)[j, i*dim3+k] = T[i,j,k]
        U_kr_W = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)  # dim2 x (dim1*dim3)
        try:
            V_new, _, _, _ = np.linalg.lstsq(U_kr_W, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W: fix U, V
        # T_(3) is dim3 x (dim1*dim2)
        # T_(3)[k, i*dim2+j] = T[i,j,k]
        U_kr_V = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)  # dim3 x (dim1*dim2)
        try:
            W_new, _, _, _ = np.linalg.lstsq(U_kr_V, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Compute loss
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((reconstructed - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
    
    if best_loss < 1e-8:
        # Round to clean values
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_rank(T, n, m, p, rank, deadline, max_trials=20):
    """Try multiple random initializations for a given rank."""
    for seed in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=80, seed=seed)
        if result is not None:
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    T = _build_tensor(n, m, p)
    
    best_result = None
    
    # Try ranks from 1 up to schoolbook
    for rank in range(1, n * m * p + 1):
        if time.time() > deadline - 15.0:
            break
        
        result = _try_rank(T, n, m, p, rank, deadline, max_trials=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return _schoolbook_decomposition(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
