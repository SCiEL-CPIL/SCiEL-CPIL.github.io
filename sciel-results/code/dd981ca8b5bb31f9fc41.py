import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    I, J, K = T.shape
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0)

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=42):
    """Alternating Least Squares for CP decomposition.
    Returns (U, V, W) or None if not exact.
    """
    I, J, K = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(I, rank)
    V = rng.randn(J, rank)
    W = rng.randn(K, rank)
    
    # Normalize columns to prevent scale drift
    for _ in range(3):
        norms = np.linalg.norm(U, axis=0)
        norms[norms < 1e-12] = 1.0
        U = U / norms
        norms = np.linalg.norm(V, axis=0)
        norms[norms < 1e-12] = 1.0
        V = V / norms
        norms = np.linalg.norm(W, axis=0)
        norms[norms < 1e-12] = 1.0
        W = W / norms
    
    prev_residual = np.inf
    for it in range(max_iter):
        # Update U: for each i, T[i,:,:] = sum_c U[i,c] * (V ⊙ W)[c,:]
        # T_i = U_i * (V ⊙ W)^T
        # So U_i = T_i * (V ⊙ W)^T * ((V ⊙ W) * (V ⊙ W)^T)^{-1}
        # More efficiently: solve (V ⊙ W)^T (V ⊙ W) * U_i^T = (V ⊙ W)^T T_i^T
        
        # Khatri-Rao product V ⊙ W: shape (J*K, rank)
        VW = np.kron(V, W)  # This is not exactly Khatri-Rao, let's do it properly
        
        # Khatri-Rao: column c is V[:,c] ⊗ W[:,c]
        VW = np.zeros((J * K, rank))
        for c in range(rank):
            VW[:, c] = np.kron(V[:, c], W[:, c])
        
        # Solve for U
        # T_i = U_i @ VW.T  =>  U_i = T_i @ VW.T @ inv(VW @ VW.T)
        VW_VWt = VW.T @ VW  # (rank, rank)
        try:
            inv_VW_VWt = np.linalg.inv(VW_VWt)
        except np.linalg.LinAlgError:
            inv_VW_VWt = np.linalg.pinv(VW_VWt)
        
        # U_i = T[i,:,:].reshape(J*K) @ VW.T @ inv_VW_VWt
        T_flat = T.reshape(I, J * K)
        U = T_flat @ VW.T @ inv_VW_VWt
        
        # Update V
        UW = np.zeros((I * K, rank))
        for c in range(rank):
            UW[:, c] = np.kron(U[:, c], W[:, c])
        
        UW_UWt = UW.T @ UW
        try:
            inv_UW_UWt = np.linalg.inv(UW_UWt)
        except np.linalg.LinAlgError:
            inv_UW_UWt = np.linalg.pinv(UW_UWt)
        
        # T_j = V_j @ UW.T  =>  V_j = T_j @ UW.T @ inv(UW @ UW.T)
        # T reshaped to (J, I*K)
        T_J = np.zeros((J, I * K))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    T_J[j, i * K + k] = T[i, j, k]
        V = T_J @ UW.T @ inv_UW_UWt
        
        # Update W
        UV = np.zeros((I * J, rank))
        for c in range(rank):
            UV[:, c] = np.kron(U[:, c], V[:, c])
        
        UV_UVt = UV.T @ UV
        try:
            inv_UV_UVt = np.linalg.inv(UV_UVt)
        except np.linalg.LinAlgError:
            inv_UV_UVt = np.linalg.pinv(UV_UVt)
        
        # T_k = W_k @ UV.T  =>  W_k = T_k @ UV.T @ inv(UV @ UV.T)
        T_K = np.zeros((K, I * J))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    T_K[k, i * J + j] = T[i, j, k]
        W = T_K @ UV.T @ inv_UV_UVt
        
        # Check convergence
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        residual = np.linalg.norm(reconstructed - T)
        if residual < tol:
            break
        if abs(prev_residual - residual) < tol * 0.1:
            break
        prev_residual = residual
    
    # Final check
    if _verify(U, V, W, T, tol=1e-8):
        return U, V, W
    return None

def _try_strassen_block(n, m, p):
    """Try to build a decomposition using Strassen-like block structure.
    For 2x4x6, we can't directly use 2x2x2 Strassen, but we can try
    to find a structured low-rank decomposition.
    
    Returns (U, V, W) or None.
    """
    # For n=2, m=4, p=6, let's try rank 40-47 with structured initialization
    # based on the idea that we might share some multiplications.
    
    # Actually, let's just try random ALS for various ranks
    T = _build_tensor(n, m, p)
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # Build schoolbook as fallback
    U_sb = np.zeros((n * m, best_rank))
    V_sb = np.zeros((m * p, best_rank))
    W_sb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 40 to 47
    for rank in range(40, 48):
        for seed in range(10):
            result = _als_decompose(T, rank, max_iter=100, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                # Round to nearest 0.5 if close
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r, V_r, W_r, T, tol=1e-8):
                    return U_r, V_r, W_r
                # Use as is if exact enough
                if _verify(U, V, W, T, tol=1e-8):
                    return U, V, W
    
    return U_sb, V_sb, W_sb

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find a lower rank using ALS
    if time.time() < deadline - 20:
        T = _build_tensor(n, m, p)
        # Try ranks from 40 to 47
        for rank in range(40, 48):
            if time.time() > deadline - 10:
                break
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                result = _als_decompose(T, rank, max_iter=50, tol=1e-10, seed=seed)
                if result is not None:
                    U, V, W = result
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify(U_r, V_r, W_r, T, tol=1e-8):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        break
                    elif _verify(U, V, W, T, tol=1e-8):
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break
            if best_rank < rank_schoolbook:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
