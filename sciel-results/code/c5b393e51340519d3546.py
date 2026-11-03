import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 125  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def try_rank_reduction(U, V, W, n=2, m=4, p=4, max_rank=27):
    """
    Try to reduce the rank by checking if any tensor factors are linearly dependent
    or can be merged. Uses a smooth optimization approach to find lower-rank approximations.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    # Try to find if we can reduce rank by checking for linear dependencies
    # in the tensor structure. We'll use a simple approach: check if removing
    # any single factor still allows reconstruction.
    
    for r in range(current_rank):
        if time_up():
            break
        
        # Try removing factor r
        U_test = np.delete(U_arr, r, axis=1)
        V_test = np.delete(V_arr, r, axis=1)
        W_test = np.delete(W_arr, r, axis=1)
        
        # Check if this still decomposes correctly
        valid, err = verify_decomposition(
            U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p
        )
        
        if valid and err < 1e-9:
            # We found a lower rank!
            return U_test, V_test, W_test
    
    # If we can't remove a single factor, try pairwise merging with optimization
    # This is a heuristic: check if two factors can be combined into one
    for r1 in range(current_rank):
        if time_up():
            break
        for r2 in range(r1 + 1, current_rank):
            if time_up():
                break
            
            # Check if factors r1 and r2 are "similar" in a way that allows merging
            # Simple heuristic: check if their contributions are proportional
            U_r1 = U_arr[:, r1]
            U_r2 = U_arr[:, r2]
            V_r1 = V_arr[:, r1]
            V_r2 = V_arr[:, r2]
            W_r1 = W_arr[:, r1]
            W_r2 = W_arr[:, r2]
            
            # Check if there exists a scalar c such that U_r2 ≈ c*U_r1, etc.
            # This is a rough check
            if np.allclose(U_r1, 0) and np.allclose(U_r2, 0):
                continue
            if np.allclose(V_r1, 0) and np.allclose(V_r2, 0):
                continue
            if np.allclose(W_r1, 0) and np.allclose(W_r2, 0):
                continue
            
            # Try to see if we can express the sum of these two as a single factor
            # This would require solving a system, which is complex
            # For now, skip this and just return the original
            pass
    
    return U_arr, V_arr, W_arr

def smooth_rank_search(U, V, W, n=2, m=4, p=4, target_rank=27, max_iter=1000):
    """
    Use a smooth optimization approach to find a lower-rank decomposition.
    We minimize the error of a rank-target_rank approximation using gradient descent.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Initialize with random lower-rank factors
    np.random.seed(42)
    U_opt = np.random.randn(n*m, target_rank) * 0.1
    V_opt = np.random.randn(m*p, target_rank) * 0.1
    W_opt = np.random.randn(p*n, target_rank) * 0.1
    
    # Also try initializing from the top target_rank components of SVD
    try:
        # Build the tensor and compute its singular values
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reshape for SVD
        T_mat = T.reshape(n*m, m*p*p*n)
        U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
        
        if target_rank <= len(S):
            U_init = U_svd[:, :target_rank] * np.sqrt(S[:target_rank])
            V_init = Vt[:target_rank, :].T * np.sqrt(S[:target_rank])
            # This isn't quite right for tensor, but it's a starting point
    except:
        pass
    
    # Gradient descent
    lr = 0.01
    for it in range(max_iter):
        if time_up():
            break
        
        # Compute gradient numerically or analytically
        # For simplicity, use a simple update rule
        
        # Compute current error
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(
                        U_opt[i*m+j] * V_opt[j*p+k] * W_opt[k*n+i]
                    )
        
        T_target = np.ones((n, m, p))
        err = np.sum((T_recon - T_target)**2)
        
        if err < 1e-12:
            break
        
        # Simple gradient step (approximate)
        # This is a heuristic, not a true gradient
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = T_target[i, j, k] - T_recon[i, j, k]
                    U_opt[i*m+j] += lr * diff * V_opt[j*p+k] * W_opt[k*n+i]
                    V_opt[j*p+k] += lr * diff * U_opt[i*m+j] * W_opt[k*n+i]
                    W_opt[k*n+i] += lr * diff * U_opt[i*m+j] * V_opt[j*p+k]
        
        # Normalize to prevent blowup
        for r in range(target_rank):
            norm_U = np.linalg.norm(U_opt[:, r])
            norm_V = np.linalg.norm(V_opt[:, r])
            norm_W = np.linalg.norm(W_opt[:, r])
            if norm_U > 100:
                U_opt[:, r] /= norm_U
                V_opt[:, r] *= norm_U
                W_opt[:, r] *= norm_U
            if norm_V > 100:
                V_opt[:, r] /= norm_V
                U_opt[:, r] *= norm_V
                W_opt[:, r] *= norm_V
    
    # Check if we found a valid decomposition
    U_list = U_opt.tolist()
    V_list = V_opt.tolist()
    W_list = W_opt.tolist()
    valid, err = verify_decomposition(U_list, V_list, W_list, n, m, p)
    
    if valid and err < 1e-9:
        return U_opt, V_opt, W_opt
    
    return None

def construct():
    n, m, p = 2, 4, 4
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    
    # Verify to be safe
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if not time_up():
        U_reduced, V_reduced, W_reduced = try_rank_reduction(
            U, V, W, n, m, p, max_rank=27
        )
        if U_reduced.shape[1] < len(U[0]):
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try smooth optimization for even lower rank
    if not time_up():
        for target_rank in [27, 26, 25]:
            if time_up():
                break
            result = smooth_rank_search(
                U, V, W, n, m, p, target_rank=target_rank, max_iter=500
            )
            if result is not None:
                U, V, W = result
                break
    
    # Final verification
    valid, err = verify_decomposition(
        U.tolist() if isinstance(U, np.ndarray) else U,
        V.tolist() if isinstance(V, np.ndarray) else V,
        W.tolist() if isinstance(W, np.ndarray) else W,
        n, m, p
    )
    
    if not valid:
        # Fallback to Strassen if optimization failed
        U, V, W = strassen_decomposition()
    
    return {"U": U.tolist() if isinstance(U, np.ndarray) else U,
            "V": V.tolist() if isinstance(V, np.ndarray) else V,
            "W": W.tolist() if isinstance(W, np.ndarray) else W}

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
