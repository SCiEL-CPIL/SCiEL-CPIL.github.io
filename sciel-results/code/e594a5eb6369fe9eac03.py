import numpy as np
import time

# Dimensions: n=2, m=3, p=6
# U shape: (n*m, rank) = (6, rank)
# V shape: (m*p, rank) = (18, rank)
# W shape: (p*n, rank) = (12, rank)
# Tensor T[i, j, k] = 1 if output index is (k*n+i) and input indices are (i*m+j) and (j*p+k)
# Wait, the problem statement says:
# "output factor follows the official trace convention W[k*n+i,r]"
# Standard matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
# C[i, k] = sum_j A[i, j] B[j, k]
# The tensor T has indices (i, j, k).
# U corresponds to A entries: index i*m + j
# V corresponds to B entries: index j*p + k
# W corresponds to C entries: index k*n + i (as per prompt "W[k*n+i,r]")

n, m, p = 2, 3, 6
N_U = n * m  # 6
N_V = m * p  # 18
N_W = p * n  # 12

def _build_tensor_slices():
    """
    Pre-compute the sparse structure of the tensor to allow fast ALS updates.
    For each rank component r, we have vectors u_r, v_r, w_r.
    The constraint is that for all (i, j, k), sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] == T[i,j,k].
    
    In ALS, when optimizing U (fixing V, W), for each row i_u in U, we need to satisfy:
    sum_k w_r[k*n+i] * (sum_j ... ) ... this is getting complicated to vectorize directly per row.
    
    Alternative view:
    The tensor T can be sliced along one dimension.
    Let's slice along the U dimension (rows of A).
    For a fixed row index `iu` in U (which corresponds to i in A, but U is flattened i*m+j),
    actually it's easier to think of the tensor as a 3D array T[i, j, k].
    
    Let's reshape U, V, W into 3D arrays for clarity during derivation, then flatten for output.
    U_3d[i, j, r]
    V_3d[j, k, r]
    W_3d[k, i, r]  <-- Note the order: k*n+i means index is k*n+i. So W_3d[k, i, r]
    
    The equation is:
    T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    
    We can rearrange this for ALS.
    To update U_3d (fixing V, W):
    For each (i, j), we have a vector over k:
    T[i, j, :] = sum_r U_3d[i, j, r] * (V_3d[j, :, r] * W_3d[:, i, r])
    
    Let M_u[i, j, k, r] = V_3d[j, k, r] * W_3d[k, i, r]
    Then T[i, j, k] = sum_r U_3d[i, j, r] * M_u[i, j, k, r]
    
    This is a linear system for U_3d[i, j, :] for each (i, j).
    Specifically, for a fixed (i, j), let u_vec be U_3d[i, j, :].
    We want min || T[i, j, :] - M_u[i, j, :, :] @ u_vec ||^2
    where M_u[i, j, :, :] is a matrix of shape (p, rank).
    The rows of this matrix are indexed by k, columns by r.
    Entry (k, r) is V_3d[j, k, r] * W_3d[k, i, r].
    
    Similarly for V and W.
    """
    pass

def _als_decompose(rank, max_iters=200, tol=1e-9, seed=42):
    """
    Perform ALS to find a rank decomposition.
    Returns U, V, W as flat lists of lists if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W as 3D arrays
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    U = rng.randn(n, m, rank) * 0.1
    V = rng.randn(m, p, rank) * 0.1
    W = rng.randn(p, n, rank) * 0.1
    
    # Pre-compute the target tensor T[i, j, k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    prev_err = np.inf
    
    for it in range(max_iters):
        # 1. Update U
        # For each i, j:
        # Solve (M_u[i,j].T @ M_u[i,j]) u = M_u[i,j].T @ T[i,j,:]
        # M_u[i,j] is shape (p, rank). Entry (k, r) = V[j, k, r] * W[k, i, r]
        
        for i in range(n):
            for j in range(m):
                # Construct M_u[i,j]
                # V[j, :, :] is (p, rank)
                # W[:, i, :] is (p, rank)
                # M_u[i,j][k, r] = V[j, k, r] * W[k, i, r]
                M_u = V[j, :, :] * W[:, i, :] # Broadcasting: (p, rank) * (p, rank) -> (p, rank)
                
                # Solve least squares
                # A = M_u.T @ M_u (rank x rank)
                # b = M_u.T @ T[i, j, :] (rank x 1)
                A = M_u.T @ M_u
                b = M_u.T @ T[i, j, :]
                
                # Add small regularization for stability
                A += 1e-10 * np.eye(rank)
                
                try:
                    U[i, j, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    U[i, j, :] = np.linalg.lstsq(M_u, T[i, j, :], rcond=None)[0]
                    
        # 2. Update V
        # For each j, k:
        # T[:, j, k] = sum_r V[j, k, r] * (U[:, j, r] * W[k, :, r])
        # M_v[j,k] shape (n, rank). Entry (i, r) = U[i, j, r] * W[k, i, r]
        
        for j in range(m):
            for k in range(p):
                M_v = U[:, j, :] * W[k, :, :] # Broadcasting: (n, rank) * (n, rank) -> (n, rank)
                
                A = M_v.T @ M_v
                b = M_v.T @ T[:, j, k]
                
                A += 1e-10 * np.eye(rank)
                
                try:
                    V[j, k, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    V[j, k, :] = np.linalg.lstsq(M_v, T[:, j, k], rcond=None)[0]

        # 3. Update W
        # For each k, i:
        # T[i, :, k] = sum_r W[k, i, r] * (U[i, :, r] * V[:, k, r])
        # M_w[k,i] shape (m, rank). Entry (j, r) = U[i, j, r] * V[j, k, r]
        
        for k in range(p):
            for i in range(n):
                M_w = U[i, :, :] * V[:, k, :] # Broadcasting: (m, rank) * (m, rank) -> (m, rank)
                
                A = M_w.T @ M_w
                b = M_w.T @ T[i, :, k]
                
                A += 1e-10 * np.eye(rank)
                
                try:
                    W[k, i, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    W[k, i, :] = np.linalg.lstsq(M_w, T[i, :, k], rcond=None)[0]

        # Calculate Error
        # T_calc[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # We can compute this efficiently using einsum or loops.
        # Given small dimensions, a loop is fine, but let's try vectorization.
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # Sum over r.
        # T_calc[i, j, k] = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = np.linalg.norm(T - T_calc)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # Verify Exactness
    # Since we want integer/half-integer entries, we should check if the solution is close to such values.
    # However, the problem asks for "integer/half-integer entries".
    # If ALS converges to a valid decomposition, the values should be close to rationals.
    # We will round to the nearest multiple of 0.5 and verify if it's exact.
    
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    T_calc_rounded = np.einsum('ijr,jkr,kir->ijk', U_rounded, V_rounded, W_rounded)
    
    if np.allclose(T, T_calc_rounded, atol=1e-9):
        # Convert to lists
        U_flat = U_rounded.reshape(-1, rank).tolist()
        V_flat = V_rounded.reshape(-1, rank).tolist()
        W_flat = W_rounded.reshape(-1, rank).tolist()
        
        # Double check indices mapping
        # U_flat row index: i*m+j. U_rounded[i, j, :] matches this.
        # V_flat row index: j*p+k. V_rounded[j, k, :] matches this.
        # W_flat row index: k*n+i. W_rounded[k, i, :] matches this.
        
        return U_flat, V_flat, W_flat
    else:
        return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the tensor.
    U: list of lists, shape (n*m, rank)
    V: list of lists, shape (m*p, rank)
    W: list of lists, shape (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Reshape
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(T, T_calc, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Known lower bounds/upper bounds
    # Schoolbook is 36.
    # Strassen-like blocks might get 33 or lower.
    # R(2,3,6) is likely around 21-27.
    
    best_result = None
    best_rank = 36
    
    # Try ranks from likely minimal upwards
    # Theoretical lower bound for R(n,m,p) is max(nm, mp, np) = max(6, 18, 12) = 18.
    # But structural constraints usually push it higher.
    # Let's try a range of ranks.
    
    # We will try a few seeds for each rank to increase chance of finding a solution.
    # Since ALS can get stuck in local minima, multiple restarts are key.
    
    ranks_to_try = list(range(18, 37))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few seeds for this rank
        # Limit seeds per rank to ensure we cover more ranks if early ones fail
        max_seeds = 5 if r < 25 else 20
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
                
            try:
                result = _als_decompose(r, max_iters=100, tol=1e-10, seed=seed)
                if result is not None:
                    U, V, W = result
                    # Final verification
                    if _verify_decomposition(U, V, W, n, m, p):
                        if r < best_rank:
                            best_rank = r
                            best_result = {"U": U, "V": V, "W": W}
                            # If we found a very low rank, we can stop early if we hit a theoretical lower bound?
                            # Lower bound is 18. If we hit 18, we can stop.
                            if r == 18:
                                return best_result
            except Exception:
                pass
                
    # Fallback to schoolbook if ALS failed or didn't find better
    if best_result is None:
        # Construct schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        
    return best_result

# Quick test in local environment (commented out for submission)
# if __name__ == "__main__":
#     res = construct()
#     print("Rank:", len(res['U'][0]))
#     print(_verify_decomposition(res['U'], res['V'], res['W'], n, m, p))
