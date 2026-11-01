"""Matrix multiplication 2,5,5 - rank minimization via Strassen embedding + ALS refinement."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized operations."""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1))
        return max_err < 1e-8

    def tensor_error(U, V, W, rank):
        """Compute max absolute error of the tensor reconstruction."""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(T - 1))

    def als_solve_vectorized(U, V, W, rank, max_iters=200, tol=1e-10):
        """Vectorized Alternating Least Squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Construct matricizations M1, M2, M3
        M1 = np.zeros((n*m, m*p * p*n))
        M2 = np.zeros((m*p, p*n * n*m))
        M3 = np.zeros((p*n, n*m * m*p))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p+k
                    c = k*n+i
                    col1 = (j*p+k)*(p*n) + i*n+k
                    M1[a, col1] = 1
                    col2 = (i*n+k)*(n*m) + i*m+j
                    M2[b, col2] = 1
                    col3 = (i*m+j)*(m*p) + j*p+k
                    M3[c, col3] = 1

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            V_flat = V.T
            W_flat = W.T
            K_VW = (V_flat[:, :, None] * W_flat[:, None, :]).reshape(rank, m*p * p*n)
            try:
                U = np.linalg.lstsq(K_VW.T, M1, rcond=None)[0]
            except:
                return None
                
            # Update V
            W_flat = W.T
            U_flat = U.T
            W_3d = W_flat.reshape(rank, p, n)
            U_3d = U_flat.reshape(rank, n, m)
            K_WU_3d = W_3d[:, :, :, None] * U_3d[:, None, :, :]
            K_WU = K_WU_3d.reshape(rank, p*n * n*m)
            try:
                V = np.linalg.lstsq(K_WU.T, M2, rcond=None)[0]
            except:
                return None
                
            # Update W
            U_flat = U.T
            V_flat = V.T
            U_3d = U_flat.reshape(rank, n, m)
            V_3d = V_flat.reshape(rank, m, p)
            K_UV_3d = U_3d[:, :, :, None] * V_3d[:, None, :, :]
            K_UV = K_UV_3d.reshape(rank, n*m * m*p)
            try:
                W = np.linalg.lstsq(K_UV.T, M3, rcond=None)[0]
            except:
                return None
                
            err = tensor_error(U, V, W, rank)
            if err < tol:
                return U, V, W
                
        return None

    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2

    rng = np.random.default_rng(42)
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Strategy: Try to decompose into two 1x5x5 blocks (rank 25 each? No, 1x5x5 schoolbook is 25).
    # The prompt suggests embedding 5x5 into 7x7 Strassen (rank 7) -> total 14.
    # However, 1x5x5 multiplication is just row * matrix.
    # T[i=0, j, k] = sum_l A[0,l] B[l,k].
    # This is a rank 25 tensor (schoolbook). Can it be lower?
    # The tensor for 1xm x mxp is isomorphic to the identity tensor? No.
    # It is the "matrix multiplication" with one fixed row.
    # Actually, the tensor T_{j,k,l} = delta_{j,l}. Wait.
    # (A B)_{0,k} = sum_l A_{0,l} B_{l,k}.
    # The tensor has indices (l, l, k)? No.
    # U index: i*m+j = 0*5+j = j.
    # V index: j*p+k = l*5+k.
    # W index: k*n+i = k*2+0 = 2k.
    # T[j, l, k] = 1 if j=l, else 0.
    # So T is a tensor that is 1 on the "diagonal" of the first two indices, and 0 otherwise.
    # Specifically, T[j, l, k] = delta_{j,l}.
    # This tensor is a direct sum of 5 identity matrices of size (5x5) in the (j,l) plane?
    # No, the third index k is free.
    # T is a 5x5x5 tensor. T[j,l,k] = delta_{j,l}.
    # This is equivalent to 5 copies of the identity tensor in 2D?
    # The rank of the identity tensor I_m (m x m) is m.
    # Here we have 5 independent "blocks" along the k dimension?
    # No, k is the same for all j,l.
    # Let's check the rank of T where T[j,l,k] = delta_{j,l}.
    # This is the tensor that maps (A, B) -> (A B)_{0,:}.
    # This is a linear map from R^5 x R^{5x5} -> R^5.
    # It is essentially the identity map on R^5 if we fix the column of B?
    # No.
    # Let's use the ALS solver to find the minimal rank for this specific structure.
    # We can initialize with the schoolbook (rank 25) and try to reduce.
    
    # First, try to find a decomposition for the whole 2x5x5 tensor.
    # Initialize with Strassen-like random starts or specific structures.
    
    # Hypothesis: The rank is 25 (sum of two 1x5x5 which are rank 12.5? No, integer rank).
    # Actually, the tensor T[j,l,k]=delta_{j,l} for 1x5x5 has rank 5?
    # Let's test rank 5 for 1x5x5.
    # If rank is 5, then 2x5x5 might be 10?
    # Or maybe 12?
    
    # Let's run ALS for ranks from 10 up to 25.
    
    # Pre-compute M1, M2, M3 for the full 2x5x5 tensor to avoid recomputation in ALS
    # The ALS function above recomputes them every call. Let's optimize that or just call it.
    # For 2x5x5, M1 is 10x250. Very small.
    
    for r in range(10, 26):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            result = als_solve_vectorized(U, V, W, r, max_iters=100)
            if result is None:
                continue
            
            U_sol, V_sol, W_sol = result
            err = tensor_error(U_sol, V_sol, W_sol, r)
            
            if err < 1e-8:
                # Try to round to half-integers
                U_r = round_to_half_int(U_sol)
                V_r = round_to_half_int(V_sol)
                W_r = round_to_half_int(W_sol)
                
                if verify(U_r, V_r, W_r, r):
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    success = True
                    break
        
        if success:
            # If we found a solution at rank r, can we go lower?
            # The loop continues to r-1? No, range is ascending.
            # We should break if we found the minimum possible?
            # We don't know the minimum. But if we found 10, we can stop.
            # If we found 25, we continue.
            # Actually, we want the MINIMAL rank.
            # So we should search from low to high and stop at the first success?
            # Yes.
            break

    return best_result

# EVOLVE-BLOCK-END
