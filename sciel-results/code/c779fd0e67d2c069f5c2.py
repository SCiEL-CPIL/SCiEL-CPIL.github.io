"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    def solve_333(rank, max_iter=1000, tol=1e-12):
        """
        Attempt to find a rank-`rank` decomposition for 3x3x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays of shape (9, rank) if found, else None.
        """
        # Construct target tensor T of shape (9, 9, 9)
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T[a, b, c] = 1.0

        # Initialize U, V, W
        # Use random initialization
        np.random.seed(42 + rank)
        U = np.random.randn(9, rank) * 0.1
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # ALS iterations
        # We want sum_r U[:,r] V[:,r] W[:,r] = T
        # Flatten T to vector of length 729
        T_flat = T.flatten()
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
                
            # Update U
            # For each r, U[:,r] is determined by T and V,W
            # This is a linear system for each column of U?
            # Actually, we can update all columns of U simultaneously.
            # T = sum_r u_r v_r^T w_r^T (in some sense)
            # Reshape T to (9, 9*9)
            T_mat = T.reshape(9, 81)
            # V_W = sum_r v_r w_r^T ? No.
            # The equation is T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed r, U[:,r] is a vector.
            # We can view this as a linear least squares problem for U.
            # Let's form the matrix for U.
            # T_flat = sum_r U[:,r] (V[:,r] * W[:,r]) (element-wise product of V and W columns? No)
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let Z_r[b,c] = V[b,r] W[c,r]
            # T[a,:,:] = sum_r U[a,r] Z_r
            # This is not a standard matrix product unless we vectorize.
            # vec(T) = sum_r (U[:,r] \otimes V[:,r] \otimes W[:,r])
            
            # Efficient ALS step for U:
            # Minimize || T - sum_r u_r v_r w_r^T ||_F
            # This is equivalent to minimizing || T_mat - U (V_W) ||_F
            # where T_mat is (9, 81) and V_W is (81, rank)?
            # Let's check dimensions.
            # T[a, b*9+c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let M_VW be a matrix of size (81, rank) where M_VW[b*9+c, r] = V[b,r] W[c,r]
            # Then T_mat = U M_VW
            # So U = T_mat M_VW^T (M_VW M_VW^T)^{-1} (least squares)
            
            M_VW = np.zeros((81, rank))
            for b in range(9):
                for c in range(9):
                    M_VW[b*9+c, :] = V[b, :] * W[c, :]
            
            # Solve U M_VW = T_mat
            # U = T_mat M_VW^T (M_VW M_VW^T)^{-1}
            try:
                M_VW_T_M_VW = M_VW.T @ M_VW
                # Add small regularization for stability
                M_VW_T_M_VW += 1e-9 * np.eye(rank)
                U = T_mat @ M_VW.T @ np.linalg.inv(M_VW_T_M_VW)
            except np.linalg.LinAlgError:
                U = T_mat @ M_VW.T @ np.linalg.pinv(M_VW_T_M_VW)

            # Update V
            # T[b, a*9+c] = sum_r V[b,r] U[a,r] W[c,r]
            # M_UW = (81, rank) where M_UW[a*9+c, r] = U[a,r] W[c,r]
            M_UW = np.zeros((81, rank))
            for a in range(9):
                for c in range(9):
                    M_UW[a*9+c, :] = U[a, :] * W[c, :]
            T_mat_V = T.transpose(1, 0, 2).reshape(9, 81) # T[b, a, c]
            try:
                M_UW_T_M_UW = M_UW.T @ M_UW
                M_UW_T_M_UW += 1e-9 * np.eye(rank)
                V = T_mat_V @ M_UW.T @ np.linalg.inv(M_UW_T_M_UW)
            except np.linalg.LinAlgError:
                V = T_mat_V @ M_UW.T @ np.linalg.pinv(M_UW_T_M_UW)

            # Update W
            # T[c, a*9+b] = sum_r W[c,r] U[a,r] V[b,r]
            M_UV = np.zeros((81, rank))
            for a in range(9):
                for b in range(9):
                    M_UV[a*9+b, :] = U[a, :] * V[b, :]
            T_mat_W = T.transpose(2, 0, 1).reshape(9, 81) # T[c, a, b]
            try:
                M_UV_T_M_UV = M_UV.T @ M_UV
                M_UV_T_M_UV += 1e-9 * np.eye(rank)
                W = T_mat_W @ M_UV.T @ np.linalg.inv(M_UV_T_M_UV)
            except np.linalg.LinAlgError:
                W = T_mat_W @ M_UV.T @ np.linalg.pinv(M_UV_T_M_UV)

            # Check convergence
            # Compute error
            # T_recon = sum_r U[:,r] V[:,r] W[:,r]
            # Efficient check:
            err = 0.0
            # Sample a few entries or compute full norm
            # Full norm is 729 * rank operations, which is small (729*23 ~ 16k)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break

        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2

        # Verify exactness
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        if np.allclose(T_recon, T, atol=1e-9):
            return U_snap, V_snap, W_snap
        return None

    # Try to find decomposition for 3x3x3
    # Start with rank 23 (best known)
    U_333 = None
    V_333 = None
    W_333 = None
    found_rank = None
    
    for r in range(23, 28): # 23, 24, 25, 26, 27
        if time.time() > deadline - 10:
            break
        res = solve_333(r)
        if res is not None:
            U_333, V_333, W_333 = res
            found_rank = r
            break

    if found_rank is not None:
        # Construct 3x3x6 decomposition
        # Rank is 2 * found_rank
        total_rank = 2 * found_rank
        
        # U: (9, total_rank)
        # First half columns: U_333
        # Second half columns: U_333
        U_full = np.zeros((9, total_rank))
        U_full[:, :found_rank] = U_333
        U_full[:, found_rank:] = U_333
        
        # V: (18, total_rank)
        # First half columns: V_333 in rows 0-8
        # Second half columns: V_333 in rows 9-17
        V_full = np.zeros((18, total_rank))
        V_full[:9, :found_rank] = V_333
        V_full[9:, found_rank:] = V_333
        
        # W: (18, total_rank)
        # First half columns: W_333 in rows 0-8
        # Second half columns: W_333 in rows 9-17
        W_full = np.zeros((18, total_rank))
        W_full[:9, :found_rank] = W_333
        W_full[9:, found_rank:] = W_333
        
        # Convert to list of lists
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
