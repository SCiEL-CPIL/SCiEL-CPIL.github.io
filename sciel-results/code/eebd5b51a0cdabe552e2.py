import numpy as np
from fractions import Fraction
from itertools import product
import time

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = 145  # 145 seconds to be safe
    
    # First, try to find a lower-rank decomposition via linear algebra
    # The tensor for matrix multiplication: T_{(i,j),k,l} = delta_{i,j} * delta_{k,l} for C_{i,l} = sum_j A_{i,j} B_{j,l}
    # We want to decompose T as sum_r U_{(i,j),r} * V_{k,l,r} * W_{(i,l),r}
    
    # Flatten indices: A has n*m=8 entries, B has m*p=24 entries, C has p*n=12 entries
    # Tensor T is 8x24x12
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i*m+j, j*p+l, l*n+i] = 1
    
    # Try ranks from 36 upward
    # Use a greedy/randomized approach to find a decomposition
    
    def try_rank(r):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random rational vectors
        np.random.seed(42)
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Iteratively refine: for each r, fix U and V, solve for W, etc.
        # But this is nonlinear. Instead, use a different approach.
        
        # Alternative: use the fact that for each slice, we have linear constraints
        # T_{a,b,c} = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # This is a 3-way tensor decomposition problem.
        # Use ALS (Alternating Least Squares) on the tensor.
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                return None
            
            # Update W: for each c, W[c,:] = (sum_a,b T[a,b,c] * U[a,:] * V[b,:]) / (sum_a,b U[a,:]^2 * V[b,:]^2)
            # Actually, the equation is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed U,V: for each c, we have a system: sum_r (U[:,r] * V[:,r])^T * W[c,r] = T[:,:,c].flatten()
            # This is: M[c] * W[c,:]^T = T[:,:,c].flatten() where M[c]_{ab,r} = U[a,r]*V[b,r]
            
            # More carefully: for each c, the vector T[:,:,c] (size n*m*m*p) should equal 
            # sum_r W[c,r] * (U[:,r] * V[:,r]) (elementwise product, size n*m*m*p)
            
            # So: T_flat = W @ (U*V)_flat where (U*V) has shape (r, n*m*m*p)
            
            UV = np.zeros((r, n*m, m*p))
            for r_idx in range(r):
                UV[r_idx] = np.outer(U[:, r_idx], V[:, r_idx])
            
            # Reshape to (r, n*m*m*p)
            UV_flat = UV.reshape(r, -1)
            
            # For each c: T[:,:,c].flatten() = sum_r W[c,r] * UV_flat[r,:]
            # So: T_c = UV_flat.T @ W[c,:]
            # W[c,:] = (UV_flat @ UV_flat.T)^{-1} @ UV_flat @ T_c
            
            T_flat = T.reshape(n*m, m*p, p*n)  # already in this shape
            W_new = np.zeros((p*n, r))
            
            for c in range(p*n):
                T_c = T_flat[:, :, c].flatten()
                # Solve: UV_flat.T @ W[c,:] = T_c
                # This is an overdetermined system if n*m*m*p > r
                # Use least squares
                try:
                    W_c, _, _, _ = np.linalg.lstsq(UV_flat.T, T_c, rcond=None)
                    W_new[c, :] = W_c
                except:
                    W_new[c, :] = W[c, :]
            
            W = W_new
            
            # Update U: for each a, U[a,:] = (sum_b,c T[a,b,c] * V[b,:] * W[c,:]) / (sum_b,c V[b,:]^2 * W[c,:]^2)
            # Similarly: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed V,W: for each a, T[a,:,:].flatten() = sum_r U[a,r] * (V[:,r] * W[:,r])
            VW = np.zeros((r, m*p, p*n))
            for r_idx in range(r):
                VW[r_idx] = np.outer(V[:, r_idx], W[:, r_idx])
            VW_flat = VW.reshape(r, -1)
            
            U_new = np.zeros((n*m, r))
            for a in range(n*m):
                T_a = T_flat[a, :, :].flatten()
                try:
                    U_a, _, _, _ = np.linalg.lstsq(VW_flat.T, T_a, rcond=None)
                    U_new[a, :] = U_a
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V: for each b, V[b,:] = (sum_a,c T[a,b,c] * U[a,:] * W[c,:]) / (sum_a,c U[a,:]^2 * W[c,:]^2)
            UW = np.zeros((r, n*m, p*n))
            for r_idx in range(r):
                UW[r_idx] = np.outer(U[:, r_idx], W[:, r_idx])
            UW_flat = UW.reshape(r, -1)
            
            V_new = np.zeros((m*p, r))
            for b in range(m*p):
                T_b = T_flat[:, b, :].flatten()
                try:
                    V_b, _, _, _ = np.linalg.lstsq(UW_flat.T, T_b, rcond=None)
                    V_new[b, :] = V_b
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
        
        # Check if decomposition is valid
        T_recon = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_recon += np.outer(U[:, r_idx], V[:, r_idx])[:, :, None] * W[None, None, :]
            # Actually: T_recon[a,b,c] += U[a,r] * V[b,r] * W[c,r]
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            outer_UV = np.outer(U[:, r_idx], V[:, r_idx])
            T_recon += outer_UV[:, :, None] * W[None, None, :]
        
        if np.allclose(T_recon, T, atol=1e-6):
            # Convert to fractions
            U_frac = [[Fraction(U[i, j]).limit_denominator(1000) for j in range(r)] for i in range(n*m)]
            V_frac = [[Fraction(V[i, j]).limit_denominator(1000) for j in range(r)] for i in range(m*p)]
            W_frac = [[Fraction(W[i, j]).limit_denominator(1000) for j in range(r)] for i in range(p*n)]
            
            # Verify exactly
            T_exact = np.zeros((n*m, m*p, p*n), dtype=object)
            for r_idx in range(r):
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            T_exact[i, j, k] += U_frac[i][r_idx] * V_frac[j][r_idx] * W_frac[k][r_idx]
            
            T_target = np.zeros((n*m, m*p, p*n), dtype=object)
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        T_target[i*m+j, j*p+l, l*n+i] = 1
            
            if np.array_equal(T_exact, T_target):
                return (U_frac, V_frac, W_frac)
        
        return None
    
    # Try ranks from 36 to 41
    for r in range(36, 42):
        result = try_rank(r)
        if result is not None:
            U_frac, V_frac, W_frac = result
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to the Strassen construction (rank 42)
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)

    return {"U": U, "V": V, "W": W}
