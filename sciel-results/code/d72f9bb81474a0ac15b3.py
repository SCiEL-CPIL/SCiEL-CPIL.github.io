"""Program for 3x3x4 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _rank_of_tensor(T, rank):
    """
    Attempt to find a rank-rank decomposition of tensor T.
    T has shape (a, b, c).
    We want U (a, rank), V (b, rank), W (c, rank) such that
    T[i,j,k] = sum_l U[i,l]*V[j,l]*W[k,l].
    """
    a, b, c = T.shape
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(a, rank)
    V = np.random.randn(b, rank)
    W = np.random.randn(c, rank)
    
    # We use a simple alternating minimization approach
    # For fixed V, W: minimize ||T - U @ (V^T * W)|| in some sense
    # Actually, the standard approach is to use gradient descent on the flattened problem
    
    # Let's use a simpler approach: treat it as a tensor completion / low-rank approximation
    # We'll use the approach of minimizing the Frobenius norm of the difference
    
    # Reshape approach:
    # T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
    # This is equivalent to: for each l, the slice T_l = U[:,l] outer V[:,l] outer W[:,l]
    
    # Let's use a coordinate descent / alternating least squares approach
    # But that's complex. Let's try a simpler gradient-based approach.
    
    # Actually, for small problems, we can try to use the fact that
    # if we fix V and W, the optimal U is given by solving a linear system.
    
    # For fixed V, W:
    # T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
    # For fixed i, this is: T[i,:,:] = sum_l U[i,l] * (V[:,l] outer W[:,l])
    # So T[i,:,:] = sum_l U[i,l] * M_l where M_l = V[:,l] outer W[:,l]
    # This is a linear system in U[i,:] for each i.
    
    # Let's build the matrix B where B[j,k,l] = V[j,l] * W[k,l]
    # Then T[i,:,:] flattened = U[i,:] @ B_flattened
    # Where B_flattened has shape (b*c, rank)
    
    # Similarly for V and W.
    
    # Let's implement alternating least squares
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for iteration in range(500):
        # Update U
        # For each i: T[i,:,:] = sum_l U[i,l] * V[:,l] outer W[:,l]
        # Flatten: T[i].flatten() = U[i,:] @ (V*W).flatten()
        # Where (V*W) has shape (b*c, rank) with (V*W)[j*c+k, l] = V[j,l]*W[k,l]
        
        VW = np.einsum('jl,kl->jkl', V, W).reshape(b*c, rank)
        # T_flat[i] = U[i,:] @ VW.T  (if we think of VW as (b*c, rank))
        # Actually: T[i,j,k] = sum_l U[i,l] * VW[j*c+k, l]
        # So T[i].flatten() = VW.T @ U[i,:]  -> no wait
        # T[i,j,k] = sum_l U[i,l] * VW[j*c+k, l]
        # Let T_i = T[i].flatten() (shape b*c)
        # T_i[j*c+k] = sum_l U[i,l] * VW[j*c+k, l]
        # So T_i = VW.T @ U[i,:]  ... no, VW has shape (b*c, rank)
        # T_i[j*c+k] = sum_l VW[j*c+k, l] * U[i,l]
        # So T_i = VW @ U[i,:]  (shape b*c)
        # We want to find U[i,:] that minimizes ||T_i - VW @ U[i,:]||
        # This is a least squares problem: U[i,:] = (VW.T @ VW)^-1 VW.T @ T_i
        
        T_flat = T.reshape(a, b*c)
        # Solve: min_{U_i} ||T_flat[i] - VW @ U_i||
        # VW has shape (b*c, rank), U_i has shape (rank,)
        # So we want to solve VW @ U_i = T_flat[i]
        # This is an underdetermined system (b*c >= rank typically)
        # Use least squares: U_i = pinv(VW) @ T_flat[i]
        
        VW_pinv = np.linalg.pinv(VW)
        U_new = (VW_pinv @ T_flat.T).T  # shape (a, rank)
        
        # Update V
        UW = np.einsum('il,kl->ikl', U, W).reshape(a*c, rank)
        # T[j].flatten() where T[j] is the slice with j fixed
        # T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
        # For fixed j: T[:,j,:].flatten() = sum_l V[j,l] * (U[:,l] outer W[:,l]).flatten()
        # Let UW have shape (a*c, rank) with UW[i*c+k, l] = U[i,l]*W[k,l]
        # T[:,j,:].flatten() = UW @ V[j,:]
        T_j_flat = T.transpose(1,0,2).reshape(b, a*c)
        UW_pinv = np.linalg.pinv(UW)
        V_new = (UW_pinv @ T_j_flat.T).T  # shape (b, rank)
        
        # Update W
        UV = np.einsum('il,jl->ijl', U, V).reshape(a*b, rank)
        # T[k].flatten() where T[k] is the slice with k fixed
        # T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
        # For fixed k: T[:,:,k].flatten() = sum_l W[k,l] * (U[:,l] outer V[:,l]).flatten()
        # Let UV have shape (a*b, rank) with UV[i*b+j, l] = U[i,l]*V[j,l]
        # T[:,:,k].flatten() = UV @ W[k,:]
        T_k_flat = T.transpose(2,0,1).reshape(c, a*b)
        UV_pinv = np.linalg.pinv(UV)
        W_new = (UV_pinv @ T_k_flat.T).T  # shape (c, rank)
        
        # Check convergence
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        T_reconstructed = np.einsum('il,jl,kl->ijk', U, V, W)
        err = np.linalg.norm(T - T_reconstructed) / np.linalg.norm(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-10:
            break
    
    # Convert to fractions
    def to_fraction_matrix(mat):
        frac_mat = []
        for row in mat:
            frac_row = []
            for val in row:
                frac_row.append(Fraction(val).limit_denominator(1000))
            frac_mat.append(frac_row)
        return frac_mat
    
    # Verify the decomposition with fractions
    U_frac = to_fraction_matrix(best_U)
    V_frac = to_fraction_matrix(best_V)
    W_frac = to_fraction_matrix(best_W)
    
    # Check if the decomposition is exact
    a, b, c = T.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                val = sum(U_frac[i][l] * V_frac[j][l] * W_frac[k][l] for l in range(rank))
                if abs(float(val) - T[i,j,k]) > 1e-9:
                    return None
    
    return U_frac, V_frac, W_frac

def construct():
    """Construct a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Try ranks from known lower bounds up to schoolbook
    # For 3x3x4, the schoolbook rank is 36
    # Known lower bound is at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Let's try from 12 up to 36
    
    for rank in range(12, 37):
        result = _rank_of_tensor(T, rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
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
