"""Rewritten program for matrix multiplication 3,4,6 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Schoolbook rank
    rank_sb = n * m * p  # 72
    
    # Try to find a lower rank decomposition using random search + local optimization
    # The tensor T[i,j,k] = e_{i,j} * e_{j,k} * e_{k,i} in the appropriate indexing
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (i',j',k') where the product is 0, the sum = 0
    
    # Actually, the matrix multiplication tensor is:
    # M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor indices are (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U has rows indexed by (i,j), V by (j,k), W by (k,i)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # And 0 for invalid combinations? No - actually all (i,j,k) triples are "valid" 
    # in the sense that they all contribute to some entry. Wait, let me reconsider.
    
    # The matrix multiplication tensor T is a 3-tensor of size n x m x p where
    # T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k])
    # Wait no. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every such term appears in the computation.
    
    # Actually, the standard formulation: the matrix multiplication tensor of size (n,m,p)
    # has entries T[i,j,k] = delta_{...} but since it's just multiplication, T[i,j,k] = 1 
    # for all i,j,k. The rank is the minimum r such that we can write:
    # T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 72 equations. We want to minimize r.
    
    # Known bounds: For 3x4x6, the rank is known to be at most 72 (schoolbook).
    # Let's try random initialization with small r and see if we can solve.
    
    best_rank = rank_sb
    best_decomp = None
    
    # First, get the schoolbook solution as fallback
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try random search for smaller ranks
    # We'll try ranks from 40 up to 71
    rng = np.random.default_rng(42)
    
    for r in range(40, min(72, rank_sb)):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            U = rng.standard_normal((n*m, r)) * 0.1
            V = rng.standard_normal((m*p, r)) * 0.1
            W = rng.standard_normal((p*n, r)) * 0.1
            
            # Use gradient descent or least squares to minimize the error
            # The objective is: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            # This is non-convex, so we'll use a simple iterative approach
            # Actually, for fixed U and V, W can be solved in closed form, and vice versa.
            
            # Let's try alternating least squares
            # For fixed U, V: for each (k,i) pair, we have:
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            # This is a linear system in W[k*n+i,:]
            
            # Actually, let's think more carefully. For a fixed (k,i), the constraint is:
            # for all j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
            # Then: sum_c A_{j,c} * W[k*n+i,c] = 1 for all j
            # This is: A @ W[k*n+i,:]^T = ones(m)
            # Where A is m x r
            
            # Similarly for other variables.
            
            # Let's implement ALS
            for iteration in range(100):
                if time.time() > deadline - 20:
                    break
                
                # Update W: for each (k,i)
                for k in range(p):
                    for i in range(n):
                        # A[j,c] = U[i*m+j,c] * V[j*p+k,c] for j in range(m)
                        A = U[i*m:i*m+m, :] * V[k:k+m*p:m*p, :]  # Wait, this indexing is wrong
                        # V[j*p+k, c] for j in range(m) -> rows k, p+k, 2p+k, ...
                        # V has shape (m*p, r), row index j*p+k
                        V_rows = V[np.arange(m) * p + k, :]  # m x r
                        U_rows = U[np.arange(m) + i * m, :]  # m x r (rows i*m+j for j in range(m))
                        # Wait, U[i*m+j, c] for j in range(m) -> rows i*m, i*m+1, ..., i*m+m-1
                        U_rows = U[i*m:(i+1)*m, :]  # m x r
                        A = U_rows * V_rows  # m x r
                        # Solve A @ w = ones(m)
                        try:
                            w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Update U: for each (i,j)
                for i in range(n):
                    for j in range(m):
                        # For fixed (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # Let B_{k,c} = V[j*p+k,c] * W[k*n+i,c] for k in range(p)
                        V_rows = V[j*p:(j+1)*p, :]  # p x r
                        W_rows = W[np.arange(p) * n + i, :]  # p x r (rows k*n+i for k in range(p))
                        B = V_rows * W_rows  # p x r
                        try:
                            u, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V: for each (j,k)
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let C_{i,c} = U[i*m+j,c] * W[k*n+i,c] for i in range(n)
                        U_rows = U[np.arange(n) * m + j, :]  # n x r
                        W_rows = W[np.arange(p) * n + k, :]  # Wait, W[k*n+i, c] for i in range(n)
                        # W has shape (p*n, r), row index k*n+i
                        W_rows = W[k*n:(k+1)*n, :]  # n x r
                        C = U_rows * W_rows  # n x r
                        try:
                            v, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                if error < 1e-10:
                    best_rank = r
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    break
            
            if best_rank == r:
                break
        
        if best_rank == r:
            break
    
    # Convert to lists if we found a better solution
    if best_decomp is not None:
        U_mat, V_mat, W_mat = best_decomp
        # Round to nearest half-integer if close
        U_list = [[round(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-6 else x for x in row] for row in U_mat]
        V_list = [[round(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-6 else x for x in row] for row in V_mat]
        W_list = [[round(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-6 else x for x in row] for row in W_mat]
        
        # Verify the decomposition
        r = U_mat.shape[1]
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = sum(U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c] for c in range(r))
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = [[0] * rank_sb for _ in range(n * m)]
    V_list = [[0] * rank_sb for _ in range(m * p)]
    W_list = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
