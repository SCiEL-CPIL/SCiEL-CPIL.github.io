"""Rewritten program for 2x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # The tensor T_{i,j,k} = 1 if (i,j) in A, (j,k) in B, (i,k) in C
    # For 2x5x5: A has 10 entries, B has 25, C has 10
    # T is a 3-tensor of shape (10, 25, 10)
    
    # Indexing:
    # U: (n*m, rank) = (10, rank), row index = i*m + j = i*5 + j
    # V: (m*p, rank) = (25, rank), row index = j*p + k = j*5 + k
    # W: (p*n, rank) = (10, rank), row index = k*n + i = k*2 + i
    
    # Tensor entry T[i*5+j, j*5+k, k*2+i] = 1 for all valid i,j,k
    
    # Build the tensor
    T = np.zeros((10, 25, 10), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # Known bounds: For 2x5x5, schoolbook gives rank 50.
    # There are known results that the rank is at most 40 or so.
    # Let's try various ranks from low to high.
    
    # Actually, let me think about this more carefully.
    # The multiplication tensor for m x n x p has rank at most m*n*p (schoolbook).
    # For 2x5x5, that's 50.
    # Can we do better?
    
    # One approach: use the fact that 2x5x5 can be related to other known decompositions.
    # For example, if we can embed 2x5x5 into a larger problem with known optimal rank.
    
    # Alternative: use numerical optimization to find a low-rank decomposition.
    # We want to find U (10 x r), V (25 x r), W (10 x r) such that:
    # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k]
    
    # This is equivalent to: for each c, U_c, V_c, W_c define a rank-1 tensor,
    # and we need the sum to equal T.
    
    # Let's try a simple approach: start with schoolbook rank 50 and try to reduce.
    
    # Actually, let's try to use the structure. The 2x5x5 tensor has some symmetry.
    
    # Let me try a direct numerical search for low rank.
    
    def try_rank(r, max_iter=200, tol=1e-8):
        """Try to find rank-r decomposition using alternating least squares."""
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(10, r)
        V = np.random.randn(25, r)
        W = np.random.randn(10, r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each i, j, U[i,j,:] should satisfy
            # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k] for all k
            # This is: U[i,j,:] * (V[j,:] ⊙ W[i,:])^T = T[i,j,:]
            # where ⊙ is element-wise product
            # So U[i,j,:] = argmin ||U[i,j,:] * (V[j,:] ⊙ W[i,:])^T - T[i,j,:]||
            
            for i in range(n):
                for j in range(m):
                    # V[j,:] has shape (25, r) but we only need entries corresponding to j
                    # V row index is j*p + k = j*5 + k, so V[j*5:(j+1)*5, :]
                    Vj = V[j*p:(j+1)*p, :]  # shape (5, r)
                    # W row index is k*n + i = k*2 + i
                    Wi = W[[k*n + i for k in range(p)], :]  # shape (5, r)
                    # Element-wise product: shape (5, r)
                    prod = Vj * Wi  # (5, r)
                    # We want U[i,j,:] such that U[i,j,:] @ prod.T ≈ T[i,j,:]
                    # T[i,j,:] has entries T[i*5+j, j*5+k, k*2+i] for k=0..4
                    target = np.array([T[i*m+j, j*p+k, k*n+i] for k in range(p)])
                    # Solve least squares: U[i,j,:] @ prod.T = target
                    # This is: prod @ U[i,j,:]^T = target^T
                    # So U[i,j,:]^T = prod^+ @ target
                    try:
                        sol, residuals, rank_u, sv = np.linalg.lstsq(prod.T, target, rcond=None)
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Update V: for each j, k, V[j,k,:] should satisfy
            # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k] for all i
            for j in range(m):
                for k in range(p):
                    # U row index is i*m + j = i*5 + j
                    Uj = U[[i*m + j for i in range(n)], :]  # shape (2, r)
                    # W row index is k*n + i = k*2 + i
                    Wk = W[[k*n + i for i in range(n)], :]  # shape (2, r)
                    prod = Uj * Wk  # (2, r)
                    target = np.array([T[i*m+j, j*p+k, k*n+i] for i in range(n)])
                    try:
                        sol, residuals, rank_v, sv = np.linalg.lstsq(prod.T, target, rcond=None)
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Update W: for each i, k, W[i,k,:] should satisfy
            # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k] for all j
            for i in range(n):
                for k in range(p):
                    # U row index is i*m + j = i*5 + j
                    Ui = U[i*m:(i+1)*m, :]  # shape (5, r)
                    # V row index is j*p + k = j*5 + k
                    Vk = V[[j*p + k for j in range(m)], :]  # shape (5, r)
                    prod = Ui * Vk  # (5, r)
                    target = np.array([T[i*m+j, j*p+k, k*n+i] for j in range(m)])
                    try:
                        sol, residuals, rank_w, sv = np.linalg.lstsq(prod.T, target, rcond=None)
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - T[i*m+j, j*p+k, k*n+i])**2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        return best_err, best_U, best_V, best_W
    
    # Try ranks from low to high
    for r in range(15, 51):
        if time.time() > deadline - 5:
            break
        err, U, V, W = try_rank(r, max_iter=100)
        if err < 1e-6:
            # Found a decomposition, try to convert to exact rational
            # For now, return the numerical solution
            # Convert to lists
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(10)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(25)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(10)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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
