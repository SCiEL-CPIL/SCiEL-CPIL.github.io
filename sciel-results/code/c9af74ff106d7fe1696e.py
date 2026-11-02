import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i,j,k
    # But wait, the tensor is M_{n,m,p}.
    # T[i, j, k] corresponds to A[i,j] * B[j,k] -> C[i,k].
    # So T is all ones.
    
    # We want U, V, W such that T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]
    # U is n*m x r
    # V is m*p x r
    # W is p*n x r
    
    # The tensor T can be reshaped to (n*m, m*p, p*n)?
    # No, the indices are (i,j) for U, (j,k) for V, (k,i) for W.
    # T[i,j,k] is the entry.
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k] = 1.
    
    # Let's try ranks from 21 to 80.
    
    def check_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute T_hat[i, j, k]
        # T_hat[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can do this via tensor contraction.
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r).
        # T_hat = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        # Compare with T = ones(n, m, p)
        pass

    def als_decomposition(rank, max_iters=100, tol=1e-9):
        # Initialize U, V, W
        # Use random initialization or Strassen
        # Strassen for 4x4x4 is rank 20.
        # For 4x4x5, we can pad.
        
        # For simplicity, start with random.
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for _ in range(max_iters):
            # Update U
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # For fixed V, W, U is linear.
            # U[i,j,:] = argmin sum_k (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
            # This is a least squares problem.
            
            # We can vectorize.
            # Let's reshape V to (m, p, r) and W to (p, n, r).
            # V_mat[j, k, r]
            # W_mat[k, i, r]
            
            # For each i, j:
            # We need to match T[i, j, :] (vector of length p)
            # T[i, j, k] = sum_r U[i, j, r] * (V[j, k, r] * W[k, i, r])
            # Let M_ij[r, k] = V[j, k, r] * W[k, i, r]
            # Then T[i, j, :] = M_ij @ U[i, j, :]
            # U[i, j, :] = lstsq(M_ij, T[i, j, :])
            
            # This is expensive if done naively.
            # We can use np.einsum.
            
            # Update U
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # M_ij[r, k] = V[j, k, r] * W[k, i, r]
            # We want to solve for U[i, j, r].
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            
            # Let's compute the linear operator for U.
            # For each i, j, the system is:
            # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1
            
            # We can build the matrix A[i, j, k, r] = V[j, k, r] * W[k, i, r]
            # Then A[i, j, k, r] U[i, j, r] = 1
            # This is a linear system for each (i, j).
            
            # A[i, j] is a (p x rank) matrix.
            # A[i, j, k, r] = V[j, k, r] * W[k, i, r]
            
            # We can compute this using einsum.
            # A = np.einsum('jkr,kir->ijk r', V_reshaped, W_reshaped)
            # A shape: (n, m, p, rank)
            
            A = np.einsum('jkr,kir->ijk r', V_reshaped, W_reshaped)
            
            # Solve A[i, j] @ U[i, j] = 1
            # U[i, j] = lstsq(A[i, j], 1)
            
            # We can do this in a loop or vectorized.
            # Vectorized lstsq is tricky.
            # Let's loop over i, j.
            U_new = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    # A[i, j] is (p, rank)
                    # b is (p,) of ones
                    b = np.ones(p)
                    # Solve A @ x = b
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(A[i, j], b, rcond=None)
                        U_new[i, j] = x
                    except:
                        U_new[i, j] = U[i*m+j]
            
            U = U_new.reshape(n*m, rank)
            
            # Similarly for V and W.
            # Update V
            U_reshaped = U.reshape(n, m, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # For fixed U, W:
            # T[i, j, k] = V[j, k, r] * (U[i, j, r] * W[k, i, r])
            # Let B[j, k, r, i] = U[i, j, r] * W[k, i, r]
            # Then sum_r V[j, k, r] * B[j, k, r, i] = 1
            # This is a system for V[j, k, :].
            
            B = np.einsum('ijr,kir->jkri', U_reshaped, W_reshaped)
            # B shape: (m, p, rank, n)
            
            V_new = np.zeros((m, p, rank))
            for j in range(m):
                for k in range(p):
                    # B[j, k] is (rank, n)
                    # We need V[j, k, r] such that sum_r V[j, k, r] * B[j, k, r, i] = 1 for all i
                    # So B[j, k].T @ V[j, k] = 1 (vector of length n)
                    # B[j, k].T is (n, rank)
                    b = np.ones(n)
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(B[j, k].T, b, rcond=None)
                        V_new[j, k] = x
                    except:
                        V_new[j, k] = V[j*p+k]
            
            V = V_new.reshape(m*p, rank)
            
            # Update W
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            
            C = np.einsum('ijr,jkr->kir', U_reshaped, V_reshaped)
            # C shape: (p, n, rank)
            
            W_new = np.zeros((p, n, rank))
            for k in range(p):
                for i in range(n):
                    # C[k, i] is (rank,)
                    # We need W[k, i, r] such that sum_r W[k, i, r] * C[k, i, r] = 1
                    # This is just dot product.
                    # W[k, i] = C[k, i] / (C[k, i] @ C[k, i])?
                    # No, W is a vector.
                    # sum_r W_r C_r = 1.
                    # This is underdetermined if rank > 1.
                    # We want the minimum norm solution?
                    # W = C / (C @ C)
                    norm_sq = np.dot(C[k, i], C[k, i])
                    if norm_sq > 1e-12:
                        W_new[k, i] = C[k, i] / norm_sq
                    else:
                        W_new[k, i] = W[k*n+i]
            
            W = W_new.reshape(p*n, rank)
            
            # Check convergence
            # Compute error
            T_hat = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
            error = np.linalg.norm(T_hat - 1.0)
            if error < tol:
                break
        
        return U, V, W

    # Try ranks
    best_rank = 80
    best_U, best_V, best_W = None, None, None
    
    for r in range(21, 81):
        if time.time() > 100: break
        U, V, W = als_decomposition(r)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check exactness
        T_hat = np.einsum('ijr,jkr,kir->ijk', 
                          U_rounded.reshape(n,m,r), 
                          V_rounded.reshape(m,p,r), 
                          W_rounded.reshape(p,n,r))
        
        if np.allclose(T_hat, 1.0, atol=1e-9):
            best_rank = r
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            break
        
    # If no exact decomposition found, use schoolbook
    if best_U is None:
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
