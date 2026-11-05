"""Program to find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Construct the multiplication tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' and j==j', k==k', i==i'
    # Indices:
    # a in [0, n*m)
    # b in [0, m*p)
    # c in [0, p*n)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Function to check if factors U, V, W of rank R decompose T
    def check_decomposition(U, V, W, T):
        # Reconstruct T
        # T_rec[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
        # Efficient computation:
        # For each k, add U[:, k] * V[:, k] * W[:, k]
        # This is sum of outer products of vectors?
        # U[:, k] is (n*m,), V[:, k] is (m*p,), W[:, k] is (p*n,)
        # We can use einsum: np.einsum('ak,bk,ck->abc', U, V, W)
        T_rec = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        error = np.linalg.norm(T_rec - T)
        return error

    # ALS Step: Update U given V, W
    def update_U(V, W, T):
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # For fixed a, T[a, :, :] = sum_k U[a, k] (V[:, k] * W[:, k])
        # Let Z[k, b, c] = V[b, k] * W[c, k]
        # Then T[a, :, :] = sum_k U[a, k] Z[k, :, :]
        # Flatten T[a, :, :] to vector y_a of size m*p*p*n
        # Flatten Z[k, :, :] to vector z_k of size m*p*p*n
        # y_a = sum_k U[a, k] z_k
        # This is a linear system: U[a, :] @ Z_flat.T = y_a.T ?
        # Z_flat is (R, size). U[a, :] is (1, R).
        # U[a, :] @ Z_flat.T is (1, size).
        # So we solve for U[a, :] such that U[a, :] @ Z_flat.T = y_a.T
        # U[a, :] = y_a.T @ (Z_flat.T) @ (Z_flat.T @ Z_flat.T)^-1 ? No.
        # Let A = Z_flat.T (size x R). We want x (1 x R) such that x A = y (1 x size).
        # x A^T = y^T (R x size? No. y is 1xsize. y^T is sizex1. A^T is R x size. x^T is R x 1.)
        # x^T = (A^T)^+ y^T
        # x = ( (A^T)^+ y^T )^T = y ( (A^T)^+ )^T = y ( A^+ )^T ?
        # Let's just use np.linalg.lstsq.
        # We want to solve X @ A.T = Y.T for X (1xR), where A is (R x Size), Y is (1 x Size).
        # Actually, it's easier to solve for each a:
        # Flatten Z to Z_flat (R, Size)
        # Flatten T[a] to Y_a (Size,)
        # We want U_a (R,) such that U_a @ Z_flat.T = Y_a.T ? No.
        # sum_k U_a[k] * Z_flat[k, j] = Y_a[j]
        # This is U_a @ Z_flat.T = Y_a.T ? No.
        # Z_flat is (R, Size). Z_flat.T is (Size, R).
        # U_a is (R,). U_a @ Z_flat.T is (Size,).
        # So we want U_a @ Z_flat.T = Y_a.
        # This is a linear system for U_a: Z_flat.T @ U_a.T = Y_a.T
        # We can use lstsq: U_a.T = np.linalg.lstsq(Z_flat.T, Y_a.T, rcond=None)[0]
        
        R = V.shape[1]
        Size = m * p * p * n
        Z = V[:, :, np.newaxis] * W[np.newaxis, :, :] # (m*p, p*n, R)
        Z_flat = Z.transpose(2, 0, 1).reshape(R, Size) # (R, Size)
        
        U_new = np.zeros((n*m, R))
        for a in range(n*m):
            Y_a = T[a].flatten()
            # Solve Z_flat.T @ U_a.T = Y_a.T
            # Z_flat.T is (Size, R). Y_a.T is (Size, 1).
            sol, residuals, rank, sv = np.linalg.lstsq(Z_flat.T, Y_a.reshape(-1, 1), rcond=None)
            U_new[a] = sol.flatten()
        return U_new

    def update_V(U, W, T):
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # For fixed b, T[:, b, :] = sum_k V[b, k] (U[:, k] * W[:, k])
        # Z[k, a, c] = U[a, k] * W[c, k]
        # T[:, b, :] = sum_k V[b, k] Z[k, :, :]
        
        R = U.shape[1]
        Size = n * m * p * n
        Z = U[:, :, np.newaxis] * W[np.newaxis, :, :] # (n*m, p*n, R)
        Z_flat = Z.transpose(2, 0, 1).reshape(R, Size) # (R, Size)
        
        V_new = np.zeros((m*p, R))
        for b in range(m*p):
            Y_b = T[:, b, :].flatten()
            sol, residuals, rank, sv = np.linalg.lstsq(Z_flat.T, Y_b.reshape(-1, 1), rcond=None)
            V_new[b] = sol.flatten()
        return V_new

    def update_W(U, V, T):
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # For fixed c, T[:, :, c] = sum_k W[c, k] (U[:, k] * V[:, k])
        # Z[k, a, b] = U[a, k] * V[b, k]
        # T[:, :, c] = sum_k W[c, k] Z[k, :, :]
        
        R = U.shape[1]
        Size = n * m * m * p
        Z = U[:, :, np.newaxis] * V[np.newaxis, :, :] # (n*m, m*p, R)
        Z_flat = Z.transpose(2, 0, 1).reshape(R, Size) # (R, Size)
        
        W_new = np.zeros((p*n, R))
        for c in range(p*n):
            Y_c = T[:, :, c].flatten()
            sol, residuals, rank, sv = np.linalg.lstsq(Z_flat.T, Y_c.reshape(-1, 1), rcond=None)
            W_new[c] = sol.flatten()
        return W_new

    # Try ranks from 56 up to 60
    # The directive suggests 56 is a target.
    for R in range(56, 61):
        # Initialize
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Run ALS
        converged = False
        for it in range(100):
            U = update_U(V, W, T)
            V = update_V(U, W, T)
            W = update_W(U, V, T)
            
            error = check_decomposition(U, V, W, T)
            if error < 1e-8:
                converged = True
                break
        
        if converged:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
