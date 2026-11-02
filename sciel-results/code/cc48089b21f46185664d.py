import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Try to find a lower rank decomposition using a more systematic approach
    # We'll use a randomized approach with local search
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, the convention is W[k*n+i, r], so W index is (k*n+i)
        # Let's reconstruct the matrix multiplication
        # C = A * B where A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = A[i,j] * B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Test with random matrices
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            # U[:, r] gives values for all (i,j)
            # V[:, r] gives values for all (j,k)
            # W[:, r] gives values for all (k,i)
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    C_decomp[i, k] += val
        
        return np.allclose(C, C_decomp, atol=tol)
    
    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using tensor operations."""
        # Build the full tensor T[i,j,k] from random A, B
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        
        # T[i,j,k] = A[i,j] * B[j,k]
        T = np.einsum('ij,jk->ijk', A, B)
        
        # Compute from decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_decomp = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            # For each (i,j,k), add U_r[i,j] * V_r[j,k] * W_r[k,i]
            T_decomp += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, T_decomp, atol=tol)
    
    # Try ranks from rank_sb - 1 down
    # For small improvements, we can try to find rank 83, 82, etc.
    
    # A better approach: use the fact that matrix multiplication tensors have structure
    # For 3x4x7, the schoolbook is 84. Known lower bounds are around 60-70.
    # Let's try a few ranks with random initialization and gradient descent.
    
    def random_decomposition(R, seed=None):
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        return U, V, W
    
    def gradient_step(U, V, W, lr=0.01):
        """One gradient descent step to minimize ||T - T_decomp||^2."""
        # Use a fixed test matrix
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        T = np.einsum('ij,jk->ijk', A, B)
        
        T_decomp = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            T_decomp += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        
        residual = T_decomp - T
        loss = np.sum(residual**2)
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(U.shape[1]):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            
            # dL/dU_r[i,j] = sum_{k} residual[i,j,k] * V_r[j,k] * W_r[k,i]
            dU_r = np.einsum('ijk,jk,ki->ij', residual, V_r, W_r)
            dU[:, r] = dU_r.flatten()
            
            # dL/dV_r[j,k] = sum_{i} residual[i,j,k] * U_r[i,j] * W_r[k,i]
            dV_r = np.einsum('ijk,ij,ki->jk', residual, U_r, W_r)
            dV[:, r] = dV_r.flatten()
            
            # dL/dW_r[k,i] = sum_{j} residual[i,j,k] * U_r[i,j] * V_r[j,k]
            dW_r = np.einsum('ijk,ij,jk->ki', residual, U_r, V_r)
            dW[:, r] = dW_r.flatten()
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, loss
    
    # Try a few ranks
    for R in range(rank_sb - 1, max(rank_sb - 20, 50), -1):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(5):
            if time.time() > deadline:
                    break
                    
            U, V, W = random_decomposition(R, seed=42 + restart)
            
            # Do a few gradient steps
            for step in range(20):
                if time.time() > deadline:
                    break
                U, V, W, loss = gradient_step(U, V, W, lr=0.005)
                if loss < 1e-10:
                    break
                    
            if loss < 1e-9:
                # Verify more carefully
                if verify_decomposition_fast(U, V, W, tol=1e-8):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
