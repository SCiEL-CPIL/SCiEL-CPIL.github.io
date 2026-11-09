"""Matrix multiplication 2x3 by 3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]
import numpy as np
import random
import time

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    T_target = _build_tensor(n, m, p)
    T_recon = np.zeros_like(T_target)
    for c in range(U.shape[1]):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.max(np.abs(T_recon - T_target)) < tol

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_ranked_decomp(n, m, p, rank, max_time=120.0):
    """
    Try to find a valid decomposition of the given rank using iterative
    alternating least squares / tensor decomposition approaches.
    """
    T_target = _build_tensor(n, m, p)
    T_flat = T_target.reshape(n*m*m*p*p*n)
    deadline = time.time() + max_time - 5
    
    # Multiple random restarts
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Alternating minimization
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Update U: for each column c, minimize ||T - sum_c U_c ⊗ V_c ⊗ W_c||
            # Given V and W, the optimal U is:
            # U[:,c] = (V_c ⊗ W_c)^+ T_flat  (for each c, but coupled)
            # Actually, we can do block coordinate descent.
            
            # For fixed V, W: solve for U
            # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # Reshape: T_flat = sum_c (W[:,c] ⊗ V[:,c] ⊗ U[:,c])
            # This is a linear system in U if we vectorize properly.
            
            # Let's use a different approach: treat it as a matrix problem
            # For each (i,j) pair in U, the contribution to T is:
            # T[i,j,:,:] += U[i,j] * V[:,c] * W[:,c]  (outer product)
            
            # Actually, let's just do gradient-free alternating projections.
            # For fixed V and W, the optimal U column c is:
            # U[:,c] = argmin_U ||T_flat - sum_c' (W[:,c'] ⊗ V[:,c'] ⊗ U[:,c'])||^2
            # Taking derivative w.r.t. U[:,c]:
            # (W[:,c] ⊗ V[:,c])^T (T_flat - sum_c' ...) = 0
            # So: (W[:,c] ⊗ V[:,c])^T U[:,c] = (W[:,c] ⊗ V[:,c])^T (T_flat - sum_{c'≠c} ...)
            
            # This is getting complex. Let's try a simpler approach:
            # Use the fact that for small problems, we can try to solve
            # the linear system directly.
            
            # Build the linear system for U given V, W
            # T_flat = sum_c (W[:,c] ⊗ V[:,c] ⊗ U[:,c])
            # Let M_c = W[:,c] ⊗ V[:,c] (length n*m*m*p*p*n... no)
            # Actually W has length p*n, V has length m*p
            # W[:,c] ⊗ V[:,c] has length (p*n)*(m*p)
            # And U[:,c] has length n*m
            # The product (W[:,c] ⊗ V[:,c] ⊗ U[:,c]) has length (p*n)*(m*p)*(n*m)
            # But T_flat has length (n*m)*(m*p)*(p*n) = n*m*m*p*p*n
            # These are the same! Just different orderings.
            
            # Let's be more careful with the tensor structure.
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
            
            # For fixed V and W, we want to find U such that:
            # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k] for all i,j,k
            
            # This is a linear system in U. Let's write it as:
            # For each (i,j,k): sum_c U[i,c] * (V[j,c] * W[k,c]) = T[i,j,k]
            
            # Group by i: for each i, we have m*p*p*n equations in rank unknowns (U[i,0..rank-1])
            # But the equations couple different i's through the same c.
            
            # Actually, let's just do a simple iterative method:
            # 1. For fixed V, W: solve for U via least squares
            #    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            #    For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
            #    So: T[i].reshape(m*p, p*n) = sum_c U[i,c] * outer(V[:,c], W[:,c])
            #    This is a linear system: T[i].flat = A_i @ U[i,:]
            #    where A_i[j*m*p + k, c] = V[j,c] * W[k,c]
            
            # Build A for all i (same structure, so just one A)
            # A has shape (m*p*p*n, rank)
            # A[j*(p*n) + k, c] = V[j,c] * W[k,c]
            
            A = np.zeros((m*p*p*n, rank))
            for j in range(m*p):
                for k in range(p*n):
                    A[j*(p*n) + k, :] = V[j, :] * W[k, :]
            
            # For each i, solve: A @ U[i,:] = T[i,:,:].flat
            T_reshaped = T_target.reshape(n*m, m*p*p*n)
            U_new = np.zeros((n*m, rank))
            for i in range(n*m):
                # Least squares solve
                U_new[i, :] = np.linalg.lstsq(A, T_reshaped[i, :], rcond=None)[0]
            
            # Update V similarly
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For each j: T[:,j,:] = sum_c V[j,c] * (U[:,c] * W[:,c])
            # T[:,j,:].flat = B @ V[j,:]
            # B[i*(p*n) + k, c] = U[i,c] * W[k,c]
            
            B = np.zeros((n*m*p*n, rank))
            for i in range(n*m):
                for k in range(p*n):
                    B[i*(p*n) + k, :] = U_new[i, :] * W[k, :]
            
            V_new = np.zeros((m*p, rank))
            for j in range(m*p):
                V_new[j, :] = np.linalg.lstsq(B, T_target[:, j, :].reshape(n*m, p*n).reshape(-1), rcond=None)[0]
            
            # Update W similarly
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For each k: T[:,:,k] = sum_c W[k,c] * (U[:,c] * V[:,c])
            # T[:,:,k].flat = C @ W[k,:]
            # C[i*(m*p) + j, c] = U[i,c] * V[j,c]
            
            C = np.zeros((n*m*m*p, rank))
            for i in range(n*m):
                for j in range(m*p):
                    C[i*(m*p) + j, :] = U_new[i, :] * V_new[j, :]
            
            W_new = np.zeros((p*n, rank))
            for k in range(p*n):
                W_new[k, :] = np.linalg.lstsq(C, T_target[:, :, k].reshape(n*m, m*p).reshape(-1), rcond=None)[0]
            
            U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.zeros_like(T_target)
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound up to schoolbook_rank
    # Start with a few promising ranks
    for rank in range(lower_bound, schoolbook_rank + 1):
        result = _try_ranked_decomp(n, m, p, rank, max_time=100.0)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook_decomp(n, m, p)
    rank = schoolbook_rank
    U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
