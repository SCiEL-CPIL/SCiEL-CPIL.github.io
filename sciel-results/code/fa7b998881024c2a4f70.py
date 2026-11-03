"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses structured search with fast vectorized verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using NumPy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the full tensor from decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape and compute efficiently
        
        # For each column c, the contribution to T is:
        # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Sum over c
        
        # Vectorized approach:
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We need to compute for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # Let's build it step by step
        # First, for each c, form the outer structure
        # This is equivalent to: sum_c (U_c reshaped to (n,m)) * (V_c reshaped to (m,p)) * (W_c reshaped to (p,n))
        # where * is element-wise after appropriate broadcasting
        
        # More efficient: iterate over rank but vectorize over i,j,k
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)  # (n, m)
            Vc = V[:, c].reshape(m, p)  # (m, p)
            Wc = W[:, c].reshape(p, n)  # (p, n)
            # Uc[i,j] * Vc[j,k] * Wc[k,i] for all i,j,k
            # This is a 3-tensor product
            # (n,m) * (m,p) * (p,n) -> sum over j,k? No, we need all i,j,k
            # Actually: for fixed i,j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = (Uc[:,j] * Vc[j,:] * Wc[:,i]) summed? No.
            # Let's do: outer(Uc, Vc) gives (n,m,m,p), then multiply by Wc
            # Too large. Instead:
            # For each i: Uc[i,:] * Wc[:,i] gives (m,p) which is Uc[i,j]*Wc[k,i] for all j,k
            # Then multiply element-wise by Vc[j,k]
            for i in range(n):
                # Uc[i, j] for all j: shape (m,)
                # Wc[k, i] for all k: shape (p,)
                # Product: Uc[i,:] * Wc[:,i] -> (m, p) via outer
                contrib = np.outer(Uc[i, :], Wc[:, i])  # (m, p), contrib[j,k] = Uc[i,j]*Wc[k,i]
                total[i, :, :] += contrib * Vc  # element-wise multiply by Vc[j,k]
        
        if np.max(np.abs(total - 1.0)) > tol:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p)."""
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
        return U, V, W
    
    def try_random_decomposition(rank, seed=0, sparsity=0.3):
        """Try random decomposition with given sparsity."""
        np.random.seed(seed)
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Random non-zero entries
        mask_U = np.random.random((n*m, rank)) < sparsity
        mask_V = np.random.random((m*p, rank)) < sparsity
        mask_W = np.random.random((p*n, rank)) < sparsity
        
        U[mask_U] = np.random.choice([-1, 1], size=mask_U.sum())
        V[mask_V] = np.random.choice([-1, 1], size=mask_V.sum())
        W[mask_W] = np.random.choice([-1, 1], size=mask_W.sum())
        
        if verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
        return None
    
    def try_block_decomposition(rank):
        """
        Try a block-structured decomposition.
        Split the 9 output columns into groups and share computation.
        """
        if time.time() > deadline - 20:
            return None
        
        # Strategy: Use the fact that B is 4x9 = 4x(3+3+3)
        # For each group of 3 columns, we do a 4x4 * 4x3 multiplication
        # Total schoolbook for three 4x4*4x3: 3 * 4*4*3 = 144
        # But we can share the A matrix computations
        
        # Alternative: Use rank-1 terms more cleverly
        # Each rank-1 term (u, v, w) contributes u[i*m+j]*v[j*p+k]*w[k*n+i] to T[i,j,k]
        
        # Let's try: for each i, we need T[i,j,k] = 1 for all j,k
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Idea: Group by i. For fixed i, define:
        #   u_i[j,c] = U[i*m+j, c]
        #   w_i[k,c] = W[k*n+i, c]
        # Then T[i,j,k] = sum_c u_i[j,c] * V[j*p+k,c] * w_i[k,c]
        
        # This is like: for fixed i, the (j,k) matrix M_i = sum_c u_i[j,c] * V_c[j,k] * w_i[k,c]
        # where V_c is the c-th column of V reshaped to (m,p)
        
        # So M_i = sum_c (u_i[:,c] * w_i[:,c]^T) * V_c  (element-wise)
        # Wait, that's not quite right. Let me re-derive.
        
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed i: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's define for fixed i:
        #   A_i[j,c] = U[i*m+j, c]  (m x rank)
        #   B_i[k,c] = W[k*n+i, c]  (p x rank)
        # Then T[i,j,k] = sum_c A_i[j,c] * V[j*p+k,c] * B_i[k,c]
        
        # We need T[i,j,k] = 1 for all j,k.
        
        # This is a system of m*p = 36 equations for each i, with variables A_i, B_i, and V.
        
        # Let's try a simple structure: assume V has a specific form.
        # If V[j*p+k, c] = a[j,c] * b[k,c] for some a (m x rank) and b (p x rank),
        # then T[i,j,k] = sum_c A_i[j,c] * a[j,c] * b[k,c] * B_i[k,c]
        
        # This doesn't simplify nicely. Let's try another approach.
        
        # Simpler: try to find a decomposition where each rank-1 term is "localized"
        # i.e., only affects a subset of (i,j,k) indices.
        
        # For now, let's do a more systematic random search with better structure
        for attempt in range(100):
            if time.time() > deadline - 15:
                break
            
            seed = 1000 + rank * 100 + attempt
            np.random.seed(seed)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each column c: choose a random subset of i's for W, j's for U and V, k's for V
            for c in range(rank):
                # Choose which i's this term affects
                i_mask = np.random.random(n) < 0.7
                j_mask = np.random.random(m) < 0.7
                k_mask = np.random.random(p) < 0.7
                
                for i in range(n):
                    if i_mask[i]:
                        for j in range(m):
                            if j_mask[j]:
                                U[i*m+j, c] = np.random.choice([-1, 1])
                
                for j in range(m):
                    for k in range(p):
                        if j_mask[j] and k_mask[k]:
                            V[j*p+k, c] = np.random.choice([-1, 1])
                
                for k in range(p):
                    for i in range(n):
                        if k_mask[k] and i_mask[i]:
                            W[k*n+i, c] = np.random.choice([-1, 1])
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find lower rank decompositions
    best_result = None
    
    # Try ranks below schoolbook with various strategies
    for rank in range(130, schoolbook_rank + 1, -2):
        if time.time() > deadline - 20:
            break
        
        # Try random decomposition
        result = try_random_decomposition(rank, seed=rank*1000, sparsity=0.4)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
        
        # Try block decomposition
        result = try_block_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
