"""Program for matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin for 120s limit
    
    # First, get the trivial decomposition as fallback
    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = trivial_decomposition()
    
    # The matrix multiplication tensor T is defined by:
    # T[(i,j,k), (i',j',k')] = delta[i,i'] * delta[j,j'] * delta[k,k']
    # Actually, for matrix multiplication (A is n x m, B is m x p, C is n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,k,j] = 1 if we're computing the (i,k) entry using j
    # More precisely: T is a 3-tensor of shape (n*p, m, m) or similar
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor for (n x m) * (m x p) has shape (n*p, m*m)
    # Actually, it's a trilinear form: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # The tensor T has shape (n*m, m*p, p*n) where:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    
    # The rank of this tensor is the minimum r such that:
    # T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    
    # For 3x3 * 3x3, the rank is 23 (Strassen).
    # For 3x3 * 3x7, we can think of it as 3x3 * 3x3 + 3x3 * 3x4
    # The rank of 3x3 * 3x4 is at most 3*3*4 = 36, but likely less.
    # Actually, the rank of (n x m) * (m x p) is at most n*m*p, and we want to minimize it.
    
    # Let's try to find a decomposition using random initialization and ALS.
    
    def build_tensor():
        """Build the full matrix multiplication tensor."""
        # T has shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j      # row of A
                    b = j * p + k      # col of B  
                    c = k * n + i      # row of C (trace convention W[k*n+i])
                    T[a, b, c] = 1
        return T
    
    T = build_tensor()
    
    def matmul_als(U, V, W, rank):
        """Check if U, V, W exactly decompose T."""
        # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a large check, so let's be smart about it
        # For each (a,b), the vector over c should match
        # Actually, let's just check a few random entries for speed, then full check at the end
        return True  # placeholder
    
    def check_decomposition(U, V, W, rank, T):
        """Check if the decomposition is exact."""
        # For efficiency, check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - T[a, b, c]) > 1e-10:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank, T):
        """Faster check using tensor operations."""
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is the mode-wise product
        # For each r, we have an outer product U[:,r] outer V[:,r] outer W[:,r]
        # Sum over r
        # Use einsum
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(reconstructed, T, atol=1e-10):
            return False
        return True
    
    def try_random_als(rank, max_iters=50, tol=1e-8):
        """Try to find a rank-`rank` decomposition using random ALS."""
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W
            # For each a, U[a,:] should minimize sum_b,c (T[a,b,c] - sum_r' U[a,r']*V[b,r']*W[c,r'])^2
            # The optimal U[a,:] is found by solving a least squares problem
            # Let M_a be the matrix with M_a[r, (b,c)] = V[b,r]*W[c,r]
            # Then U[a,:] = (M_a^T M_a)^{-1} M_a^T T[a,:,:].ravel()
            
            # Precompute V*W for all (b,c)
            VW = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, rank)
            
            for a in range(n * m):
                # M_a has shape (rank, m*p * p*n)
                M_a = VW[:, :, :].reshape(m * p, p * n, rank).transpose(2, 0, 1).reshape(rank, m * p * p * n)
                rhs = T[a, :, :].reshape(-1)
                # Solve M_a @ u = rhs
                try:
                    U[a, :] = np.linalg.lstsq(M_a, rhs, rcond=None)[0]
                except:
                    U[a, :] = np.random.randn(rank)
            
            # Update V similarly
            UW = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, rank)
            for b in range(m * p):
                M_b = UW[:, :, :].reshape(n * m, p * n, rank).transpose(2, 0, 1).reshape(rank, n * m * p * n)
                rhs = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(M_b, rhs, rcond=None)[0]
                except:
                    V[b, :] = np.random.randn(rank)
            
            # Update W
            UV = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, rank)
            for c in range(p * n):
                M_c = UV[:, :, :].reshape(n * m, m * p, rank).transpose(2, 0, 1).reshape(rank, n * m * m * p)
                rhs = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(M_c, rhs, rcond=None)[0]
                except:
                    W[c, :] = np.random.randn(rank)
            
            # Check convergence
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(reconstructed - T) / max(1, np.linalg.norm(T))
            if err < tol:
                break
        
        # Try rounding to integers/half-integers
        # First check if current is good enough
        if check_decomposition_fast(U, V, W, rank, T):
            return U, V, W
        
        # Try rounding
        for factor in [1, 2]:
            U_r = np.round(U * factor) / factor
            V_r = np.round(V * factor) / factor
            W_r = np.round(W * factor) / factor
            if check_decomposition_fast(U_r, V_r, W_r, rank, T):
                return U_r, V_r, W_r
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_decomposition_fast(U_i, V_i, W_i, rank, T):
            return U_i, V_i, W_i
        
        return None
    
    # Try various ranks, starting from lower
    # Known: rank(3x3 * 3x3) = 23
    # rank(3x3 * 3x7) >= rank(3x3 * 3x3) = 23 (since 3x7 contains 3x3 as a subproblem)
    # Upper bound: 3*3*7 = 63
    # Let's try ranks from 23 upward
    
    for rank in range(23, 64):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each rank
        max_restarts = 5 if rank < 40 else 2
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = try_random_als(rank, max_iters=30)
            if result is not None:
                U, V, W = result
                # Verify
                if check_decomposition_fast(U, V, W, rank, T):
                    # Found a valid decomposition
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    break
        
        if best_rank == rank:
            break
    
    # Convert to lists for the return interface
    def to_list(arr):
        return [[float(val) for val in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
