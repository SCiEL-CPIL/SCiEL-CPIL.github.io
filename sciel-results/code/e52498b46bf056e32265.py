"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
  1. Start from the schoolbook (rank 80) decomposition.
  2. Use a structured search to find lower-rank decompositions by exploiting
     the algebraic structure of matrix multiplication tensors.
  3. Use Riemannian optimization on the Stiefel manifold to refine candidate
     decompositions and check for exactness with integer/half-integer entries.

  For 4x4 * 4x5, the known upper bound is 80 (schoolbook). The theoretical
  lower bound is n*m*p / max(n,m,p) = 80/5 = 16, but practical bounds are
  much higher. We aim to find any rank < 80 decomposition.

  Given the 150s time limit, we use a hybrid approach:
  - Try known constructions (Strassen-like block decompositions)
  - Use random initialization + Riemannian gradient descent to find low-rank
    approximations, then round to half-integers and verify exactness.
"""

import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For MM: T[i,j,k] should be 1 if (i,j,k) corresponds to a valid term
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    # where the output entry (i,k) = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The tensor we need to decompose has entries:
    # T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # This is the all-ones tensor of shape (n, m, p)
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # (A ⊗ B ⊗ C) where C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # Yes, it's the all-ones tensor of shape (n, m, p).
    
    # Schoolbook: rank = n*m*p = 4*4*5 = 80
    
    # First, return schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "_rank": rank_school
    }
    
    # Target: find rank < 80
    # Theoretical lower bound for MM(4,4,5):
    # Using Strassen's bound: r >= 3 * max(n,m,p)^(log2(7)/3) roughly
    # But let's just search for better decompositions.
    
    # Approach: Use the fact that the all-ones tensor can be decomposed
    # using rank-1 tensors. We want sum_c u_c ⊗ v_c ⊗ w_c = 1_{n×m×p}
    
    # Try Riemannian optimization:
    # Parameterize U, V, W as matrices with orthonormal columns (Stiefel manifold)
    # Then scale to get the right magnitudes.
    
    # Actually, for exact decomposition with integer/half-integer entries,
    # we need a different approach. Let's try to find a decomposition
    # where the entries are from {0, ±1/2, ±1, ±3/2, ...}
    
    # Key insight: The all-ones tensor of shape (n,m,p) can be written as
    # sum over a subset of rank-1 tensors if we allow negative entries.
    
    # Let's try a structured approach using block decomposition.
    # For 4x4 * 4x5, we can use:
    # - 2x2 Strassen blocks: each 2x2*2x2 block needs rank 7 instead of 8
    # - But 4x4 * 4x5 doesn't decompose cleanly into 2x2 blocks for the 5 dimension
    
    # Let's try random search with Riemannian refinement:
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that U, V, W exactly decompose the all-ones tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check each entry
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(x):
        """Round to nearest half-integer."""
        return round(2*x) / 2.0
    
    def try_rank(r, max_iters=500, lr=0.01, seed=None):
        """Try to find a rank-r decomposition using Riemannian optimization."""
        if seed is not None:
            np.random.seed(seed)
        
        n, m, p = DIMENSIONS
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        
        # Initialize with random Stiefel matrices
        def random_stiefel(d, r):
            """Generate a random d x r matrix with orthonormal columns."""
            A = np.random.randn(d, r)
            Q, R = np.linalg.qr(A)
            # Ensure unique QR decomposition
            d_sign = np.diag(R)
            ph = np.sign(d_sign)
            ph[ph == 0] = 1
            Q = Q * ph
            return Q
        
        U = random_stiefel(dim_U, r)
        V = random_stiefel(dim_V, r)
        W = random_stiefel(dim_W, r)
        
        # We need to find scalings alpha_c, beta_c, gamma_c such that
        # sum_c alpha_c * U[:,c] ⊗ beta_c * V[:,c] ⊗ gamma_c * W[:,c] = 1
        # This is equivalent to finding U, V, W directly.
        
        # Use gradient descent on the loss:
        # L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        # Precompute for efficiency
        def compute_loss_and_grad(U, V, W):
            n, m, p = DIMENSIONS
            r = U.shape[1]
            loss = 0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = val - 1.0
                        loss += err * err
                        for c in range(r):
                            du = V[j*p+k, c] * W[k*n+i, c]
                            dv = U[i*m+j, c] * W[k*n+i, c]
                            dw = U[i*m+j, c] * V[j*p+k, c]
                            grad_U[i*m+j, c] += 2 * err * du
                            grad_V[j*p+k, c] += 2 * err * dv
                            grad_W[k*n+i, c] += 2 * err * dw
            
            return loss, grad_U, grad_V, grad_W
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            loss, gU, gV, gW = compute_loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Riemannian gradient step (simplified: Euclidean step + projection)
            U = U - lr * gU
            V = V - lr * gV
            W = W - lr * gW
            
            # Project back to Stiefel manifold
            for mat in [U, V, W]:
                Q, R = np.linalg.qr(mat)
                d_sign = np.diag(R)
                ph = np.sign(d_sign)
                ph[ph == 0] = 1
                mat[:] = Q * ph
        
        # Check if we found a good solution
        if best_loss < 1e-8:
            # Round to half-integers
            U_r = np.array([[round_to_half_int(x) for x in row] for row in best_U])
            V_r = np.array([[round_to_half_int(x) for x in row] for row in best_V])
            W_r = np.array([[round_to_half_int(x) for x in row] for row in best_W])
            
            if verify_decomposition(U_r, V_r, W_r, tol=1e-9):
                return r, U_r, V_r, W_r
        
        return None, None, None, None
    
    # Try various ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something
    # For 4x4*4x5: dim_U=16, dim_V=20, dim_W=20
    # Minimum rank is at least max(16,20,20) = 20? No, that's not a valid bound.
    # The rank must be at least the max of the matrix dimensions in some sense.
    # Actually, the rank must be at least the maximum of n, m, p in some cases.
    # Let's just try from 20 up to 79.
    
    found = False
    for r in range(20, 80):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, max_iters=200, lr=0.005, seed=r)
        if result[0] is not None:
            rank, U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
                "_rank": rank
            }
            found = True
            break
    
    if not found:
        # Try a few more seeds for the schoolbook rank or slightly lower
        for r in [79, 78, 77, 76, 75]:
            if time.time() > deadline - 3:
                break
            for seed in [42, 123, 456]:
                result = try_rank(r, max_iters=300, lr=0.003, seed=seed)
                if result[0] is not None:
                    rank, U, V, W = result
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist(),
                        "_rank": rank
                    }
                    found = True
                    break
            if found:
                break
    
    return {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
