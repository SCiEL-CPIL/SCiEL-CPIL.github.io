"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  The multiplication tensor T_{i,j,k} for multiplying (n x m) * (m x p) has
  trivial rank n*m*p = 63. We attempt to find a lower-rank decomposition.
  
  Approach:
  1. Use rank-revealing QR decomposition on the flattened tensor to identify
     linearly independent output columns.
  2. Use ALS (Alternating Least Squares) to find low-rank decompositions.
  3. Round to half-integers and verify exactness.
"""

import time
import numpy as np
from scipy.linalg import qr

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Precompute target tensor
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
    
    def is_valid_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        # Compute full tensor
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check all on-diagonal terms equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
        
        # Check all off-diagonal terms equal 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    if abs(T[a, b, c]) > tol:
                        return False
        return True
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded
    
    def als_decompose(target_tensor, R, max_iters=50, tol=1e-12):
        """
        Alternating Least Squares for tensor decomposition.
        T = sum_r U[:,r] * V[:,r] * W[:,r]
        """
        # Initialize randomly
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iters):
            # Fix V and W, solve for U
            # For each a: sum_r U[a,r] * (V*W)[:,r] = target[a,:,:].flatten()
            # This is a linear system: U[a,:] * M_VW = target[a,:,:].flatten()
            # where M_VW[r, c] = V[:,r] * W[:,r] flattened
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # So we solve: U[a,:] @ M_VW = T[a,:,:].flatten()
            # where M_VW is (R, m*p*p*n)
            
            # Build M_VW: M_VW[r, idx] = V[b,r] * W[c,r] where idx = b*(p*n) + c
            M_VW = np.zeros((R, m * p * p * n))
            for r in range(R):
                M_VW[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve for U
            for a in range(n * m):
                rhs = target_tensor[a, :, :].flatten()
                # Solve U[a,:] @ M_VW = rhs
                # This is underdetermined if R < m*p*p*n, so use least squares
                try:
                    U[a, :] = np.linalg.lstsq(M_VW.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Fix U and W, solve for V
            M_UW = np.zeros((R, n * m * p * n))
            for r in range(R):
                M_UW[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            for b in range(m * p):
                rhs = np.zeros(n * m * p * n)
                for a in range(n * m):
                    for c in range(p * n):
                        rhs[a * p * n + c] = target_tensor[a, b, c]
                try:
                    V[b, :] = np.linalg.lstsq(M_UW.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Fix U and V, solve for W
            M_UV = np.zeros((R, n * m * m * p))
            for r in range(R):
                M_UV[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            for c in range(p * n):
                rhs = np.zeros(n * m * m * p)
                for a in range(n * m):
                    for b in range(m * p):
                        rhs[a * m * p + b] = target_tensor[a, b, c]
                try:
                    W[c, :] = np.linalg.lstsq(M_UV.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_current - target_tensor) ** 2)
            if error < tol:
                break
        
        return U, V, W, error
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # First, use rank-revealing QR to estimate the minimum rank
    # Flatten tensor to matrix
    T_flat = target_tensor.reshape(n * m, -1)
    Q, R_mat, P = qr(T_flat, pivoting=True)
    
    # Find numerical rank
    diag_R = np.abs(np.diag(R_mat))
    if len(diag_R) > 0:
        rank_estimate = np.sum(diag_R > 1e-10 * diag_R[0])
    else:
        rank_estimate = trivial_rank
    
    # Try ranks from a reasonable upper bound down
    # Based on QR estimate, but ensure we try enough
    max_rank_to_try = max(rank_estimate, 30)
    min_rank_to_try = min(20, max_rank_to_try)
    
    ranks_to_try = list(range(max_rank_to_try, min_rank_to_try - 1, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 1000 + R)
            
            U, V, W, error = als_decompose(target_tensor, R, max_iters=30)
            
            if error < 1e-10:
                # Round to half-integers
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                
                # Verify
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-10):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break # Found valid for this rank, try lower
        
        if best_rank < trivial_rank:
            # Keep trying lower ranks
            pass
    
    # Final verification
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }
