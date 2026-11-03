"""
Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition.

We search for a decomposition of the matrix multiplication tensor T[i,j,k] = delta(i,k)
(where A is 4x4, B is 4x5, C is 4x5) as:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

using the trace convention W[k*n + i, r].

Strategy:
1. Start with the schoolbook rank-80 solution.
2. Use a randomized ALS (Alternating Least Squares) style search to try to reduce rank.
   For each target rank R from a small set of candidates, we try to find matrices
   U (16 x R), V (20 x R), W (20 x R) with half-integer entries that exactly
   reconstruct the tensor.

Since exact half-integer solutions are hard to find via continuous optimization,
we use a hybrid approach:
- Use continuous ALS to get close to a solution.
- Then round to nearest half-integers and check exactness.
- If exact, we have a valid decomposition.

However, for 4x4x5, the known minimal rank is not trivially less than 80.
Let's use a smarter initialization and more targeted search.

Actually, let's use a different approach: we'll use the fact that we can
parameterize the problem and use scipy to solve least squares for each factor
in turn, then round.

Given the time constraint, let's implement a practical search:
1. Try several random initializations for rank R.
2. For each, run ALS iterations (continuous).
3. Round to half-integers.
4. Check if the rounded solution is exact.
5. Return the best (lowest rank) exact solution found, or fall back to schoolbook.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for 4x4 * 4x5 -> 4x5."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = 1 if this triple contributes to C[i,k]
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T[i,j,k] = delta(i,k) ... wait, that's not right.
                # The multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
                # Actually, the tensor is: T[i,j,k] = 1 if we multiply A[i,j] by B[j,k] to get C[i,k].
                # So every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p) has T[i,j,k] = 1.
                # Wait, that's the full tensor. Let me reconsider.
                
                # The matrix multiplication tensor T is in R^{n x m x p} where
                # T[i,j,k] = 1 for all valid i,j,k. No, that's not right either.
                
                # Let me think again. We want: C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T has entries T[i,j,k] such that
                # <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
                # But C[i,k] = sum_j A[i,j] B[j,k], so:
                # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (sum_j A[i,j] B[j,k])
                # This means T[i,j,k] = 1 for all i,j,k? No, that would give
                # sum_{i,j,k} A[i,j] B[j,k] which is not sum_{i,k} sum_j A[i,j]B[j,k]... 
                # Actually it IS the same thing. sum_{i,j,k} A[i,j]B[j,k] = sum_i sum_j sum_k A[i,j]B[j,k]
                # = sum_i sum_j A[i,j] (sum_k B[j,k])... no that's wrong.
                
                # Let me be more careful:
                # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] should equal sum_{i,k} C[i,k] = sum_{i,k} sum_j A[i,j] B[j,k]
                # = sum_{i,j,k} A[i,j] B[j,k]
                # So T[i,j,k] = 1 for all i,j,k.
                
                T[i, j, k] = 1
    return T

def _tensor_to_vector(T, n, m, p):
    """Flatten tensor to vector in order matching the factorization convention."""
    # We need T[i,j,k] such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    vec = np.zeros(n * m * p, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = T[i, j, k]
    return vec

def _reconstruct_from_factors(U, V, W, n, m, p, R):
    """Reconstruct tensor from factors."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                T[i, j, k] = s
    return T

def _is_exact(T, tol=1e-9):
    """Check if tensor is exactly the multiplication tensor (all entries 1)."""
    n, m, p = DIMENSIONS
    return np.allclose(T, 1.0, atol=tol)

def _round_to_halfint(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _verify_solution(U, V, W, n, m, p, R, tol=1e-9):
    """Verify that the factors exactly decompose the tensor."""
    T = _reconstruct_from_factors(U, V, W, n, m, p, R)
    return _is_exact(T, tol)

def construct():
    """Construct a rank-minimal exact tensor decomposition."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Build the target tensor
    T_target = _build_tensor()
    
    best_rank = schoolbook_rank
    best_solution = None
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try to find lower-rank solutions
    # Try ranks from a strategic set
    candidate_ranks = [79, 78, 77, 76, 75, 74, 73, 72, 70, 68, 64]
    
    for R in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed i,j: T[i,j,:] = U[i*m+j,:] @ (V[j*p+j*p+p-1? ...])
                # Let me vectorize properly.
                
                # For fixed i,j: T[i,j,k] for all k
                # = sum_r U[i*m+j,r] * (sum over k of V[j*p+k,r] * W[k*n+i,r] for each k)
                # Actually: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Define M_U[i,j,k] = sum_r V[j*p+k,r] * W[k*n+i,r]
                # Then T[i,j,k] = sum_r U[i*m+j,r] * M_U[i,j,k,r]... no.
                
                # Let me think differently. For fixed i,j, we have:
                # T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # Let A[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then T[i,j,k] = sum_r U[i*m+j,r] * A[i,j,k,r]
                # So for fixed i,j: T[i,j,:] = U[i*m+j,:] @ A[i,j,:,:].T  (where A[i,j,:,:] is p x R)
                
                # Build A for all i,j
                A = np.zeros((n, m, p, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A[i, j, k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # For each i,j: U[i*m+j, :] = argmin sum_k (T[i,j,k] - U[i*m+j,:] @ A[i,j,k,:].T)^2
                # This is a least squares problem: A[i,j,:,:].T @ U[i*m+j,:] = T[i,j,:]
                # i.e., (A[i,j,:,:].T @ A[i,j,:,:]) @ U[i*m+j,:] = A[i,j,:,:].T @ T[i,j,:]
                
                for i in range(n):
                    for j in range(m):
                        # A[i,j,:,:] is p x R
                        At = A[i, j, :, :].T  # R x p
                        # Solve: At @ At @ U_row = At @ T[i,j,:]
                        lhs = At @ At  # R x R
                        rhs = At @ T_target[i, j, :]  # R x 1
                        try:
                            U[i * m + j, :] = np.linalg.solve(lhs, rhs)
                        except np.linalg.LinAlgError:
                            U[i * m + j, :] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                
                # Update V: for fixed j, T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed j: T[:,j,k] = sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r]
                # Let B[i,j,k,r] = U[i*m+j,r] * W[k*n+i,r]
                # T[i,j,k] = sum_r B[i,j,k,r] * V[j*p+k,r]
                # For fixed j: T[:,j,:] (n x p) = sum_r B[:,j,:,r] * V[j*p+?,r]... 
                # This is tricky. Let me index differently.
                
                # For fixed j, we have n*p equations: for each (i,k):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let C[i,k,r] = U[i*m+j,r] * W[k*n+i,r]
                # T[i,j,k] = sum_r C[i,k,r] * V[j*p+k,r]
                # For fixed j, unknowns are V[j*p+0, :], ..., V[j*p+p-1, :] (p rows of R)
                # We have n*p equations.
                
                B = np.zeros((n, p, R))
                for i in range(n):
                    for k in range(p):
                        B[i, k, :] = U[i * m + j, :] * W[k * n + i, :]
                
                # Now: for each (i,k): T[i,j,k] = B[i,k,:] @ V[j*p+k,:]
                # This is a coupled system. Let's vectorize.
                # Stack all equations: 
                # For i in 0..n-1, k in 0..p-1: T[i,j,k] = B[i,k,:] @ V[j*p+k,:]
                # This is n*p scalar equations with p*R unknowns.
                
                # Build the system: 
                # Let x = [V[j*p+0,:], V[j*p+1,:], ..., V[j*p+p-1,:]] (p*R vector)
                # For equation (i,k): sum_r B[i,k,r] * x[k*R+r] = T[i,j,k]
                
                lhs_mat = np.zeros((n * p, p * R))
                rhs_vec = np.zeros(n * p)
                idx = 0
                for i in range(n):
                    for k in range(p):
                        lhs_mat[idx, k * R:(k+1) * R] = B[i, k, :]
                        rhs_vec[idx] = T_target[i, j, k]
                        idx += 1
                
                try:
                    sol = np.linalg.lstsq(lhs_mat, rhs_vec, rcond=None)[0]
                    for k in range(p):
                        V[j * p + k, :] = sol[k * R:(k+1) * R]
                except:
                    pass
                
                # Update W: similar to V but with different indexing
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed k: T[i,j,k] = sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
                # For fixed k, unknowns are W[k*n+0,:], ..., W[k*n+n-1,:] (n rows of R)
                # We have m*n equations (for each i,j).
                
                C = np.zeros((m, n, R))
                for j in range(m):
                    for i in range(n):
                        C[j, i, :] = U[i * m + j, :] * V[j * p + k, :]
                
                lhs_mat = np.zeros((m * n, n * R))
                rhs_vec = np.zeros(m * n)
                idx = 0
                for j in range(m):
                    for i in range(n):
                        lhs_mat[idx, i * R:(i+1) * R] = C[j, i, :]
                        rhs_vec[idx] = T_target[i, j, k]
                        idx += 1
                
                try:
                    sol = np.linalg.lstsq(lhs_mat, rhs_vec, rcond=None)[0]
                    for i in range(n):
                        W[k * n + i, :] = sol[i * R:(i+1) * R]
                except:
                    pass
            
            # After ALS, round to half-integers
            U_rounded = _round_to_halfint(U)
            V_rounded = _round_to_halfint(V)
            W_rounded = _round_to_halfint(W)
            
            if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, R, tol=1e-9):
                best_rank = R
                best_solution = (U_rounded, V_rounded, W_rounded)
                found = True
                break
        
        if found:
            # Try even lower ranks
            continue
    
    # Convert to lists for output
    U_list = [[float(x) for x in row] for row in best_solution[0]]
    V_list = [[float(x) for x in row] for row in best_solution[1]]
    W_list = [[float(x) for x in row] for row in best_solution[2]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
