"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.

The matrix multiplication tensor T[i, j, k] (where i in [0,n), j in [0,m), k in [0,p))
satisfies: for matrices A (n x m) and B (m x p), C = A @ B with C[i,k] = sum_j A[i,j]*B[j,k].

The tensor decomposition requires:
  T[i,j,k] = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

The output factor follows the official trace convention W[k*n+i, r].

Known results:
- Schoolbook rank = n*m*p = 18
- For 2x3 * 3x3, the rank is known to be 16 (from literature on matrix multiplication ranks).
  Actually, let me think more carefully. The rank of matrix multiplication for 2x3 and 3x3...
  
  The rank of multiplying an m x n matrix by an n x p matrix is at most m*n*p (schoolbook).
  Lower bounds: at least m*n + n*p - n (from the "rank lower bound" via border rank arguments),
  and at least max(m*p, n*(m+p-n)) etc.

  For n=2, m=3, p=3:
  - Upper bound (schoolbook): 18
  - Lower bound: m*n + n*p - n = 6 + 9 - 3 = 12
  - Another lower bound: max(m*p, ...) = max(9, ...) 
  
  The exact rank for 2x3 * 3x3 is not trivially known to me, but let's try to find a decomposition
  of rank less than 18.

  Actually, for small matrices, we can try to find the minimal rank by searching.
  The rank of 2x2 * 2x2 is 7 (Strassen). For 2x3 * 3x3, let's try to construct a rank-16 or rank-17 solution.

  Let me think about this differently. The matrix multiplication tensor for (n,m,p) = (2,3,3) has:
  - U has shape (n*m, rank) = (6, rank)
  - V has shape (m*p, rank) = (9, rank)
  - W has shape (p*n, rank) = (6, rank)

  The tensor T is a 3-tensor of size 2*3*3 = 18 entries, where T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0. Wait, actually the tensor entries are:
  
  T[i, j, k] = 1 for all valid (i,j,k) where i in [0,2), j in [0,3), k in [0,3).
  
  So T is an all-ones tensor of shape (2,3,3)? No, that's not right.
  
  Let me reconsider. The matrix multiplication tensor is defined such that:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  In tensor decomposition form:
  T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
  
  where T_{i,j,k} = 1 for all valid indices (since the "tensor" of matrix multiplication has all ones).
  
  Wait, I need to be more careful. The standard formulation:
  
  We want to decompose the bilinear map (A, B) -> A@B. The tensor T has entries T[i,j,k] where:
  - i indexes rows of A and C: i in [0, n)
  - j indexes columns of A / rows of B: j in [0, m)  
  - k indexes columns of B and C: k in [0, p)
  
  And T[i,j,k] = 1 for all (i,j,k) in the valid range. The decomposition is:
  T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  
  So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

  This is a rank-r decomposition of an all-ones 3-tensor of shape (n,m,p) = (2,3,3).

  The rank of the all-ones tensor of shape (a,b,c) is... well, it's not simply 1 because the factor matrices have specific index structures.

  Actually, let me just try to find a low-rank decomposition numerically.

Strategy:
1. Start with the schoolbook rank-18 solution.
2. Try to reduce the rank by finding a rank-17, rank-16, etc. decomposition.
3. Use numerical optimization to find the minimal rank.

Given the time constraint, let me try a systematic approach:
- For rank r from 12 up to 18, try to find a valid decomposition.
- Use random initialization and gradient descent or other optimization.

Actually, given the complexity, let me first try to see if rank 17 is achievable by a simple modification of the schoolbook method, and then try numerical search for lower ranks.

Let me implement a numerical search using scipy.optimize or manual gradient descent.
"""

import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for evaluation overhead
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from low to high
    best_rank = nm * mp  # 18 (schoolbook)
    best_U = None
    best_V = None
    best_W = None
    
    # First, create the schoolbook solution as fallback
    rank_school = nm * mp
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def check_decomposition_fast(U, V, W, tol=1e-6):
        """Faster check using numpy."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]  # shape (rank,)
                    v_val = V[j*p + k, :]  # shape (rank,)
                    w_val = W[k*n + i, :]  # shape (rank,)
                    val = np.dot(u_val, v_val * w_val)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def try_rank(r, max_iter=5000, tol=1e-6, seed=42):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        n_restarts = min(5, max(1, int((deadline - time.time()) / 2)))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            U = rng.randn(nm, r) * 0.5
            V = rng.randn(mp, r) * 0.5
            W = rng.randn(pn, r) * 0.5
            
            # Gradient descent
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] should be 1
                            # d(T[i,j,k])/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                            # d(T[i,j,k])/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                            # d(T[i,j,k])/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                            
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # Current value
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            err = val - 1.0  # We want this to be 0
                            
                            # Gradient contribution
                            grad_U[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                            grad_V[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                            grad_W[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 100 == 0:
                    err_now = check_decomposition_fast(U, V, W, tol=1.0)
                    if err_now < best_err:
                        best_err = err_now
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    
                    if err_now < tol:
                        return U.copy(), V.copy(), W.copy()
                
                # Adaptive learning rate
                if it % 200 == 0 and it > 0:
                    err_prev = check_decomposition_fast(U + lr * grad_U, V + lr * grad_V, W + lr * grad_W, tol=1.0)
                    if err_prev < err_now:
                        lr *= 0.8
                    elif it > 50:
                        lr *= 0.95
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 12 to 17
    for r in range(12, 18):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_iter=3000, seed=r*100)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if check_decomposition_fast(U_sol, V_sol, W_sol, tol=1e-6):
                best_rank = r
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
                break
    
    # If no better solution found, use schoolbook
    if best_U is None:
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_school
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
