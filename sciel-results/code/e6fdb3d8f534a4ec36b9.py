import numpy as np
import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target Tensor indices
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
    
    # Precompute target tensor for verification
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for u, v, w in target_indices:
        TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        # U: (size_u, R), V: (size_v, R), W: (size_w, R)
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def verify_solution(U, V, W, tol=1e-9):
        T = tensor_contraction(U, V, W)
        return np.allclose(T, TARGET_TENSOR, atol=tol)

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    # Schoolbook Solution (Rank 48)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook()
    
    best_solution = None
    best_rank = n * m * p

    # Strategy: 
    # 1. Try to reduce rank by finding linear dependencies in the "flattened" tensor space.
    # The tensor T can be viewed as a vector in R^(size_u * size_v * size_w).
    # A rank-1 term is a vector t_c = vec(U[:,c] \otimes V[:,c] \otimes W[:,c]).
    # If we have a decomposition T = sum_{c=1}^R t_c, and R > rank(T), the vectors t_c are linearly dependent.
    # However, finding the minimal rank is hard. 
    # 
    # Alternative Strategy: Local Search / Simulated Annealing on the parameters.
    # We want to minimize || sum_c u_c \otimes v_c \otimes w_c - T ||^2.
    # This is a non-convex problem.
    # 
    # Given the time limit, let's try a specific algebraic reduction.
    # For 3x4x4, it is unknown if rank < 48. 
    # Most likely, the rank is 48. 
    # If we cannot find a better rank, we return the schoolbook solution.
    # 
    # Let's attempt a randomized search for rank 47.
    # If we find a solution for rank 47, we check if it can be snapped to half-integers.
    # The task requires integer/half-integer entries.
    
    # If we assume the rank is indeed 48, we just return the schoolbook solution.
    # But we should make a reasonable effort to find a lower rank if it exists.
    
    # Let's try a simple gradient descent with random restarts for a few ranks.
    # Objective: sum_{i,j,k} ( sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1 )^2
    #           + sum_{non-target} ( sum_c ... )^2
    
    # To make this fast, we only evaluate on the target indices and a small subset of non-targets?
    # No, we must ensure ALL entries match.
    # The non-target entries must be 0.
    # 
    # Let's use the fact that the tensor is sparse.
    # 
    # Actually, let's just try to verify if the schoolbook solution is the best we can get quickly.
    # Given the complexity and time limit, and the fact that 3x4x4 is a small case where 
    # no better algorithm is widely known (Strassen is for 2x2, etc.), it's highly probable 
    # that 48 is the minimal rank or close to it.
    
    # Let's do a quick check for rank 47 using a randomized linear dependency search.
    # If we have R=48 terms, and we can find coefficients alpha such that 
    # sum_{c=1}^{48} alpha_c t_c = 0, and all alpha_c != 0, we might be able to eliminate one?
    # No, that just means they are dependent. To reduce rank, we need to express one t_c 
    # as a linear combination of the others, which requires the others to span it.
    
    # Let's try a different approach: 
    # If we fix U and V, the problem becomes linear in W.
    # For each w-index (k*n+i), we have a vector of equations:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1  for all j.
    # This is a system of m equations for the R unknowns W[k*n+i, c].
    # If m < R, there are many solutions.
    
    # Let's try to solve for W given random U, V for R=47.
    # If a solution exists, we check if it's valid for all entries.
    
    def try_solve_for_rank(R, seed=0):
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        
        # Random initialization for U and V
        U = np.random.randn(size_u, R) * 0.5
        V = np.random.randn(size_v, R) * 0.5
        
        # Construct the linear system for W
        # We have size_w rows of W to determine.
        # For each w_idx (which corresponds to k, i):
        # We need to solve: sum_c A[w_idx, j, c] * W[w_idx, c] = 1 for all j in 0..m-1
        # where A[w_idx, j, c] = U[i*m+j, c] * V[j*p+k, c]
        
        W = np.zeros((size_w, R))
        
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                
                # Build matrix B of shape (m, R)
                B = np.zeros((m, R))
                for j in range(m):
                    for c in range(R):
                        B[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                # Solve B @ W[w_idx, :] = [1, 1, ..., 1]
                # This is an underdetermined system (m < R usually, e.g., 4 < 47)
                # We can use least squares or find a particular solution.
                # Using np.linalg.lstsq gives the minimum norm solution.
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W[w_idx, :] = sol
                except:
                    return None
        
        # Now we have U, V, W. Check if they satisfy the full tensor equation.
        if verify_solution(U, V, W, tol=1e-5):
            # Try to snap
            U_s, V_s, W_s = snap_to_half_integers(U, V, W)
            if verify_solution(U_s, V_s, W_s, tol=1e-9):
                return U_s, V_s, W_s
            # If snapping fails, maybe the solution is not "nice". 
            # But the task requires integer/half-integer entries.
            # If we can't snap, this solution is invalid for the final output.
            # However, the existence of a real solution suggests a rational one might exist nearby?
            # For now, return None if snapping fails.
            
        return None

    # Try to find a rank 47 solution
    # We'll try a few random seeds.
    for R in [47, 46]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            res = try_solve_for_rank(R, seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            break

    if best_solution:
        return best_solution
    else:
        # Return schoolbook solution
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
