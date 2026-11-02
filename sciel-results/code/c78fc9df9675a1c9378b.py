"""Tensor decomposition search for 3x3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T of shape (n, m, p) where T[i,j,k]=1."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i*m + j, :]
                v_vec = V[j*p + k, :]
                w_vec = W[k*n + i, :]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank decomposition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _try_improve_rank(n, m, p, current_rank, time_budget, start_time):
    """Try to find a decomposition of rank lower than current_rank."""
    best_rank = current_rank
    best_U, best_V, best_W = None, None, None
    
    return best_rank, best_U, best_V, best_W

def _structured_decomposition_3x3x4():
    """
    Attempt to find a rank-reduced decomposition for 3x3x4 using algebraic structure.
    
    Strategy: The 3x3x4 multiplication can be viewed as 3x3x3 plus one extra output column.
    We'll try to construct a decomposition with rank < 36 by exploiting linearity.
    
    For C = A * B where A is 3x3, B is 3x4:
    C[i,k] = sum_j A[i,j] B[j,k] for i=0,1,2 and k=0,1,2,3
    
    We can write this as:
    For k=0,1,2: C[:,k] = A * B[:,k] (standard 3x3 multiplication)
    For k=3: C[:,3] = A * B[:,3] (another 3x3 multiplication)
    
    So we need two 3x3 multiplications. If we use Strassen's 7-term algorithm for each,
    we'd get rank 14. But we can do better by sharing terms.
    
    Let's try a direct algebraic approach:
    We want to find U, V, W of rank r such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This is a system of n*m*p = 36 equations in r * (n*m + m*p + p*n) = r * (9 + 12 + 12) = 33r unknowns.
    For r = 22, we have 726 unknowns and 36 equations - highly overdetermined but the structure
    might allow solutions.
    
    Let's try a gradient-based approach to minimize the error.
    """
    n, m, p = 3, 3, 4
    
    # Start with a reasonable rank and try to optimize
    for target_rank in [22, 24, 26, 28, 30]:
        if time.time() > 120:
            break
            
        # Initialize with random values
        rng = np.random.RandomState(42)
        
        for attempt in range(20):
            if time.time() > 120:
                break
                
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Define the objective function
            def objective(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_vec = U_flat[i*m + j, :]
                            v_vec = V_flat[j*p + k, :]
                            w_vec = W_flat[k*n + i, :]
                            val = np.sum(u_vec * v_vec * w_vec)
                            error += (val - 1.0) ** 2
                return error
            
            # Flatten parameters
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Try L-BFGS-B optimization
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    # Extract optimized parameters
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Verify
                    if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                        return target_rank, U_opt, V_opt, W_opt
                        
            except Exception as e:
                continue
        
    return None

def _ladder_reduction_3x3x4():
    """
    Apply ladder/peeling reduction technique.
    
    Idea: Start from the known rank-36 schoolbook and try to eliminate terms
    by finding linear dependencies.
    
    The schoolbook has terms for each (i,j,k). We can try to combine terms
    where the same (i,j) or (j,k) or (k,i) pairs appear.
    
    For 3x3x4, let's think of it as:
    - 3x3x3 part (27 terms) + 3x3x1 part (9 terms) = 36 terms
    
    Can we reduce the 3x3x3 part? Strassen reduces 3x3x3 to rank 7.
    If we use Strassen for the first 3 columns and add 9 terms for the 4th,
    we get rank 7 + 9 = 16. But this might not work because the Strassen
    terms for different output columns need to be compatible.
    
    Let's try a different approach: use a known rank-13 decomposition for 3x3x3
    (if it exists) and extend it.
    """
    n, m, p = 3, 3, 4
    
    # Try to construct a decomposition using block structure
    # A is 3x3, B is 3x4
    # C = A * B is 3x4
    
    # We can write B as [B1 | B2] where B1 is 3x3 and B2 is 3x1
    # C = [A*B1 | A*B2]
    
    # If we have a rank-r decomposition for 3x3x3, we can reuse it for A*B1
    # and add r' terms for A*B2
    # Total rank = r + r'
    
    # For A*B2 (3x3 * 3x1 = 3x1), we need at most 9 terms (schoolbook)
    # But we might do better.
    
    # Let's try to find a minimal decomposition for 3x3x1
    # C[i,0] = sum_j A[i,j] B[j,0]
    # This is just a dot product, so we need 3 terms if we use:
    # U[i,j,r] = A[i,j], V[j,0,r] = B[j,0], W[0,i,r] = 1 for r=0,1,2
    # But this doesn't work because W depends on i.
    
    # Actually, for 3x3x1:
    # sum_r U[i*m+j, r] * V[j*p+0, r] * W[0*n+i, r] = 1
    # We can set:
    # U[i,j,0] = 1, V[j,0,0] = 1, W[0,i,0] = 1 for all i,j -> gives 1 for all i,j
    # But we need sum_j A[i,j] B[j,0], not 1.
    
    # The decomposition is for the tensor, not the actual multiplication.
    # The tensor T[i,j,k] = 1 for all i,j,k in the 3x3x1 case.
    # So we need sum_r U[i,j,r] V[j,0,r] W[0,i,r] = 1 for all i,j.
    
    # This is a 3x3 matrix of equations:
    # For each (i,j): sum_r U[i,j,r] V[j,0,r] W[0,i,r] = 1
    
    # We can use rank 3:
    # U[i,j,0] = 1, V[j,0,0] = 1, W[0,i,0] = 1 -> sum = 1*1*1 = 1 ✓
    # But we need 3 different terms? No, one term suffices if we set all to 1.
    
    # Wait, that can't be right. Let me reconsider.
    
    # The tensor for 3x3x1 multiplication is T[i,j,0] = 1 for i=0,1,2, j=0,1,2.
    # A rank-1 decomposition would be:
    # U[i,j,0] * V[j,0,0] * W[0,i,0] = 1 for all i,j
    # This requires U[i,j,0] = a[i,j], V[j,0,0] = b[j], W[0,i,0] = c[i]
    # such that a[i,j] * b[j] * c[i] = 1 for all i,j.
    # This is possible: set a[i,j] = 1, b[j] = 1, c[i] = 1.
    
    # So 3x3x1 has rank 1!
    
    # Similarly, 3x3x3 might have rank less than 27.
    # The best known rank for 3x3x3 is 13 (or 14?).
    
    # Let's assume we can find a rank-r decomposition for 3x3x3.
    # Then 3x3x4 = 3x3x3 + 3x3x1 would have rank r + 1.
    
    # If r = 13, then rank = 14.
    # If r = 14, then rank = 15.
    
    # Let's try to search for a 3x3x3 decomposition first.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Schoolbook Decomposition (Rank 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Verify schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try structured decomposition
    result = _structured_decomposition_3x3x4()
    if result is not None:
        rank_found, U_found, V_found, W_found = result
        if rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = U_found, V_found, W_found
            print(f"Found rank {rank_found} decomposition")

    # Try ladder reduction
    result = _ladder_reduction_3x3x4()
    if result is not None:
        rank_found, U_found, V_found, W_found = result
        if rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = U_found, V_found, W_found
            print(f"Found rank {rank_found} decomposition via ladder reduction")

    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        print("Warning: Final decomposition failed verification, using schoolbook")
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
