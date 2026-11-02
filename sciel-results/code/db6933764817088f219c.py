"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor from U, V, W: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But we need to check against the standard matrix multiplication tensor
    # T[i,j,k] = 1 if k == i*m + j (i.e., the (i,j) entry of A times (j,k) entry of B gives (i,k))
    # Wait, let's think about the tensor for matrix multiplication C = A * B
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if this term contributes to C[i,k]
    # Actually the tensor for matmul is T[i, j, k] = 1 if (i,j,k) corresponds to a valid term
    # More precisely, the matmul tensor has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the output index is (i,k) and the input indices are (i,j) and (j,k)
    
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor
    
    # For matrix multiplication, the tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]
    
    # Wait, that's not right either. Let me reconsider.
    # The matmul tensor T is defined such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's just the identity in the summation)
    # No wait, T[i,j,k] should be 1 only when the indices are consistent.
    # Actually for standard matmul, T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # because every combination (i,j,k) is a valid term in the sum.
    
    # So the tensor we're decomposing has T[i,j,k] = 1 for all i,j,k.
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tol:
                    errors.append((i, j, k, val))
    
    return len(errors) == 0, errors

def _try_random_sparse_init(n, m, p, rank, max_attempts=50, time_limit=10.0):
    """Try to find a rank-r decomposition using random sparse initialization and optimization."""
    start_time = time.time()
    target_time = start_time + time_limit
    
    for attempt in range(max_attempts):
        if time.time() > target_time:
            break
        
        # Create sparse random initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Each column of U, V, W should have some non-zero entries
        # For a valid decomposition, each "rank-1 slice" should contribute to the tensor
        # Let's try a structured random init
        
        for r in range(rank):
            # Randomly select a few non-zero positions
            u_idx = random.sample(range(n * m), min(3, n * m))
            v_idx = random.sample(range(m * p), min(3, m * p))
            w_idx = random.sample(range(p * n), min(3, p * n))
            
            for idx in u_idx:
                U[idx, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            for idx in v_idx:
                V[idx, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            for idx in w_idx:
                W[idx, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*rank], x[n*m*rank:2*n*m*rank], x[2*n*m*rank:]
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            # Compute error: sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        err += (val - 1.0) ** 2
            return err
        
        def gradient(x):
            U_flat, V_flat, W_flat = x[:n*m*rank], x[n*m*rank:2*n*m*rank], x[2*n*m*rank:]
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            grad_U = np.zeros_like(U_mat)
            grad_V = np.zeros_like(V_mat)
            grad_W = np.zeros_like(W_mat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        err_term = 2.0 * (val - 1.0)
                        for r in range(rank):
                            grad_U[i*m+j, r] += err_term * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                            grad_V[j*p+k, r] += err_term * U_mat[i*m+j, r] * W_mat[k*n+i, r]
                            grad_W[k*n+i, r] += err_term * U_mat[i*m+j, r] * V_mat[j*p+k, r]
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # Try L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
                # Found a solution!
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_sol = result.x[2*n*m*rank:].reshape(p*n, rank)
                
                # Convert to lists and check
                U_list = [[U_sol[i, r] for r in range(rank)] for i in range(n*m)]
                V_list = [[V_sol[i, r] for r in range(rank)] for i in range(m*p)]
                W_list = [[W_sol[i, r] for r in range(rank)] for i in range(p*n)]
                
                valid, errors = _verify(U_list, V_list, W_list, n, m, p)
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 3x3x6 matmul is around 54 (but this is conjectural)
    # Let's try ranks from lower to higher
    start_time = time.time()
    deadline = start_time + 110  # Leave 10 seconds margin
    
    # Try various ranks
    for rank in [54, 55, 56, 57, 58, 59, 60]:
        if time.time() > deadline:
            break
        
        result = _try_random_sparse_init(n, m, p, rank, max_attempts=20, time_limit=10.0)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
