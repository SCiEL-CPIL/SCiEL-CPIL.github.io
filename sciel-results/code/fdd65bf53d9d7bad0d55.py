"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from scipy.optimize import minimize

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = n, m, p
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i,j,k (it's the identity tensor for matrix mult)
    # Actually, the tensor for matrix multiplication is T[i,j,k] = delta_{i,k}... 
    # No. The matrix multiplication tensor is T[i,j,k] = 1 if we think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k? No.
    # Let me re-read the task.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The tensor is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k]
    # So we need T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's not right either. Let me think again.
    # The matrix multiplication tensor is a 3-tensor T where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, in the standard tensor decomposition of matrix multiplication:
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should equal 1 for all i,j,k.
    # Because C[i,k] = sum_j sum_c A[i,j] U[i*m+j,c] * B[j,k] V[j*p+k,c] W[k*n+i,c]
    # = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c]) * W[k*n+i,c]
    # Hmm, this isn't quite right. Let me reconsider.
    
    # The standard formulation: we want to compute C = AB where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_c U[i*m+j? no...
    
    # Let me look at the seed code more carefully:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # For each (i,j,k) triple, there's one column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The tensor entry T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For the schoolbook, T[i,j,k] = 1 for each (i,j,k) because there's exactly one column c (the one for that (i,j,k)) where all three are 1.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But wait, is that the matrix multiplication tensor? Let me verify:
    # If T[i,j,k] = 1 for all i,j,k, then:
    # sum_j T[i,j,k] A[i,j] B[j,k]... no, that's not how it works.
    
    # Actually, I think the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] must be 1 for all i,j,k to get the correct result.
    # Yes! That's it. The matrix multiplication tensor has all entries equal to 1.
    
    # So we need: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_search_for_lower_rank(n, m, p, max_rank, time_limit=100):
    """Try to find a rank-minimal decomposition using optimization."""
    start_time = time.time()
    U0, V0, W0 = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    for target_rank in range(max(1, best_rank - 1), 0, -1):
        if time.time() - start_time > time_limit:
            break
        # Try random initialization and optimization
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            def loss(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                loss_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                            loss_val += (val - 1.0) ** 2
                return loss_val
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            try:
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                if res.fun < 1e-6:
                    U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        break
            except Exception:
                pass
        if best_rank == target_rank:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get schoolbook as baseline
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank with a time budget
    time_budget = 120 - (time.time() - start_time) - 10  # 10s safety margin
    if time_budget > 30:
        target_max_rank = max(1, best_rank - 5)  # Try ranks from best_rank-5 down to 1
        found_rank, found_U, found_V, found_W = _try_search_for_lower_rank(
            n, m, p, target_max_rank, time_limit=time_budget
        )
        if found_rank < best_rank:
            best_rank = found_rank
            best_U, best_V, best_W = found_U, found_V, found_W
    
    # Convert to lists of lists with integer/half-integer entries if possible
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest half-integer
                r_val = round(val * 2) / 2.0
                if abs(r_val - val) < 1e-6:
                    r.append(r_val)
                else:
                    r.append(val)
            result.append(r)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
