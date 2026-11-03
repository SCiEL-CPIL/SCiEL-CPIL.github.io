import numpy as np
from scipy.optimize import minimize
import time

# Matrix dimensions
n, m, p = 3, 4, 7

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = 120.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 84
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = build_schoolbook()
    
    # Try to find a lower-rank decomposition using optimization
    # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if i,j,k valid, 0 otherwise
    # Actually the tensor is: T[i,j,k] = delta_{i,k?} No.
    # Matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T_{i,j,k} = 1 for all valid (i,j,k) where the contraction gives the product
    # The multiplication tensor has entries: M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition is: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Wait, let me re-read the problem.
    # "U factor follows the official trace convention W[k*n+i,r]"
    # So U has shape (n*m, rank), indexed by [i*m + j, r]
    # V has shape (m*p, rank), indexed by [j*p + k, r]  
    # W has shape (p*n, rank), indexed by [k*n + i, r]
    
    # The tensor identity is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # (since the matrix multiplication tensor has all ones for valid indices)
    
    # Actually no. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with schoolbook: each term is 1 only when the triple (i,j,k) matches exactly one column
    # So yes, the sum equals 1 for all valid (i,j,k).
    
    # Try smaller ranks
    # Known results: for 3x4x7, the minimum rank is likely around 70-80
    # Let's try to use a randomized approach with rounding
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, rank, tol=1e-6):
        """Check if the decomposition is exact (within tolerance for floating point)"""
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Vectorized check
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    errors.append(abs(val - 1.0))
        return max(errors) < tol
    
    def try_randomized_rank(rank, num_trials=20):
        """Try to find a decomposition of given rank using random initialization and optimization"""
        for trial in range(num_trials):
            if time.time() - start_time > deadline - 10:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Define the loss function
            def loss(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_flat = params[2*n*m*rank:].reshape(p*n, rank)
                
                total_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            total_err += (val - 1.0)**2
                return total_err
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B for optimization
            result = minimize(loss, params, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-4})
            
            if result.success or result.fun < 1e-4:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_opt = result.x[2*n*m*rank:].reshape(p*n, rank)
                
                # Round to nearest integer or half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Check if rounded version is exact
                if check_decomposition(U_round, V_round, W_round, rank, tol=1e-9):
                    return U_round, V_round, W_round
        
        return None
    
    # Try ranks from slightly below schoolbook down
    for rank in range(rank_schoolbook - 1, max(50, rank_schoolbook - 20), -1):
        if time.time() - start_time > deadline - 15:
            break
        
        result = try_randomized_rank(rank, num_trials=5)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_rank = rank
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            print(f"Found rank {rank} decomposition")
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
