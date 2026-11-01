"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Build the multiplication tensor T[i,j,k] = 1 if this term appears in (AB)[i,k]
    # T is a 3D tensor of shape (n*m, m*p, p*n) but we work with the flattened form
    # The tensor M[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # M[a,b,c] = 1 if j==j' and i==i' and k==k' (i.e., the product term contributes)
    
    # Actually, the multiplication tensor T has entries T[i,j,k] for the (i,k) output
    # T[i,j,k] = delta_{j, j'} * delta_{i, i'} * delta_{k, k'}
    # In the standard form: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for each (i,j,k) triple
    
    # The tensor as a 3D array of shape (n*m, m*p, p*n) doesn't quite work directly.
    # Let me think about this differently.
    
    # The multiplication tensor M of order (n*m, m*p, p*n) has:
    # M[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' AND j==j' AND i==i' AND k==k'
    # Wait, that's not right either.
    
    # Standard formulation: The tensor T has shape (n*m, m*p, p*n) where
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # But this only makes sense when the indices align.
    
    # Actually for matrix multiplication A (n×m) * B (m×p) = C (n×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T of shape (n*m, m*p, p*n) is defined as:
    # T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j', i=i', k=k'
    # This means T has exactly n*m*p = 54 nonzero entries (all 1s)
    
    # For rank-r decomposition: T = sum_{t=1}^{r} U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Let's use a different approach: try to find a low-rank decomposition using
    # the structure of the problem.
    
    # One known approach: use the fact that for small matrices, there may be
    # structured decompositions. Let's try random initialization + gradient descent
    # on the tensor reconstruction error, with exact verification.
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        r = len(U[0])
        # Reconstruct tensor
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_recon += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Check exact match (with tolerance for floating point)
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True, T_recon
        return False, T_recon
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, current_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), current_rank
    
    # Try to reduce rank by finding linear dependencies
    # Approach: try random rank-(r) initializations and use iterative refinement
    
    def try_rank_reduction(target_rank, num_trials=20, max_iter=200):
        """Try to find a rank-target_rank decomposition."""
        if target_rank >= current_rank:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            np.random.seed(trial * 1000 + 42)
            U_t = np.random.randn(n*m, target_rank) * 0.1
            V_t = np.random.randn(m*p, target_rank) * 0.1
            W_t = np.random.randn(p*n, target_rank) * 0.1
            
            # Build target tensor
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Iterative refinement: minimize ||T_target - sum_t U_t[:,t]⊗V_t[:,t]⊗W_t[:,t]||_F
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute reconstruction
                T_recon = np.zeros((n*m, m*p, p*n))
                for t in range(target_rank):
                    T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
                
                # Compute gradient
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                # Gradients
                dU = np.zeros_like(U_t)
                dV = np.zeros_like(V_t)
                dW = np.zeros_like(W_t)
                for t in range(target_rank):
                    # dU[:,t] = sum_{b,c} error[a,b,c] * V[b,t] * W[c,t] for all a
                    # dV[:,t] = sum_{a,c} error[a,b,c] * U[a,t] * W[c,t] for all b
                    # dW[:,t] = sum_{a,b} error[a,b,c] * U[a,t] * V[b,t] for all c
                    dU[:,t] = np.einsum('abc,bc->a', error, V_t[:,t] * W_t[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, U_t[:,t] * W_t[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, U_t[:,t] * V_t[:,t])
                
                # Update
                U_t -= lr * dU
                V_t -= lr * dV
                W_t -= lr * dW
                
                # Check convergence
                if loss < 1e-12:
                    break
                
                # Adaptive learning rate
                if it % 10 == 0 and it > 0:
                    lr *= 0.95
            
            # Final check
            T_recon = np.zeros((n*m, m*p, p*n))
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
            
            if np.allclose(T_recon, T_target, atol=1e-10, rtol=0):
                return U_t, V_t, W_t, target_rank
        
        return None
    
    # Try progressively lower ranks
    # Known lower bounds: for 3x3x3, rank is at least 19 (but we don't know for 3x3x6)
    # Let's try from rank 40 downward
    for target_rank in range(40, max(10, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        result = try_rank_reduction(target_rank, num_trials=5, max_iter=100)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    U_final, V_final, W_final, r_final = best_U, best_V, best_W, best_rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        """Convert numpy array to list of lists, rounding to nearest half-integer if close."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Check if close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    new_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    new_row.append(nearest_half)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
