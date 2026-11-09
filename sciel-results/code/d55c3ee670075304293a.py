"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W form a valid decomposition.
        For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
        """
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_rank_optimization(target_rank, num_trials=5, max_iter=200):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline - 5:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Iterative optimization: minimize ||T - 1||^2 where T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # Use gradient descent with learning rate scheduling
            
            lr = 0.01
            best_loss = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Compute current tensor
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    return (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                diff = 2.0 * (T - 1.0)
                
                # Gradient for U: shape (n, m, target_rank)
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                # Update
                U -= lr * grad_U.reshape(nm, target_rank)
                V -= lr * grad_V.reshape(mp, target_rank)
                W -= lr * grad_W.reshape(pn, target_rank)
                
                # Decay learning rate
                lr *= 0.995
            
            # Check best state
            if check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks from high to low
    for target_rank in [70, 65, 60, 55, 50, 45, 40, 35, 30]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank_optimization(target_rank, num_trials=3, max_iter=150)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition, stop searching
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
