"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_target_tensor(n, m, p):
        """Build the multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j', i=i', k=k'"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def check_decomposition(U, V, W, T_target, tol=1e-10):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        r = U.shape[1]
        T_recon = np.zeros_like(T_target)
        for t in range(r):
            T_recon += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True, T_recon
        return False, T_recon
    
    def schoolbook_decomposition(n, m, p):
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
    
    # Build target tensor
    T_target = build_target_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W, current_rank = schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), current_rank
    
    def try_rank_decomposition(target_rank, num_trials=20, max_iter=500):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Use different initialization strategies
            if trial % 3 == 0:
                # Small random initialization
                np.random.seed(trial * 7919 + 42)
                U_t = np.random.randn(n*m, target_rank) * 0.05
                V_t = np.random.randn(m*p, target_rank) * 0.05
                W_t = np.random.randn(p*n, target_rank) * 0.05
            elif trial % 3 == 1:
                # Sparse initialization - start with random subsets of schoolbook
                np.random.seed(trial * 104729 + 12345)
                U_t = np.zeros((n*m, target_rank))
                V_t = np.zeros((m*p, target_rank))
                W_t = np.zeros((p*n, target_rank))
                for t in range(target_rank):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U_t[i*m+j, t] = 1.0
                    V_t[j*p+k, t] = 1.0
                    W_t[k*n+i, t] = 1.0
                # Add small random perturbation
                U_t += np.random.randn(n*m, target_rank) * 0.01
                V_t += np.random.randn(m*p, target_rank) * 0.01
                W_t += np.random.randn(p*n, target_rank) * 0.01
            else:
                # Mixed initialization
                np.random.seed(trial * 15485863 + 999)
                U_t = np.random.randn(n*m, target_rank) * 0.1
                V_t = np.random.randn(m*p, target_rank) * 0.1
                W_t = np.random.randn(p*n, target_rank) * 0.1
                # Set some entries to 1
                for t in range(min(target_rank, 10)):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U_t[i*m+j, t] = 1.0
                    V_t[j*p+k, t] = 1.0
                    W_t[k*n+i, t] = 1.0
            
            lr = 0.02
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute reconstruction
                T_recon = np.zeros_like(T_target)
                for t in range(target_rank):
                    T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
                
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                # Compute gradients
                dU = np.zeros_like(U_t)
                dV = np.zeros_like(V_t)
                dW = np.zeros_like(W_t)
                for t in range(target_rank):
                    dU[:,t] = np.einsum('abc,bc->a', error, V_t[:,t] * W_t[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, U_t[:,t] * W_t[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, U_t[:,t] * V_t[:,t])
                
                # Adaptive learning rate
                if it % 50 == 0 and it > 0:
                    lr *= 0.9
                if lr < 0.001:
                    lr = 0.001
                
                # Update
                U_t -= lr * dU
                V_t -= lr * dV
                W_t -= lr * dW
                
                # Check convergence
                if loss < 1e-14:
                    break
            
            # Final check
            T_recon = np.zeros_like(T_target)
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U_t[:,t], V_t[:,t]), W_t[:,t])
            
            if np.allclose(T_recon, T_target, atol=1e-10, rtol=0):
                return U_t, V_t, W_t, target_rank
        
        return None
    
    # Try ranks from 50 down to 35
    for target_rank in range(50, 34, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_decomposition(target_rank, num_trials=15, max_iter=400)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            break
    
    # Convert to lists
    def to_list(mat):
        """Convert numpy array to list of lists, rounding to nearest half-integer if close."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
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
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
