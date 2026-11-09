"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    global START_TIME
    if 'START_TIME' not in globals() or START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    schoolbook_rank = n * m * p  # 48
    
    # Build the target tensor
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W[k*n + i, c]
    
    # Strategy: Use block decomposition
    # Split 3x8 into two 3x4 blocks
    # For each 2x3x4 sub-problem, use optimized decomposition
    
    # First, let's try to find a good decomposition using optimization
    # with structured initialization
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
                    val = np.sum(prod)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_optimization(rank, num_trials=10, max_iter=5000):
        """Try to find a rank-r decomposition using Adam optimization."""
        best_result = None
        best_loss = float('inf')
        
        for trial in range(num_trials):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Adam optimizer
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            lr = 0.05
            
            for it in range(max_iter):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                t += 1
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
                            val = np.sum(prod)
                            err = val - 1.0
                            loss += err*err
                            
                            dU[i*m+j,:] += 2.0 * err * V[j*p+k,:] * W[k*n+i,:]
                            dV[j*p+k,:] += 2.0 * err * U[i*m+j,:] * W[k*n+i,:]
                            dW[k*n+i,:] += 2.0 * err * U[i*m+j,:] * V[j*p+k,:]
                
                # Adam update
                m_U = beta1*m_U + (1-beta1)*dU
                v_U = beta2*v_U + (1-beta2)*(dU**2)
                m_U_hat = m_U/(1-beta1**t)
                v_U_hat = v_U/(1-beta2**t)
                U -= lr * m_U_hat/(np.sqrt(v_U_hat)+eps)
                
                m_V = beta1*m_V + (1-beta1)*dV
                v_V = beta2*v_V + (1-beta2)*(dV**2)
                m_V_hat = m_V/(1-beta1**t)
                v_V_hat = v_V/(1-beta2**t)
                V -= lr * m_V_hat/(np.sqrt(v_V_hat)+eps)
                
                m_W = beta1*m_W + (1-beta1)*dW
                v_W = beta2*v_W + (1-beta2)*(dW**2)
                m_W_hat = m_W/(1-beta1**t)
                v_W_hat = v_W/(1-beta2**t)
                W -= lr * m_W_hat/(np.sqrt(v_W_hat)+eps)
                
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-12:
                    break
            
            if best_loss < 1e-10:
                # Try to round to half-integers
                U_round = np.round(best_result[0] * 2) / 2
                V_round = np.round(best_result[1] * 2) / 2
                W_round = np.round(best_result[2] * 2) / 2
                err = check_decomposition(U_round, V_round, W_round)
                if err < 1e-9:
                    return U_round, V_round, W_round
                else:
                    err2 = check_decomposition(best_result[0], best_result[1], best_result[2])
                    if err2 < 1e-9:
                        return best_result[0], best_result[1], best_result[2]
        
        return best_result if best_loss < 1e-8 else None
    
    # Try ranks from 47 down to 24
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Initialize with schoolbook
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try lower ranks
    for rank in range(schoolbook_rank - 1, 23, -1):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        result = try_optimization(rank, num_trials=5, max_iter=3000)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            # Found a valid decomposition, try even lower
            continue
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
