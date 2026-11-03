import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((U_dim, rank_sb))
    V_sb = np.zeros((V_dim, rank_sb))
    W_sb = np.zeros((W_dim, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T with half-integer entries."""
        # Check half-integer constraint
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        if not (np.allclose(U, U2/2, atol=tol) and 
                np.allclose(V, V2/2, atol=tol) and 
                np.allclose(W, W2/2, atol=tol)):
            return False
        # Check tensor decomposition
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=tol)
    
    def try_find_lower_rank(rank, time_budget):
        """Try to find a rank-r decomposition using linear algebra + local search."""
        if time.time() > start_time + time_budget:
            return None
        
        # Strategy: Start from schoolbook and try to express one term as a 
        # linear combination of others, or use random perturbation + optimization
        
        # Approach 1: Try to find a rank-(rank_sb-1) decomposition by 
        # solving for coefficients that make one column redundant
        # This is complex, so let's try gradient-based optimization with 
        # half-integer snapping
        
        U = np.random.randn(U_dim, rank) * 0.3
        V = np.random.randn(V_dim, rank) * 0.3
        W = np.random.randn(W_dim, rank) * 0.3
        
        # Use Adam-like optimizer
        lr = 0.005
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        
        for step in range(2000):
            if time.time() > start_time + time_budget:
                break
            
            # Compute tensor product
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            R = prod - T  # Residual
            
            # Gradients
            grad_U = np.einsum('abc,br,cr->ar', R, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V, optimize=True)
            
            # Adam update
            t = step + 1
            m_U = b1*m_U + (1-b1)*grad_U
            v_U = b2*v_U + (1-b2)*grad_U**2
            m_U_hat = m_U/(1-b1**t)
            v_U_hat = v_U/(1-b2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = b1*m_V + (1-b1)*grad_V
            v_V = b2*v_V + (1-b2)*grad_V**2
            m_V_hat = m_V/(1-b1**t)
            v_V_hat = v_V/(1-b2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = b1*m_W + (1-b1)*grad_W
            v_W = b2*v_W + (1-b2)*grad_W**2
            m_W_hat = m_W/(1-b1**t)
            v_W_hat = v_W/(1-b2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Periodically try to snap to half-integers
            if step % 50 == 0:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify_decomposition(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap
        
        # Final check
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_decomposition(U_snap, V_snap, W_snap, tol=1e-8):
            return U_snap, V_snap, W_snap
        
        return None
    
    # Try ranks from 49 down to 40
    for rank in range(rank_sb - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Multiple restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + rank * 100 + restart)
            result = try_find_lower_rank(rank, time_budget=15)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
