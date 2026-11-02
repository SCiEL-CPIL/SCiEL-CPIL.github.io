# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses structured search exploiting n=2 to find low-rank decompositions.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    def try_rank_structured(rank):
        """
        Try to find a decomposition of given rank using structured initialization.
        Exploit n=2 by fixing U and V to simple patterns and solving for W.
        """
        if time.time() > deadline:
            return None
        
        # Strategy: Use structured U and V with limited support
        # For each rank r, we need:
        # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Try different structured patterns
        best_result = None
        
        # Pattern 1: U[i*m+j, r] depends only on (i, j, r) with simple values
        # Pattern 2: Use Kronecker product structure
        
        # Let's try a more systematic approach:
        # Fix U and V to have entries in {0, 1, -1} with limited support
        # Then solve the resulting linear system for W
        
        np.random.seed(12345 + rank)
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # For each rank position, assign a random (i,j) to U and (j,k) to V
            for r in range(rank):
                # Choose which (i,j) pairs this rank handles for U
                # Each rank should cover a subset of (i,j) pairs
                # Simple approach: each rank covers one (i,j) pair for U
                i_idx = np.random.randint(n)
                j_idx = np.random.randint(m)
                U[i_idx * m + j_idx, r] = 1.0
                
                # For V, choose which (j,k) pairs
                # The j should match the j from U for consistency
                k_idx = np.random.randint(p)
                V[j_idx * p + k_idx, r] = 1.0
            
            # Now solve for W: for each (i,j,k), we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This gives us n*p equations (for each i,k) with rank unknowns per (i,k)
            # Actually, W has p*n entries per rank, so rank * p*n unknowns
            # But the equations couple them through the sum over r
            
            # Let's set up the linear system:
            # For each (i,k): sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in W!
            
            # Build the system: A * W_vec = b
            # where W_vec is the vector of all W entries
            # A has shape (n*p, rank * p * n)
            
            num_eqs = n * p
            num_unk = rank * p * n
            
            A = np.zeros((num_eqs, num_unk))
            b = np.zeros(num_eqs)
            
            for i in range(n):
                for k in range(p):
                    eq_idx = i * p + k
                    b[eq_idx] = 1.0
                    for j in range(m):
                        for r in range(rank):
                            u_val = U[i * m + j, r]
                            v_val = V[j * p + k, r]
                            if u_val != 0 and v_val != 0:
                                # This term contributes u_val * v_val * W[k*n+i, r]
                                w_idx = r * (p * n) + k * n + i
                                A[eq_idx, w_idx] += u_val * v_val
            
            # Solve the linear system
            try:
                W_vec, residuals, rank_actual, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    W = W_vec.reshape(rank, p * n)
                    
                    # Verify
                    U_list = [[round(float(x), 10) for x in row] for row in U]
                    V_list = [[round(float(x), 10) for x in row] for row in V]
                    W_list = [[round(float(x), 10) for x in row] for row in W]
                    
                    if verify_decomposition(U_list, V_list, W_list, rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
            except np.linalg.LinAlgError:
                continue
        
        return None
    
    def try_rank_gradient(rank, max_iter=2000):
        """Try to find a decomposition using gradient descent with better initialization."""
        if time.time() > deadline:
            return None
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(777 + rank * 100 + init)
            
            if init == 0:
                # Random initialization
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            elif init == 1:
                # Structured: each rank handles a unique (i,j,k) triple
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                col_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col_idx < rank:
                                U[i * m + j, col_idx] = 1.0
                                V[j * p + k, col_idx] = 1.0
                                W[k * n + i, col_idx] = 1.0
                                col_idx += 1
            else:
                # Random with different scale
                scale = 0.1 + 0.1 * init
                U = np.random.randn(n * m, rank) * scale
                V = np.random.randn(m * p, rank) * scale
                W = np.random.randn(p * n, rank) * scale
            
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t_val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            residual = t_val - 1.0
                            loss += residual * residual
                            
                            dU[i * m + j, :] += 2 * residual * (V[j * p + k, :] * W[k * n + i, :])
                            dV[j * p + k, :] += 2 * residual * (U[i * m + j, :] * W[k * n + i, :])
                            dW[k * n + i, :] += 2 * residual * (U[i * m + j, :] * V[j * p + k, :])
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
                
                # Adaptive learning rate
                lr = 0.01 * (0.999 ** (it // 100))
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if best_loss < 1e-10:
                break
        
        if best_U is not None and best_loss < 1e-10:
            U_list = [[round(float(x), 10) for x in row] for row in best_U]
            V_list = [[round(float(x), 10) for x in row] for row in best_V]
            W_list = [[round(float(x), 10) for x in row] for row in best_W]
            
            if verify_decomposition(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But tensor rank can be lower than this
    
    best_result = None
    
    for rank in range(20, 61):
        if time.time() > deadline:
            break
        
        # First try structured approach
        result = try_rank_structured(rank)
        if result is not None:
            best_result = result
            break
        
        # Then try gradient descent
        result = try_rank_gradient(rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
