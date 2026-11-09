import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _solve_linear_system(A, b):
    """Solve A x = b using least squares."""
    if A.shape[1] == 0:
        return np.zeros(0)
    # Use lstsq for overdetermined systems
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return x

def _als_iteration(U, V, W, T, num_a, num_b, num_c, rank):
    """One full ALS iteration: update U, V, W."""
    new_U = np.zeros_like(U)
    for a in range(num_a):
        M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_b * num_c, rank)
        b_vec = T[a, :, :].reshape(-1)
        new_U[a, :] = _solve_linear_system(M_a, b_vec)
    U = new_U
    
    new_V = np.zeros_like(V)
    for b in range(num_b):
        M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_a * num_c, rank)
        b_vec = T[:, b, :].reshape(-1)
        new_V[b, :] = _solve_linear_system(M_b, b_vec)
    V = new_V
    
    new_W = np.zeros_like(W)
    for c in range(num_c):
        M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(num_a * num_b, rank)
        b_vec = T[:, :, c].reshape(-1)
        new_W[c, :] = _solve_linear_system(M_c, b_vec)
    W = new_W
    
    return U, V, W

def _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=50, lr=0.01):
    """Polish the solution using gradient descent on the loss function."""
    def loss_and_grad(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T
        loss = np.sum(diff ** 2)
        
        # Gradients: dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        return loss, grad_U, grad_V, grad_W
    
    for it in range(n_iter):
        loss, gU, gV, gW = loss_and_grad(U, V, W)
        if loss < 1e-15:
            break
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    num_a = n * m  # 6
    num_b = m * p  # 21
    num_c = p * n  # 14
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as fallback
    
    # Try ranks from a lower bound upward
    # Theoretical lower bound is max(num_a, num_b, num_c) = 21
    # But we can try even lower
    min_rank = 15
    max_rank_to_try = 35
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts for each rank
        num_restarts = 5
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 42 + restart * 1000
            np.random.seed(seed)
            U = np.random.randn(num_a, rank) * 0.5
            V = np.random.randn(num_b, rank) * 0.5
            W = np.random.randn(num_c, rank) * 0.5
            
            # ALS iterations
            prev_loss = np.inf
            for it in range(100):
                if time.time() > deadline - 8:
                    break
                
                U, V, W = _als_iteration(U, V, W, T, num_a, num_b, num_c, rank)
                
                # Compute loss
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T_recon - T) ** 2)
                
                if loss < 1e-12:
                    found = True
                    break
                
                if abs(prev_loss - loss) < 1e-14:
                    # Stuck, try gradient polish
                    U, V, W = _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=30, lr=0.005)
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    loss = np.sum((T_recon - T) ** 2)
                    if loss < 1e-12:
                        found = True
                        break
                
                prev_loss = loss
            
            if found:
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    # If we didn't find a better rank, use schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((num_a, rank))
        V = np.zeros((num_b, rank))
        W = np.zeros((num_c, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    U, V, W = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
