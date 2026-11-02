import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Target ranks to try, from lower to higher
    target_ranks = [50, 55, 60, 65, 70]
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def solve_for_U(V, W, r):
        """Solve for U given V and W using least squares."""
        # For each i*m+j, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is: U[i*m+j] @ (V[j*p+k] * W[k*n+i]) = 1 for all k
        # So: U[i*m+j] @ A = 1 where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                # Solve U[i*m+j] @ A = ones(p)
                # This is A.T @ U[i*m+j].T = ones(p)
                try:
                    U[i * m + j] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    U[i * m + j] = np.random.randn(r)
        return U
    
    def solve_for_V(U, W, r):
        """Solve for V given U and W using least squares."""
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                try:
                    V[j * p + k] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    V[j * p + k] = np.random.randn(r)
        return V
    
    def solve_for_W(U, V, r):
        """Solve for W given U and V using least squares."""
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                try:
                    W[k * n + i] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                except:
                    W[k * n + i] = np.random.randn(r)
        return W
    
    np.random.seed(42)
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterate ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                U = solve_for_U(V, W, target_rank)
                V = solve_for_V(U, W, target_rank)
                W = solve_for_W(U, V, target_rank)
                
                # Check convergence
                if check_decomposition(U, V, W, tol=1e-6):
                    # Refine with tighter tolerance
                    for refine_it in range(50):
                        U = solve_for_U(V, W, target_rank)
                        V = solve_for_V(U, W, target_rank)
                        W = solve_for_W(U, V, target_rank)
                        if check_decomposition(U, V, W, tol=1e-10):
                            break
                    
                    if check_decomposition(U, V, W, tol=1e-9):
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
