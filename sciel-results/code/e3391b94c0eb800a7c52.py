import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_A_for_W(U, V, rank):
        """Build linear system matrix for solving W given U,V"""
        A_mat = np.zeros((n * m * p, p * n * rank))
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    col_base = k * n * rank + i * rank + r
                    for j in range(m):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
        return A_mat
    
    def build_A_for_V(U, W, rank):
        """Build linear system matrix for solving V given U,W"""
        A_mat = np.zeros((n * m * p, m * p * rank))
        for r in range(rank):
            for j in range(m):
                for k in range(p):
                    col_base = j * p * rank + k * rank + r
                    for i in range(n):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * W[k * n + i, r]
        return A_mat
    
    def build_A_for_U(V, W, rank):
        """Build linear system matrix for solving U given V,W"""
        A_mat = np.zeros((n * m * p, n * m * rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    col_base = i * m * rank + j * rank + r
                    for k in range(p):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = V[j * p + k, r] * W[k * n + i, r]
        return A_mat
    
    def alternating_minimization(U, V, W, rank, max_iter=50, tol=1e-10):
        """Alternating least squares optimization"""
        target = np.ones(n * m * p)
        best_err = np.inf
        
        for it in range(max_iter):
            if time_left() < 10:
                break
            
            # Solve for W given U, V
            A_W = build_A_for_W(U, V, rank)
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_W, target, rcond=None)
                W_new = W_flat.reshape(p * n, rank)
                err_W = np.linalg.norm(A_W @ W_flat - target)
                if err_W < best_err:
                    best_err = err_W
                    W = W_new
            except:
                pass
            
            # Solve for V given U, W
            A_V = build_A_for_V(U, W, rank)
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_V, target, rcond=None)
                V_new = V_flat.reshape(m * p, rank)
                err_V = np.linalg.norm(A_V @ V_flat - target)
                if err_V < best_err:
                    best_err = err_V
                    V = V_new
            except:
                pass
            
            # Solve for U given V, W
            A_U = build_A_for_U(V, W, rank)
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, target, rcond=None)
                U_new = U_flat.reshape(n * m, rank)
                err_U = np.linalg.norm(A_U @ U_flat - target)
                if err_U < best_err:
                    best_err = err_U
                    U = U_new
            except:
                pass
            
            # Check convergence
            if best_err < tol:
                return U, V, W, best_err
        
        return U, V, W, best_err
    
    def try_rank(rank, max_trials=100):
        """Try to find a valid decomposition of given rank"""
        if time_left() < 15:
            return None
        
        target = np.ones(n * m * p)
        
        for trial in range(max_trials):
            if time_left() < 15:
                return None
            
            # Different initialization strategies
            strategy = trial % 6
            
            if strategy == 0:
                # Random ±1
                U = np.random.randint(-1, 2, (n * m, rank))
                V = np.random.randint(-1, 2, (m * p, rank))
                W = np.random.randint(-1, 2, (p * n, rank))
            elif strategy == 1:
                # Sparse random
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for _ in range(int(n * m * rank * 0.4)):
                    U[np.random.randint(n * m), np.random.randint(rank)] = np.random.choice([-1, 1])
                for _ in range(int(m * p * rank * 0.4)):
                    V[np.random.randint(m * p), np.random.randint(rank)] = np.random.choice([-1, 1])
                for _ in range(int(p * n * rank * 0.4)):
                    W[np.random.randint(p * n), np.random.randint(rank)] = np.random.choice([-1, 1])
            elif strategy == 2:
                # Block structure based on i-index
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                block_size = max(1, rank // n)
                for i in range(n):
                    b_start = i * block_size
                    b_end = min(b_start + block_size, rank)
                    for j in range(m):
                        U[i * m + j, b_start:b_end] = np.random.randint(-1, 2, (b_end - b_start,))
                    for k in range(p):
                        W[k * n + i, b_start:b_end] = np.random.randint(-1, 2, (b_end - b_start,))
                V = np.random.randint(-1, 2, (m * p, rank))
            elif strategy == 3:
                # Gaussian initialization
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
                # Normalize columns
                for r in range(rank):
                    if np.linalg.norm(U[:, r]) > 0:
                        U[:, r] /= np.linalg.norm(U[:, r])
                    if np.linalg.norm(V[:, r]) > 0:
                        V[:, r] /= np.linalg.norm(V[:, r])
                    if np.linalg.norm(W[:, r]) > 0:
                        W[:, r] /= np.linalg.norm(W[:, r])
            elif strategy == 4:
                # Structured: use Kronecker-like patterns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for r in range(rank):
                    # Assign random signs with some structure
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, r] = np.random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i, r] = np.random.choice([-1, 0, 1])
            else:
                # Hybrid: start with schoolbook-like structure, then perturb
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                # Initialize with a subset of schoolbook terms
                num_init = min(rank, n * m * p // 3)
                indices = np.random.choice(n * m * p, num_init, replace=False)
                for idx in indices:
                    col = idx % rank
                    i = idx // (m * p)
                    j = (idx // p) % m
                    k = idx % p
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                # Add random perturbations to remaining columns
                for col in range(num_init, rank):
                    U[:, col] = np.random.randint(-1, 2, n * m)
                    V[:, col] = np.random.randint(-1, 2, m * p)
                    W[:, col] = np.random.randint(-1, 2, p * n)
            
            # Run alternating minimization
            U_opt, V_opt, W_opt, err = alternating_minimization(U, V, W, rank, max_iter=30)
            
            # Check if we found an exact solution
            if err < 1e-9:
                return U_opt, V_opt, W_opt
            
            # Try rounding to integers/half-integers
            U_round = np.round(U_opt)
            V_round = np.round(V_opt)
            W_round = np.round(W_opt)
            if verify(U_round, V_round, W_round):
                return U_round, V_round, W_round
            
            # Try half-integers
            U_half = np.round(2 * U_opt) / 2.0
            V_half = np.round(2 * V_opt) / 2.0
            W_half = np.round(2 * W_opt) / 2.0
            if verify(U_half, V_half, W_half):
                return U_half, V_half, W_half
        
        return None
    
    # Search from low to high rank
    # For 3x4x4, try ranks from 30 to 47
    for rank in range(30, 48):
        if time_left() < 20:
            break
        
        # More trials for lower ranks (harder to find)
        trials = 50 if rank < 38 else 30
        result = try_rank(rank, max_trials=trials)
        
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
