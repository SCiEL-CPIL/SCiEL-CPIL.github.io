"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - SAFETY_MARGIN - elapsed()
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 45
    
    def tensor_error(U, V, W, rank):
        """Compute the sum of squared errors for the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    err += (val - 1.0) ** 2
        return err
    
    def local_search(U, V, W, rank, max_iter=5000):
        """Perform local search to reduce the error."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        lr = 0.01
        best_err = tensor_error(U, V, W, rank)
        
        for it in range(max_iter):
            if remaining() < 5:
                break
            # Randomly perturb one matrix
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = U[row, col]
                U[row, col] += delta
            elif mat == 'V':
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = V[row, col]
                V[row, col] += delta
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                delta = random.uniform(-0.1, 0.1)
                old = W[row, col]
                W[row, col] += delta
            
            new_err = tensor_error(U, V, W, rank)
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-12:
                    break
            else:
                # Revert
                if mat == 'U':
                    U[row, col] = old
                elif mat == 'V':
                    V[row, col] = old
                else:
                    W[row, col] = old
        
        return U, V, W, best_err
    
    def gradient_descent(U, V, W, R, max_iter=5000, lr=0.005):
        """Gradient descent on the tensor error."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        best_err = tensor_error(U, V, W, R)
        
        for it in range(max_iter):
            if remaining() < 5:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :R], V[j*p+k, :R] * W[k*n+i, :R])
                        diff = val - 1.0
                        # d/dU[i*m+j, r] = diff * V[j*p+k, r] * W[k*n+i, r]
                        dU[i*m+j, :R] += diff * V[j*p+k, :R] * W[k*n+i, :R]
                        dV[j*p+k, :R] += diff * U[i*m+j, :R] * W[k*n+i, :R]
                        dW[k*n+i, :R] += diff * U[i*m+j, :R] * V[j*p+k, :R]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_err = tensor_error(U, V, W, R)
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-12:
                    break
            else:
                # Revert and reduce learning rate
                U += lr * dU
                V += lr * dV
                W += lr * dW
                lr *= 0.95
        
        return U, V, W, best_err
    
    def try_rank(R, max_attempts=20):
        """Try to find a rank-R decomposition using random init + gradient descent."""
        if remaining() < 10:
            return None
        
        for attempt in range(max_attempts):
            if remaining() < 5:
                break
            
            # Random initialization with varied scales
            scale = 0.1 * (1 + 0.5 * attempt)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            # Gradient descent
            U, V, W, err = gradient_descent(U, V, W, R, max_iter=3000, lr=0.01)
            
            if err < 1e-10:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                # Also try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, R):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None
    
    def try_column_merging(R, max_attempts=100):
        """Try to find a rank-R decomposition by merging columns from schoolbook."""
        if remaining() < 10:
            return None
        
        # Start with schoolbook
        sb = make_schoolbook()
        U0 = np.array(sb["U"], dtype=float)
        V0 = np.array(sb["V"], dtype=float)
        W0 = np.array(sb["W"], dtype=float)
        R0 = n * m * p  # 45
        
        # We need to reduce from R0 to R, so we need to merge (R0 - R) pairs of columns
        merges_needed = R0 - R
        
        for attempt in range(max_attempts):
            if remaining() < 5:
                break
            
            # Randomly choose which columns to merge
            # A merge means: replace columns r1 and r2 with a single column that is a combination
            # We try: new_col = a*col_r1 + b*col_r2 for some coefficients
            # But this changes the tensor values, so we need to adjust other columns
            
            # Alternative approach: start with random integer matrices of rank R
            # and use simulated annealing to minimize error
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            W = np.random.randint(-2, 3, size=(p*n, R))
            
            err = tensor_error(U, V, W, R)
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            temp = 1.0
            for sa_iter in range(5000):
                if remaining() < 3:
                    break
                
                # Random perturbation
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old = U[row, col]
                    U[row, col] += delta
                elif mat == 'V':
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old = V[row, col]
                    V[row, col] += delta
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old = W[row, col]
                    W[row, col] += delta
                
                new_err = tensor_error(U, V, W, R)
                
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-12:
                        break
                elif random.random() < np.exp((best_err - new_err) / temp):
                    pass  # Accept worse solution
                else:
                    # Revert
                    if mat == 'U':
                        U[row, col] = old
                    elif mat == 'V':
                        V[row, col] = old
                    else:
                        W[row, col] = old
                
                temp *= 0.999
            
            if best_err < 1e-10:
                # Round to nearest half-integer
                U_round = np.round(best_U * 2) / 2
                V_round = np.round(best_V * 2) / 2
                W_round = np.round(best_W * 2) / 2
                if verify(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                if verify(U_int, V_int, W_int, R):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None
    
    # Try ranks from 43 upward
    for target_rank in [43, 44]:
        if remaining() < 15:
            break
        
        # Try column merging / simulated annealing
        result = try_column_merging(target_rank, max_attempts=30)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
        
        # Then try the gradient descent search
        result = try_rank(target_rank, max_attempts=10)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
