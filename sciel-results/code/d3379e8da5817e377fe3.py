import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check the 0s
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                support.add((u, v, w))
                
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) in support:
                    continue
                val = sum(U[u, c] * V[v, c] * W[w, c] for c in range(rank))
                if abs(val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_optimize(n, m, p, rank, seed, max_iter=2000, tol=1e-10):
    """Try to find a valid decomposition of given rank using optimization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute current result
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        err = np.sum((result - target) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        # dU[i,j,c] = sum_{k} (result[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        # dV[j,k,c] = sum_{i} (result[i,j,k] - 1) * U[i,j,c] * W[k,i,c]
        # dW[k,i,c] = sum_{j} (result[i,j,k] - 1) * U[i,j,c] * V[j,k,c]
        
        diff = result - target  # (n, m, p)
        
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3)
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3)
        
        # Gradient descent with adaptive step size
        lr = 0.01 / (1 + it * 0.001)
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None

def _try_structured(n, m, p, rank, seed):
    """Try a structured initialization based on block patterns."""
    rng = np.random.RandomState(seed)
    
    # Create a structured pattern: use blocks of size that divide evenly
    # For 3x4x6, try to embed smaller Strassen-like structures
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with random small values for unstructured parts
    for c in range(rank):
        # Random initialization for some columns
        if c % 3 == 0:
            U[:, c] = rng.randn(n * m) * 0.5
            V[:, c] = rng.randn(m * p) * 0.5
            W[:, c] = rng.randn(p * n) * 0.5
        elif c % 3 == 1:
            # Structured: focus on certain indices
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = rng.choice([-1, 1]) * 0.5
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = rng.choice([-1, 1]) * 0.5
            for k in range(p):
                for i in range(n):
                    W[k*n+i, c] = rng.choice([-1, 1]) * 0.5
        else:
            # Mixed
            U[:, c] = rng.randn(n * m) * 0.3
            V[:, c] = rng.randn(m * p) * 0.3
            W[:, c] = rng.randn(p * n) * 0.3
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward, but limit total attempts
    max_rank_to_try = min(60, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 5 if rank < 30 else 3
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            # Method 1: Pure optimization
            result = _try_optimize(n, m, p, rank, seed, max_iter=1500)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                # Round to nearest 0.5 if close
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
            
            # Method 2: Structured initialization + optimization
            if time.time() > deadline - 15:
                break
            U_struct, V_struct, W_struct = _try_structured(n, m, p, rank, seed)
            
            # Optimize from structured init
            U3 = U_struct.reshape(n, m, rank)
            V3 = V_struct.reshape(m, p, rank)
            W3 = W_struct.reshape(p, n, rank)
            
            best_err = np.inf
            for it in range(1000):
                result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                target = np.ones((n, m, p))
                err = np.sum((result - target) ** 2)
                
                if err < best_err:
                    best_err = err
                
                if err < 1e-10:
                    break
                
                diff = result - target
                dU = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
                dV = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3)
                dW = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3)
                
                lr = 0.005 / (1 + it * 0.0005)
                U_struct -= lr * dU.reshape(n*m, rank)
                V_struct -= lr * dV.reshape(m*p, rank)
                W_struct -= lr * dW.reshape(p*n, rank)
                
                U3 = U_struct.reshape(n, m, rank)
                V3 = V_struct.reshape(m, p, rank)
                W3 = W_struct.reshape(p, n, rank)
            
            if best_err < 1e-8:
                U_round = np.round(U_struct * 2) / 2
                V_round = np.round(V_struct * 2) / 2
                W_round = np.round(W_struct * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
    
    return best_result
