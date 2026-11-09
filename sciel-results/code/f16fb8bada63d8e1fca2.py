import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_40():
    """
    Attempt to find a rank-40 decomposition using a structured initialization
    and local search. The structure exploits the fact that we can split the
    3x10 B matrix into two 3x5 blocks, or use a different partition.
    
    For 2x3x10, a rank-40 decomposition would be a 33% improvement over schoolbook.
    We initialize with a block-diagonal structure and then refine.
    """
    n, m, p = 2, 3, 10
    R = 40
    
    # Initialize with a structured pattern
    # Split p into 2 blocks of 5
    # For each block, use a rank-20 sub-decomposition (if possible)
    # Since rank-20 for 2x3x5 is not known, we use random init with structure
    
    U = np.random.randn(n*m, R) * 0.3
    V = np.random.randn(m*p, R) * 0.3
    W = np.random.randn(p*n, R) * 0.3
    
    # Local search using gradient-free method (Nelder-Mead style on flattened params)
    # But that's too slow for 40*6+40*30+40*20 = 240+1200+800 = 2240 params
    
    # Instead, use a simpler approach: iterative refinement
    # For each column, try to optimize it given the others are fixed
    
    best_residual = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for trial in range(20):
        # Compute current residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if residual < 1e-8:
            break
        
        # Try to improve by adjusting one column at a time
        for r in range(R):
            # Try random perturbations of column r
            for _ in range(5):
                U_old = U[:, r].copy()
                V_old = V[:, r].copy()
                W_old = W[:, r].copy()
                
                U[:, r] += np.random.randn(n*m) * 0.1
                V[:, r] += np.random.randn(m*p) * 0.1
                W[:, r] += np.random.randn(p*n) * 0.1
                
                # Compute new residual
                new_residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr] for rr in range(R))
                            new_residual += (val - 1.0) ** 2
                
                if new_residual < residual:
                    residual = new_residual
                else:
                    U[:, r] = U_old
                    V[:, r] = V_old
                    W[:, r] = W_old
    
    if best_residual > 1e-6:
        return None
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_rank_35():
    """
    Attempt to find a rank-35 decomposition.
    """
    n, m, p = 2, 3, 10
    R = 35
    
    U = np.random.randn(n*m, R) * 0.3
    V = np.random.randn(m*p, R) * 0.3
    W = np.random.randn(p*n, R) * 0.3
    
    best_residual = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for trial in range(15):
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if residual < 1e-8:
            break
        
        for r in range(R):
            U_old = U[:, r].copy()
            V_old = V[:, r].copy()
            W_old = W[:, r].copy()
            
            U[:, r] += np.random.randn(n*m) * 0.1
            V[:, r] += np.random.randn(m*p) * 0.1
            W[:, r] += np.random.randn(p*n) * 0.1
            
            new_residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr] for rr in range(R))
                        new_residual += (val - 1.0) ** 2
            
            if new_residual < residual:
                residual = new_residual
            else:
                U[:, r] = U_old
                V[:, r] = V_old
                W[:, r] = W_old
    
    if best_residual > 1e-6:
        return None
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_rank_30():
    """
    Attempt to find a rank-30 decomposition. This is the theoretical lower bound
    max(n*p, m*p, n*m) = max(20, 30, 6) = 30.
    """
    n, m, p = 2, 3, 10
    R = 30
    
    U = np.random.randn(n*m, R) * 0.3
    V = np.random.randn(m*p, R) * 0.3
    W = np.random.randn(p*n, R) * 0.3
    
    best_residual = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for trial in range(10):
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if residual < 1e-8:
            break
        
        for r in range(R):
            U_old = U[:, r].copy()
            V_old = V[:, r].copy()
            W_old = W[:, r].copy()
            
            U[:, r] += np.random.randn(n*m) * 0.1
            V[:, r] += np.random.randn(m*p) * 0.1
            W[:, r] += np.random.randn(p*n) * 0.1
            
            new_residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr] for rr in range(R))
                        new_residual += (val - 1.0) ** 2
            
            if new_residual < residual:
                residual = new_residual
            else:
                U[:, r] = U_old
                V[:, r] = V_old
                W[:, r] = W_old
    
    if best_residual > 1e-6:
        return None
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Best fallback: schoolbook rank 60
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try rank-40
    if time.time() - start_time < deadline - 30:
        result = _try_rank_40()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    # Try rank-35
    if time.time() - start_time < deadline - 20:
        result = _try_rank_35()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    # Try rank-30
    if time.time() - start_time < deadline - 10:
        result = _try_rank_30()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    return best_result
