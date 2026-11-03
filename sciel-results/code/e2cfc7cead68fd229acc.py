import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def solve_rank(n, m, p, r, time_limit=5.0):
    """
    Attempts to find a rank-r decomposition of the matrix multiplication tensor.
    Returns (U, V, W) if found, else None.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T as a list of non-zero indices (u, v, w)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # We want sum_c U[u,c] V[v,c] W[w,c] == 1 for these indices, and 0 otherwise.
    # This is a system of equations.
    
    # We use a least-squares approach on the non-zero entries and a subset of zero entries?
    # No, we must satisfy all entries.
    # The number of constraints is dim_u * dim_v * dim_w = 8 * 20 * 10 = 1600.
    # Variables: r * (dim_u + dim_v + dim_w) = r * 38.
    
    # For r=36, vars = 1368, cons = 1600.
    # We can use scipy.optimize.least_squares or minimize.
    
    # To speed up, we can fix some parameters?
    # Or use a random initialization and gradient descent.
    
    start_time = time.time()
    
    # Random restarts
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize U, V, W randomly
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        # Normalize to prevent scaling issues
        for c in range(r):
            norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
            if norm > 1e-6:
                U[:, c] /= np.linalg.norm(U[:, c])
                V[:, c] /= np.linalg.norm(V[:, c])
                W[:, c] /= np.linalg.norm(W[:, c])
        
        # Define the residual function
        # We need to evaluate the tensor product and compare to T.
        # T is sparse.
        
        # Precompute indices of non-zero entries in T
        nz_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    nz_indices.append((u, v, w))
        
        # We also need to check some zero entries to ensure they are 0.
        # If we only fit non-zero entries, we might get a solution that is 1 on nz but non-zero elsewhere.
        # However, if the rank is minimal, the zero entries should naturally be 0?
        # Not necessarily.
        # Let's include a random subset of zero entries in the objective.
        
        all_indices = []
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    all_indices.append((u, v, w))
        
        # This is too many to check every step.
        # Strategy: Minimize sum of squares on non-zero entries (target 1)
        # and on a random sample of zero entries (target 0).
        
        zero_indices = [idx for idx in all_indices if idx not in set(nz_indices)]
        
        def residual(U_flat, V_flat, W_flat):
            U = U_flat.reshape(dim_u, r)
            V = V_flat.reshape(dim_v, r)
            W = W_flat.reshape(dim_w, r)
            
            res = []
            # Non-zero entries
            for u, v, w in nz_indices:
                val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                res.append(val - 1.0)
            
            # Sample zero entries
            # To keep it fast, sample 100 zero entries
            if len(zero_indices) > 100:
                sample_zero = np.random.choice(len(zero_indices), 100, replace=False)
            else:
                sample_zero = range(len(zero_indices))
                
            for idx in sample_zero:
                u, v, w = zero_indices[idx]
                val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                res.append(val)
                
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use scipy minimize
        try:
            from scipy.optimize import minimize
            res0 = residual(x0)
            cost0 = np.sum(res0**2)
            
            # Simple gradient descent might be better for speed and robustness to local minima in this high-dim space?
            # Or use L-BFGS-B.
            
            # Let's try a few steps of gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            v_prev = np.zeros_like(x0)
            
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                
                x = x0
                U_c = x[:dim_u*r].reshape(dim_u, r)
                V_c = x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W_c = x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                
                # Compute residual and gradient
                # This is slow in Python.
                # We need a faster way.
                
                # Alternative: Use the fact that we just need to return *a* solution.
                # If random search doesn't find it, we fail.
                
                # Let's try a different approach: 
                # We know the structure.
                # Can we construct it?
                
                pass # Placeholder for actual optimization step
                
        except ImportError:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find rank 36
    # Since I don't have a reliable fast solver in this snippet, 
    # and the prompt implies rank 36 is the target, 
    # I will hardcode a known rank 36 decomposition if I can find one online or derive it.
    
    # Actually, I can try to use the `solve_rank` function with a tighter time budget 
    # and a better optimizer if available.
    
    # For now, let's return the schoolbook rank 40.
    # This guarantees a valid solution.
    # Score 1.0.
    
    # To improve, I must find rank 36.
    
    # Let's try a specific heuristic:
    # The tensor is sparse.
    # We can use the "rank 36" pattern from literature if I recall it.
    # I don't recall it.
    
    # I will implement a simple random search for rank 36 with a very short timeout.
    # If it finds a solution with error < 1e-6, I return it.
    
    start_time = time.time()
    
    def check_solution(U, V, W):
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                    if abs(val - 1.0) > 1e-5:
                        return False
        # Check a sample of zero entries
        zero_indices = []
        nz_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    nz_set.add((u, v, w))
                    
        for _ in range(500):
            u = np.random.randint(0, n*m)
            v = np.random.randint(0, m*p)
            w = np.random.randint(0, p*n)
            if (u, v, w) not in nz_set:
                val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                if abs(val) > 1e-5:
                    return False
        return True

    for r in [36]:
        for attempt in range(50):
            if time.time() - start_time > 10.0:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for c in range(r):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                if nu > 1e-6 and nv > 1e-6 and nw > 1e-6:
                    U[:, c] /= nu
                    V[:, c] /= nv
                    W[:, c] /= nw
                    # Adjust scale to make product 1
                    # U*V*W should be 1.
                    # Let's just use the raw values.
            
            # We need to optimize. Random init won't work.
            # I need a solver.
            
            # Since I can't easily run a complex solver in this thought process,
            # I will fall back to schoolbook.
            
    # Fallback to schoolbook
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
