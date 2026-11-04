"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The tensor T is defined such that for A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition uses U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 if the indices are compatible (j matches between A and B, and i matches in output)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    def verify_full(U, V, W):
        """Full verification - slower but exact."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                if j != j2:
                    vec = U[a] * V[b] * W
                    if np.max(np.abs(vec)) > 1e-9:
                        return False
                else:
                    vec = U[a] * V[b] * W
                    for c in range(p * n):
                        i2 = c % n
                        target = 1.0 if i2 == i else 0.0
                        if abs(vec[c] - target) > 1e-9:
                            return False
        return True
    
    # Generate schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try structured decompositions using Strassen-like block decomposition
    # For 4x4 * 4x9, we can use block structure
    # A is 4x4, B is 4x9
    
    # Try to construct a decomposition using block methods
    # Split A into 2x2 blocks of size 2x2, B into 2x? blocks
    
    # Actually, let's try a different approach: use the fact that we can
    # potentially use Kronecker products or exploit the rectangular structure
    
    # For 4x4 * 4x9:
    # n=4, m=4, p=9
    # n*m = 16, m*p = 36, p*n = 36
    
    # Let's try a coordinate descent approach with better initialization
    
    best_rank = n * m * p  # schoolbook = 144
    best_result = None
    
    # Try to find a low-rank decomposition using structured search
    # Key insight: use block structure and Strassen-like ideas
    
    # For 4x4 * 4x4, Strassen gives rank 49 (7*7)
    # For 4x4 * 4x9, we can potentially do better
    
    # Let's try a gradient-free optimization with multiple restarts
    # and structured initialization
    
    def try_decomposition(target_rank, seed=42):
        """Try to find a decomposition with given rank using iterative refinement."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with structured random matrices
        U = np.random.uniform(-1.0, 1.0, (n * m, target_rank))
        V = np.random.uniform(-1.0, 1.0, (m * p, target_rank))
        W = np.random.uniform(-1.0, 1.0, (p * n, target_rank))
        
        # Build the coefficient matrix B for the linear system
        # B has shape (n*m*m*p, target_rank)
        B = np.zeros((n * m * m * p, target_rank))
        idx = 0
        for a in range(n * m):
            for b in range(m * p):
                B[idx] = U[a] * V[b]
                idx += 1
        
        # Compute target tensor entries
        # T[a,b,c] = 1 iff (a%m == b//p) and (a//m == c%n)
        
        # For each c, solve B @ W[c,:] = t_c
        # where t_c[a*m*p + b] = T[a,b,c]
        
        for it in range(30):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each c, solve least squares
            for c in range(p * n):
                t_c = np.zeros(n * m * m * p)
                idx2 = 0
                for a in range(n * m):
                    i = a // m
                    j = a % m
                    for b in range(m * p):
                        j2 = b // p
                        if j == j2:
                            i2 = c % n
                            if i2 == i:
                                t_c[idx2] = 1.0
                        idx2 += 1
                
                try:
                    W[c] = np.linalg.lstsq(B, t_c, rcond=None)[0]
                except:
                    W[c] = np.random.uniform(-0.1, 0.1, target_rank)
            
            # Update U: for each a, we need to solve for U[a,:]
            # The equation is: sum_c W[c,:] * (V[b,:] * U[a,:]) = T[a,b,c] for all b,c
            # This is more complex. Let's use a different approach.
            
            # Actually, let's just check the error and stop if it's small enough
            err = 0
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        s = np.dot(U[a] * V[b], W[c])
                        i = a // m
                        j = a % m
                        j2 = b // p
                        i2 = c % n
                        target = 1.0 if (j == j2 and i2 == i) else 0.0
                        err += (s - target) ** 2
            err = np.sqrt(err)
            
            if err < 1e-6:
                # Verify fully
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if verify_full(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try various rank values
    for target_rank in [80, 90, 100, 110, 120, 130, 140]:
        if time.time() > deadline - 15:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(target_rank, seed=42 + attempt * 13)
            if result is not None:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
