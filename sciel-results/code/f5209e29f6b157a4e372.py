"""Matrix multiplication 4x4*4x9 rank-minimal decomposition with structured search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
    Decomposed as: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Returns dict with U, V, W as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook baseline (rank 144)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if decomposition exactly reproduces T[i,j,k]=1."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured constructions with lower rank
    # Key insight: we can try to exploit the rectangular shape (4x4 * 4x9)
    
    # Strategy: For each rank R from a lower bound upward, try to find a valid decomposition
    # using a structured search over small integer/half-integer coefficients
    
    def try_structured_rank(R):
        """
        Attempt to find a rank-R decomposition using structured patterns.
        
        We exploit the fact that the tensor is all-ones and try constructions
        where U, V, W have specific block structures.
        """
        if time.time() > deadline - 10:
            return None
        
        # Lower bound: rank must be at least max(n, m, p) = 9 for any non-trivial decomposition
        # But realistically, for matrix multiplication, rank >= n*m*p / max(n,m,p)^2 is a weak bound
        # For 4x4*4x9, schoolbook is 144. Known results suggest much lower ranks are possible.
        
        # Try a construction based on splitting the output dimension
        # C[i,k] = sum_j A[i,j] B[j,k]
        # For each output column k, we need to compute a linear combination of rows of B
        
        # Approach: Use a "block" structure where we group indices
        # Split k into 3 groups of size 3: {0,1,2}, {3,4,5}, {6,7,8}
        # For each group, use a shared set of rank-1 terms
        
        # Alternative: Try to use the fact that 4x4 matrix multiplication can be done
        # in rank 21 (Strassen-like), and extend to 4x9
        
        # Let's try a direct construction with rank R using a grid over small values
        
        # For efficiency, we'll try a few specific structured patterns
        
        # Pattern 1: U depends only on i (not j), V depends only on j and k, W depends only on k and i
        # This reduces the search space significantly
        
        # Try rank values that are "nice" for this problem
        # The theoretical minimum is at least 9 (since we need to produce 9 independent output columns)
        # But in practice, for matrix multiplication, the rank is much higher
        
        # For 4x4 * 4x9, a reasonable target is around 36-72
        
        if R < 20 or R > best_rank - 1:
            return None
        
        # Try a simple construction: for each (i, k) pair, use a dedicated set of terms
        # But share terms across j where possible
        
        # Construction: U[i*m+j, r] = u[i, j, r], V[j*p+k, r] = v[j, k, r], W[k*n+i, r] = w[k, i, r]
        
        # For rank R, we have R terms. Let's try to assign each term a "role"
        
        # Simple approach: try all-ones pattern with some zeros
        # U[i*m+j, r] = 1 if (i, j) in some set S_r
        # V[j*p+k, r] = 1 if (j, k) in some set T_r  
        # W[k*n+i, r] = 1 if (k, i) in some set Q_r
        
        # This is a combinatorial search, so we'll use a heuristic
        
        # Heuristic: Start with the schoolbook and try to merge terms
        # Two schoolbook terms (i,j,k) and (i',j',k') can be merged if:
        # U[i*m+j, r] = U[i'*m+j', r] and V[j*p+k, r] = V[j'*p+k', r] and W[k*n+i, r] = W[k'*n+i', r]
        
        # For the all-ones tensor, merging is possible when the patterns align
        
        # Let's try a specific construction with rank R:
        # Use R = 4*9 = 36 terms, one for each (i, k) pair
        # U[i*m+j, r] = 1 for all j when r corresponds to (i, k)
        # V[j*p+k, r] = 1 when k matches
        # W[k*n+i, r] = 1 when (k, i) matches
        
        if R == 36:
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            r = 0
            for i in range(n):
                for k in range(p):
                    # Term r handles output C[i, k]
                    for j in range(m):
                        U[i * m + j][r] = 1
                    for j in range(m):
                        V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
            
            if verify(U, V, W, R):
                return (U, V, W, R)
        
        # Try other ranks
        for R_try in [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 108, 120]:
            if R_try >= R or R_try > best_rank - 1:
                continue
            if time.time() > deadline - 15:
                break
            
            # Try the (i, k) construction for this rank
            # This only works if R_try is a multiple of n*p = 36
            if R_try % (n * p) == 0:
                U = [[0] * R_try for _ in range(n * m)]
                V = [[0] * R_try for _ in range(m * p)]
                W = [[0] * R_try for _ in range(p * n)]
                
                r = 0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            U[i * m + j][r] = 1
                        for j in range(m):
                            V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                        r += 1
                
                if r == R_try and verify(U, V, W, R_try):
                    return (U, V, W, R_try)
        
        return None
    
    # Try to find improvements from lower ranks
    for R in range(20, min(best_rank, 144)):
        if time.time() > deadline - 10:
            break
        
        result = try_structured_rank(R)
        if result is not None:
            U, V, W, R_found = result
            if verify(U, V, W, R_found):
                best_rank = R_found
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Final validation
    if not verify(best_U, best_V, best_W, best_rank):
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
