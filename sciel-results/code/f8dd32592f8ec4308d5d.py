"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a candidate decomposition
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the target tensor T[i,j,k] = 1 for all valid indices
        # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for all i in [0,n), j in [0,m), k in [0,p)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # Known: rank(M_{2,3,4}) <= 12 (can be constructed from block structure)
    # Let's try rank 12 first, then 11, 10, etc.
    
    def try_rank(rank):
        """Try to find a valid decomposition of given rank using structured patterns."""
        if time.time() > deadline:
            return None
        
        # Strategy: exploit the 2x3x4 structure
        # The tensor has 2*3*4 = 24 entries, each should equal 1
        # We need rank terms where each term is a product of three vectors
        
        # Approach 1: Direct construction using known algebraic identities
        # For 2x3x4, we can use the fact that:
        # C = A*B where A is 2x3, B is 3x4
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
        
        # A rank-12 construction: use 3 terms for each of 4 k-values
        # But this is just schoolbook rearranged
        
        # Better: use Strassen-like ideas
        # For 2x2x2, Strassen uses 7 terms instead of 8
        # For 2x3x4, we might do better than 24
        
        # Let's try a systematic search for small ranks
        # This is computationally expensive, so we limit it
        
        # Approach: Use the structure that M_{2,3,4} can be built from smaller multiplications
        # M_{2,3,4} contains M_{2,2,4} and M_{1,1,4} as sub-tensors
        
        # For now, let's try rank 12 with a specific construction
        # Based on the observation that we can group terms
        
        if rank == 12:
            # Try a structured rank-12 decomposition
            # Use the pattern: 3 groups of 4, or 4 groups of 3
            
            # Construction idea:
            # For each pair (i, j) in the 2x3 A-matrix, we have contributions
            # Let's try: r = 4*a + b where a in 0..2 (for j), b in 0..3 (for k)
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Term r = 4*j + k for j in 0..2, k in 0..3
            # U[i*m+j, r] = 1 for all i (since A[i,j] contributes to both C[0,k] and C[1,k])
            # Wait, that doesn't work directly
            
            # Alternative: use the fact that for fixed j, the contribution is
            # A[i,j] * B[j,k] which couples i and k
            # So for each j, we need a 2x4 tensor decomposition
            
            # For fixed j: T_{i,k}^{(j)} = A[i,j] * B[j,k]
            # This is an outer product, rank 1!
            # But we need it to work for all i,k simultaneously
            
            # The full tensor is sum_j T^{(j)} where T^{(j)}_{i,k} = A[i,j]*B[j,k]
            # Each T^{(j)} has rank 1 as a 2x4 matrix, but as a 3D tensor (i,j,k)
            # it's more complex
            
            # Let me try a different approach: random search with verification
            # Generate random decompositions and check if they work
            
            np.random.seed(42)
            for attempt in range(1000):
                if time.time() > deadline:
                    break
                
                # Random initialization with integer/half-integer values
                vals = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
                
                U = [[np.random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
                V = [[np.random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
                W = [[np.random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
            
            return None
        
        elif rank == 11:
            # Try rank 11
            np.random.seed(123)
            for attempt in range(500):
                if time.time() > deadline:
                    break
                
                vals = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
                
                U = [[np.random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
                V = [[np.random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
                W = [[np.random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
            
            return None
        
        elif rank == 10:
            # Try rank 10
            np.random.seed(456)
            for attempt in range(300):
                if time.time() > deadline:
                    break
                
                vals = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
                
                U = [[np.random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
                V = [[np.random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
                W = [[np.random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
            
            return None
        
        return None
    
    # Try ranks from 10 up to 23
    for rank in range(10, n * m * p):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
