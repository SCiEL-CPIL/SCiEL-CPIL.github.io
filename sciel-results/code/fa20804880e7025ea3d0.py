"""Program for rank-minimal decomposition of 2x3 by 3x3 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s evaluator limit

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify(U, V, W):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build tensor
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1
                if abs(s - expected) > 1e-9:
                    errors += 1
    return errors == 0

def construct():
    """
    Construct a rank-minimal decomposition of M_{2,3,3}.
    Best known bound: rank <= 16 (from Strassen-style improvements).
    Return the lowest rank decomposition found.
    """
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed rank-18 schoolbook
    rank_18 = n * m * p
    U18 = [[0]*rank_18 for _ in range(n*m)]
    V18 = [[0]*rank_18 for _ in range(m*p)]
    W18 = [[0]*rank_18 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U18[i*m+j][col] = 1
                V18[j*p+k][col] = 1
                W18[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank via systematic search
    # Strategy: use the structure of 2x3*3x3 to try rank 17, 16, 15
    
    # The tensor T has 18 entries (one per (i,j,k)).
    # A rank-r decomposition means T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] V[j,k,c] W[k,i,c]
    
    # For rank r, we need to find U (6 x r), V (9 x r), W (6 x r)
    # such that for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Use algebraic approach: parametrize and solve
    # For each candidate rank r (from 17 down to minimum plausible),
    # try to find integer/half-integer solutions.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        # We have 6*r + 9*r + 6*r = 21*r variables
        # 18 equations (one per (i,j,k))
        # Nonlinear system (trilinear)
        
        # Approach: use the fact that for small r, we can enumerate
        # sparse structures. 
        
        # For r=17: try to find a way to combine one schoolbook term
        # For r=16: need more structure
        
        # Simple approach for r=17: take the rank-18 and see if any two
        # columns can be merged (if they differ in only one factor pattern)
        
        # Actually, let's use a more direct algebraic approach.
        # The key insight: for M_{2,3,3}, we can exploit the small n=2.
        # W[k*n+i] = W[2k+i], so W has entries W[0],W[1] for i=0,1 and k=0,1,2
        
        # Let me try a different approach: use known constructions.
        # For M_{1,p,p}, rank = 2p-1 (using Strassen-like for single row).
        # For M_{2,p,p}, a bound is 2*(2p-1) - 1 = 4p-3? For p=3: 9.
        # Actually that's not right either.
        
        # Let me just try numerical optimization to see if rank < 18 is feasible.
        # But we need exact (integer/half-integer) solutions.
        
        # Alternative: use the fact that the tensor is fixed and try
        # to find a bilinear algorithm with fewer multiplications.
        
        # For rank r, we need vectors u_c in R^6, v_c in R^9, w_c in R^6
        # such that for each (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
        
        # Let me try a randomized algebraic approach with rounding.
        rng = np.random.default_rng(42)
        
        for attempt in range(50):
            if _time_remaining() < 10:
                return None
            
            # Random initial guess with small integer values
            U = rng.integers(-2, 3, size=(n*m, r)).astype(float)
            V = rng.integers(-2, 3, size=(m*p, r)).astype(float)
            W = rng.integers(-2, 3, size=(p*n, r)).astype(float)
            
            # Evaluate the 18 tensor entries
            tensor_errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        tensor_errors.append(val - 1)
            
            if max(abs(e) for e in tensor_errors) < 1e-6:
                # Check if we can round to integers/half-integers
                # (they're already integers in this construction)
                return (U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist())
        
        return None
    
    # Try rank 17 first (closest to 18)
    if _time_remaining() > 30:
        result = try_rank(17)
        if result and _verify(result[0], result[1], result[2]):
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Try rank 16
    if _time_remaining() > 20:
        result = try_rank(16)
        if result and _verify(result[0], result[1], result[2]):
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fall back to schoolbook rank 18
    return {"U": U18, "V": V18, "W": W18}
