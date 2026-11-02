"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 20s safety margin
    
    # Try to find rank-20 decomposition with half-integers
    result = try_rank_20(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 24
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def try_rank_20(n, m, p, deadline):
    """
    Try to construct a rank-20 decomposition with half-integer entries.
    Uses the structure: M_{2,3,4} = M_{2,3,3} + M_{2,3,1}
    where rank(M_{2,3,3}) = 14 and rank(M_{2,3,1}) = 6.
    """
    r = 20
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eq = len(triples)  # 24 equations
    
    I_idx = [i * m + j for i, j, k in triples]
    J_idx = [j * p + k for i, j, k in triples]
    K_idx = [k * n + i for i, j, k in triples]
    
    # Strategy: Use the known algebraic structure
    # For M_{2,3,4}, we can use a decomposition based on the fact that
    # the tensor can be written as a sum of 20 rank-1 tensors.
    # 
    # Let's try a structured approach:
    # Split into two parts: k=0,1,2 (3 columns) and k=3 (1 column)
    # Part 1: 2x3x3 tensor (rank 14)
    # Part 2: 2x3x1 tensor (rank 6)
    
    # For the 2x3x1 part (k=3 only):
    # We need: for each (i,j), sum_c U[i*m+j, c] * V[j*p+3, c] * W[3*n+i, c] = 1
    # This is a 2x3 = 6 equation system with 6 unknowns per term
    # Rank 6: use one term per (i,j) pair
    # U[i*m+j, 14+i*m+j] = 1, V[j*p+3, 14+i*m+j] = 1, W[3*n+i, 14+i*m+j] = 1
    
    # For the 2x3x3 part (k=0,1,2):
    # We need rank 14. The known decomposition for 2x3x3 uses 14 terms.
    # Let's use a structured approach.
    
    # Initialize with the 2x3x1 part (terms 14-19)
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Part 2: 2x3x1 (k=3), 6 terms (indices 14-19)
    for i in range(n):
        for j in range(m):
            c = 14 + i*m + j
            U[i*m + j, c] = 1.0
            V[j*p + 3, c] = 1.0
            W[3*n + i, c] = 1.0
    
    # Now we need to fill in terms 0-13 for the 2x3x3 part
    # The 2x3x3 tensor needs: for each (i,j,k) with k in {0,1,2}:
    #   sum_{c=0}^{13} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 18 equations (2*3*3) with 14*3 = 42 unknowns
    # (each of the 14 terms has 3 values in U, V, W that contribute)
    
    # Let's use a parameterized approach. 
    # Known result: rank(2x3x3) = 14, and there exists a decomposition.
    # Let's try to find it using optimization.
    
    # Use scipy's least_squares or similar
    try:
        from scipy.optimize import least_squares
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        # Flatten the 14*3 = 42 variables
        # Variables: for c in 0..13, we have U[i*m+j,c] for each (i,j) in 2x3,
        #           V[j*p+k,c] for each (j,k) in 3x3, W[k*n+i,c] for each (k,i) in 3x2
        # Total per term: 6 + 9 + 6 = 21 values, but many are zero by structure
        
        # Actually, let's just optimize all 14*3*3 = 126 variables (but with constraints)
        # Better: optimize the 14 terms, each contributing to U, V, W
        
        # Let's set up: for each term c in 0..13, we have:
        #   u_c[i,j] for i in 0..1, j in 0..2 (6 values)
        #   v_c[j,k] for j in 0..2, k in 0..2 (9 values)  
        #   w_c[k,i] for k in 0..2, i in 0..1 (6 values)
        # Total: 14 * 21 = 294 variables
        
        # This is too many. Let's use a smarter parameterization.
        
        # Alternative: Use the fact that we can scale. 
        # Let's try random initialization with ALS, but with better initialization.
        
        # Initialize terms 0-13 randomly
        np.random.seed(42)
        for c in range(14):
            U[:6, c] = np.random.randn(6)
            V[:9, c] = np.random.randn(9)
            W[:6, c] = np.random.randn(6)
            # Zero out the k=3 parts for these terms
            V[9:, c] = 0
            W[6:, c] = 0
        
        # ALS optimization for the 2x3x3 part only
        time_limit = deadline - time.time() - 5
        if time_limit > 2:
            U, V, W = als_optimize_2x3x3(U, V, W, n, m, p, time_limit)
            
            # Check if we got a good solution
            if U is not None:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r, triples, I_idx, J_idx, K_idx):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        # Try multiple random restarts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + attempt)
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Re-initialize the 2x3x1 part
            for i in range(n):
                for j in range(m):
                    c = 14 + i*m + j
                    U[i*m + j, c] = 1.0
                    V[j*p + 3, c] = 1.0
                    W[3*n + i, c] = 1.0
            
            # Random init for 2x3x3 part
            for c in range(14):
                U[:6, c] = np.random.randn(6)
                V[:9, c] = np.random.randn(9)
                W[:6, c] = np.random.randn(6)
                V[9:, c] = 0
                W[6:, c] = 0
            
            time_limit = deadline - time.time() - 5
            if time_limit > 1:
                U, V, W = als_optimize_2x3x3(U, V, W, n, m, p, min(time_limit, 3.0))
                
                if U is not None:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify(U_r, V_r, W_r, triples, I_idx, J_idx, K_idx):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
    
    # Try a completely different approach: direct construction
    result = try_direct_construction(n, m, p, r, deadline)
    if result is not None:
        return result
    
    return None

def als_optimize_2x3x3(U, V, W, n, m, p, time_limit):
    """ALS optimization for the 2x3x3 part (k=0,1,2 only)."""
    r = 14  # Only using first 14 terms
    triples_233 = [(i, j, k) for i in range(n) for j in range(m) for k in range(3)]
    num_eq = len(triples_233)  # 18
    
    I_idx = [i * m + j for i, j, k in triples_233]
    J_idx = [j * p + k for i, j, k in triples_233]
    K_idx = [k * n + i for i, j, k in triples_233]
    
    U_curr = U[:, :r].copy()
    V_curr = V[:, :r].copy()
    W_curr = W[:, :r].copy()
    
    for step in range(100):
        if time.time() > time_limit:
            break
        
        # Solve for U
        M_U = np.zeros((num_eq, n * m * r))
        for e in range(num_eq):
            for c in range(r):
                M_U[e, I_idx[e] * r + c] = V_curr[J_idx[e], c] * W_curr[K_idx[e], c]
        
        U_flat, _, _, _ = np.linalg.lstsq(M_U, np.ones(num_eq), rcond=None)
        U_curr = U_flat.reshape(n * m, r)
        
        # Solve for V
        M_V = np.zeros((num_eq, m * p * r))
        for e in range(num_eq):
            for c in range(r):
                M_V[e, J_idx[e] * r + c] = U_curr[I_idx[e], c] * W_curr[K_idx[e], c]
        
        V_flat, _, _, _ = np.linalg.lstsq(M_V, np.ones(num_eq), rcond=None)
        V_curr = V_flat.reshape(m * p, r)
        
        # Solve for W
        M_W = np.zeros((num_eq, p * n * r))
        for e in range(num_eq):
            for c in range(r):
                M_W[e, K_idx[e] * r + c] = U_curr[I_idx[e], c] * V_curr[J_idx[e], c]
        
        W_flat, _, _, _ = np.linalg.lstsq(M_W, np.ones(num_eq), rcond=None)
        W_curr = W_flat.reshape(p * n, r)
        
        # Check error
        error = 0.0
        for e in range(num_eq):
            val = 0.0
            for c in range(r):
                val += U_curr[I_idx[e], c] * V_curr[J_idx[e], c] * W_curr[K_idx[e], c]
            error += (val - 1.0) ** 2
        
        if error < 1e-8:
            # Combine with the 2x3x1 part
            U_full = U.copy()
            V_full = V.copy()
            W_full = W.copy()
            U_full[:, :r] = U_curr
            V_full[:, :r] = V_curr
            W_full[:, :r] = W_curr
            return U_full, V_full, W_full
    
    # Return None if not converged
    return None, None, None

def verify(U, V, W, triples, I_idx, J_idx, K_idx):
    r = U.shape[1]
    for e, (i, j, k) in enumerate(triples):
        val = 0.0
        for c in range(r):
            val += U[I_idx[e], c] * V[J_idx[e], c] * W[K_idx[e], c]
        if abs(val - 1.0) > 1e-9:
            return False
    return True

def try_direct_construction(n, m, p, r, deadline):
    """Try to directly construct a rank-20 decomposition."""
    # Based on known algebraic results, the rank of 2x3x4 is 20.
    # Let's try to find a half-integer solution using a structured approach.
    
    # The tensor is: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try a construction where we use the structure of the indices.
    # For the 2x3x4 case, let's try:
    # - Use 6 terms for the "diagonal" part
    # - Use 14 terms for the "off-diagonal" part
    
    # Actually, let's try a simpler approach: use the fact that
    # we can write 1 = sum of products with simple coefficients.
    
    # Try: for each (i,j,k), we want the sum to be 1.
    # Let's use a construction where U, V, W have simple patterns.
    
    # One idea: use the Kronecker product structure
    # 2x3x4 = (2x1x1) * (1x3x1) * (1x1x4) but this gives rank 24
    
    # Better idea: use the known decomposition from the literature.
    # The rank-20 decomposition of 2x3x4 can be constructed as follows:
    # 
    # Let's try to find it by solving the system of equations directly.
    # We have 24 equations and 20*3 = 60 unknowns (but with structure).
    
    # For now, return None to fall back to schoolbook
    return None
# EVOLVE-BLOCK-END
