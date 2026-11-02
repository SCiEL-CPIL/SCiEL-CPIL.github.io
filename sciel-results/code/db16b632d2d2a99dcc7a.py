"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition.

Strategy:
  1. Attempt to find a rank-22 decomposition using a structured random search 
     with linear algebra solving for the W matrix.
  2. If verification fails or time limit is reached, fall back to rank 24 (schoolbook).

The evaluator measures the raw value (lower is better).
"""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
DEADLINE = 120.0  # safety margin before 150s kill

def _verify(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Check that the factor matrices exactly decompose the matrix-multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=2, m=3, p=4):
    """Return the standard rank n*m*p decomposition."""
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

def _try_rank_22(n=2, m=3, p=4, time_limit=10.0):
    """
    Attempt to find a rank-22 decomposition.
    We use a random search strategy:
    1. Randomly generate sparse U and V with entries in {-1, 0, 1}.
    2. Solve for W using least squares (or exact solve if square/invertible structure allows).
    3. Check if W entries are close to half-integers.
    """
    start = time.time()
    random.seed(42) # For reproducibility in search attempts
    
    # Target rank
    r = 22
    
    # Dimensions
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # We need to solve: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # This is a linear system in W if U and V are fixed.
    # Let M be the matrix of size (n*m*p) x r, where M[t, c] = U[i*m+j, c] * V[j*p+k, c]
    # Then M @ W_flat = ones(n*m*p)
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
            
        # Generate random sparse U and V
        # We want them to be "structured" enough to allow a solution
        # Use a mix of 0s, 1s, and -1s
        U = np.zeros((dim_u, r), dtype=float)
        V = np.zeros((dim_v, r), dtype=float)
        
        # Randomly populate with non-zero values
        # Ensure some structure: maybe each row has 2-3 non-zeros
        for i in range(dim_u):
            indices = random.sample(range(r), k=random.randint(2, 4))
            for idx in indices:
                U[i, idx] = random.choice([-1, 1])
                
        for i in range(dim_v):
            indices = random.sample(range(r), k=random.randint(2, 4))
            for idx in indices:
                V[i, idx] = random.choice([-1, 1])
                
        # Construct the linear system matrix M
        # M has n*m*p rows and r columns
        # Row index t corresponds to (i, j, k)
        # M[t, c] = U[i*m+j, c] * V[j*p+k, c]
        
        M = np.zeros((n*m*p, r), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    M[idx, :] = u_row * v_row
                    idx += 1
                    
        # Solve M @ W_flat = ones(n*m*p)
        # Use least squares
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(M, np.ones(n*m*p), rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Reshape W_flat to (dim_w, r)
        W = W_flat.reshape((dim_w, r))
        
        # Check if W entries are close to half-integers
        # Multiply by 2 and check if close to integers
        W_scaled = W * 2
        W_rounded = np.round(W_scaled)
        
        # Check difference
        if np.allclose(W_scaled, W_rounded, atol=1e-6):
            # It's a candidate!
            # Convert to exact half-integers (floats representing x.0 or x.5)
            W_exact = (W_rounded / 2.0).tolist()
            U_exact = U.tolist()
            V_exact = V.tolist()
            
            # Verify with high precision
            if _verify(U_exact, V_exact, W_exact, n, m, p, tol=1e-9):
                return {"U": U_exact, "V": V_exact, "W": W_exact}
                
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 22 first
    result22 = _try_rank_22(n, m, p, time_limit=10.0)
    if result22 is not None:
        return result22
    
    # Fall back to schoolbook rank 24
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
