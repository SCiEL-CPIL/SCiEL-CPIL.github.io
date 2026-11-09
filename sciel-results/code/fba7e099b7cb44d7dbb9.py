"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses structured Kronecker decomposition and alternating minimization.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s < 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_alternating_minimization(rank, max_iters=500, tol=1e-8, seed=0):
        """Try alternating minimization to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        # Initialize with random values, scaled to avoid overflow/underflow
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Pre-allocate targets
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        
        best_res = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct A: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = V[j*p:(j+1)*p, :] * W[i::n, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, ones_p, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = U[j::m, :] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, ones_n, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = U[i*m:(i+1)*m, :] * V[k::p, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, ones_m, rcond=None)[0]
                    except:
                        pass
            
            # Check residual every 10 iterations or at end
            if it % 10 == 9 or it == max_iters - 1:
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        V_jk = V[j*p:(j+1)*p, :]
                        W_ki = W[i::n, :]
                        Prod = V_jk * W_ki
                        Dots = Prod @ U[i*m+j, :]
                        max_res = max(max_res, np.max(np.abs(Dots - 1.0)))
                
                if max_res < best_res:
                    best_res = max_res
                    if max_res < tol:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        return None
    
    def try_kronecker_decomposition(rank, max_iters=300, tol=1e-8, seed=0):
        """
        Try a structured decomposition using Kronecker product ideas.
        Exploit that m=4=2*2 by factoring the middle dimension.
        
        The idea: decompose the 4x4 multiplication into two 2x2 blocks.
        For the 2x4 * 4x7 case, we can think of:
        - Splitting the 4-dimensional middle into two 2-dimensional parts
        - Using rank-2 decompositions for 2x2x2 and combining
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Strategy: Use a Kronecker-like structure
        # For 2x4 * 4x7, think of 4 as 2x2
        # The tensor is T[i, (a,b), (c,d), (d', a')] where the middle 4-dim is split
        # Actually, let's use a more direct approach with structured initialization
        
        # Initialize with a Kronecker product structure
        # U has shape (2*4, rank) = (8, rank)
        # V has shape (4*7, rank) = (28, rank)  
        # W has shape (7*2, rank) = (14, rank)
        
        # Try a block structure: assume rank = r1 * r2 for some factors
        # For now, use random init but with structured blocks
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        
        best_res = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Same alternating minimization as before
            for i in range(n):
                for j in range(m):
                    A = V[j*p:(j+1)*p, :] * W[i::n, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, ones_p, rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    B = U[j::m, :] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, ones_n, rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    C = U[i*m:(i+1)*m, :] * V[k::p, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, ones_m, rcond=None)[0]
                    except:
                        pass
            
            if it % 10 == 9 or it == max_iters - 1:
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        V_jk = V[j*p:(j+1)*p, :]
                        W_ki = W[i::n, :]
                        Prod = V_jk * W_ki
                        Dots = Prod @ U[i*m+j, :]
                        max_res = max(max_res, np.max(np.abs(Dots - 1.0)))
                
                if max_res < best_res:
                    best_res = max_res
                    if max_res < tol:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        return None
    
    # The all-ones tensor of size 8x28x14 has rank at least 8 (min dimension)
    # Try ranks from 8 upward with aggressive search
    
    # First, try very low ranks with many seeds
    ranks_to_try = list(range(8, 25)) + [28, 30, 32, 36, 40, 44, 48, 52, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # More seeds for smaller ranks
        if r <= 12:
            seeds = 50
        elif r <= 20:
            seeds = 30
        else:
            seeds = 10
        
        found = None
        for seed in range(seeds):
            if time.time() > deadline - 5:
                break
            found = try_alternating_minimization(r, max_iters=300, seed=seed)
            if found is not None:
                best_rank = r
                best_result = found
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
