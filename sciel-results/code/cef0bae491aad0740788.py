"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses alternating minimization to find low-rank decompositions.
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
                    A = V[j*p:(j+1)*p, :] * W[i::n, :] # W[k*n+i] for k in range(p) is W[i::n] if stride is n?
                    # Wait, W index is k*n+i.
                    # k=0 -> i, k=1 -> n+i, ... k=p-1 -> (p-1)n+i.
                    # So W[i::n] is correct for k=0..p-1 if W is flattened k*n+i.
                    # Yes.
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, ones_p, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Construct B: B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # U indices: i*m+j for i in 0..n-1.
                    # U[i*m+j] is U[j::m] ? No.
                    # U is (n*m, rank). Row index is i*m+j.
                    # For fixed j, i varies. Indices: j, m+j, 2m+j...
                    # So U[j::m] gives rows j, m+j, ...
                    # W indices: k*n+i for i in 0..n-1.
                    # For fixed k, i varies. Indices: k*n, k*n+1...
                    # So W[k*n:(k+1)*n]
                    B = U[j::m, :] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, ones_n, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Construct C: C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # U indices: i*m+j for j in 0..m-1.
                    # For fixed i, j varies. Indices: i*m, i*m+1...
                    # So U[i*m:(i+1)*m]
                    # V indices: j*p+k for j in 0..m-1.
                    # For fixed k, j varies. Indices: k, p+k, 2p+k...
                    # So V[k::p]
                    C = U[i*m:(i+1)*m, :] * V[k::p, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, ones_m, rcond=None)[0]
                    except:
                        pass
            
            # Check residual every 10 iterations or at end
            if it % 10 == 9 or it == max_iters - 1:
                # Vectorized residual check
                # Res[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # This is hard to vectorize fully without large intermediate tensors, but n,m,p are small.
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        # U_row = U[i*m+j]
                        # For each k, dot(U_row, V[j*p+k] * W[k*n+i])
                        # We can do this for all k at once?
                        # V_jk = V[j*p:(j+1)*p] # shape (p, rank)
                        # W_ki = W[i::n] # shape (p, rank)
                        # Prod = V_jk * W_ki # shape (p, rank)
                        # Dots = Prod @ U_row # shape (p,)
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
    
    # Try alternating minimization for various ranks
    # Focus on lower ranks first with more seeds
    ranks_to_try = [28, 29, 30, 32, 34, 36, 40, 44, 48, 52, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # More seeds for smaller ranks
        seeds = 30 if r <= 30 else 10
        
        found = None
        for seed in range(seeds):
            if time.time() > deadline - 5:
                break
            found = try_alternating_minimization(r, max_iters=400, seed=seed)
            if found is not None:
                best_rank = r
                best_result = found
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
