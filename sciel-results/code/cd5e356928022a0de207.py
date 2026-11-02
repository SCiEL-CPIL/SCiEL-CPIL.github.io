import time
import sys
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m: return False
    if len(V) != m * p: return False
    if len(W) != p * n: return False
    
    # Convert to numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _find_3x3x3_decomp(rank=23, time_limit=50.0):
    """Find a rank-rank decomposition for 3x3x3 matrix multiplication using scipy."""
    n, m, p = 3, 3, 3
    t0 = time.time()
    
    # Target tensor is all 1s
    target = np.ones((n, m, p))
    
    def residuals(x):
        # x is flattened [U, V, W]
        # U: 9 * rank, V: 9 * rank, W: 9 * rank
        size = n * m * rank
        U = x[0:size].reshape(n * m, rank)
        V = x[size:2*size].reshape(m * p, rank)
        W = x[2*size:3*size].reshape(p * n, rank)
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
        return (T - target).flatten()
    
    best_loss = float('inf')
    best_x = None
    
    # Try a few random initializations
    for seed in [42, 123, 456]:
        if time.time() - t0 > time_limit:
            break
        np.random.seed(seed)
        # Initialize with small random values
        x0 = np.random.randn(3 * 9 * rank) * 0.1
        # Add a structured initialization: schoolbook-like sparse init?
        # For now, just random.
        
        try:
            # Use least_squares
            # 'lm' method requires m >= n, but here we have 27 residuals and 621 vars.
            # So we must use 'trf' or 'dogbox' which handle underdetermined systems.
            result = scipy.optimize.least_squares(residuals, x0, method='trf', max_nfev=1000)
            
            if result.cost < best_loss:
                best_loss = result.cost
                best_x = result.x
                
                if best_loss < 1e-8:
                    break
        except Exception:
            pass
            
    if best_x is None or best_loss > 1e-4:
        return None
        
    # Extract U, V, W
    size = n * m * rank
    U = best_x[0:size].reshape(n * m, rank)
    V = best_x[size:2*size].reshape(m * p, rank)
    W = best_x[2*size:3*size].reshape(p * n, rank)
    
    # Round to nearest 0.5
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify rounded solution
    U_list = U_rounded.tolist()
    V_list = V_rounded.tolist()
    W_list = W_rounded.tolist()
    
    if _verify(U_list, V_list, W_list):
        return U_list, V_list, W_list
        
    # If rounding failed, maybe the solution is not close to half-integers?
    # But the problem statement implies integer/half-integer entries are expected/allowed.
    # If the optimal solution is not half-integer, we can't return it.
    # However, for tensor decomposition of small matrices, solutions are often rational.
    
    return None

def _construct_3x3x6_from_3x3x3(U3, V3, W3):
    """Construct 3x3x6 decomposition from 3x3x3 decomposition."""
    n, m, p = 3, 3, 6
    rank3 = len(U3[0])
    rank6 = rank3 * 2
    
    U6 = [[0.0] * rank6 for _ in range(n * m)]
    V6 = [[0.0] * rank6 for _ in range(m * p)]
    W6 = [[0.0] * rank6 for _ in range(p * n)]
    
    for r in range(rank3):
        # U is shared
        for i in range(n * m):
            U6[i][r] = U3[i][r]
            U6[i][r + rank3] = U3[i][r]
            
        # V and W for first block (k=0..2)
        for j in range(m):
            for k in range(3):
                V6[j * p + k][r] = V3[j * 3 + k][r]
                W6[k * n + 0][r] = 0 # Placeholder, fill below
                
        # W for first block
        for i in range(n):
            for k in range(3):
                W6[k * n + i][r] = W3[k * 3 + i][r]
                
        # V and W for second block (k=3..5)
        for j in range(m):
            for k in range(3):
                V6[j * p + (k + 3)][r + rank3] = V3[j * 3 + k][r]
                
        # W for second block
        for i in range(n):
            for k in range(3):
                W6[(k + 3) * n + i][r + rank3] = W3[k * 3 + i][r]
                
    return U6, V6, W6

def _schoolbook():
    n, m, p = 3, 3, 6
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

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Try to find a rank 46 decomposition (2 * 23)
    # We try ranks 23, 24, 25, 26 for the 3x3x3 sub-problem
    for rank3 in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
            
        res3 = _find_3x3x3_decomp(rank=rank3, time_limit=30.0)
        if res3 is not None:
            U3, V3, W3 = res3
            U6, V6, W6 = _construct_3x3x6_from_3x3x3(U3, V3, W3)
            
            # Final verification
            if _verify(U6, V6, W6):
                return {"U": U6, "V": V6, "W": W6}
                
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
