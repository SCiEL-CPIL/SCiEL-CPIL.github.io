"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _block_decompose(n, m, p, m_split, p_split):
    """
    Decompose n×m×p by splitting m into m1+m2 and p into p1+p2.
    This gives rank = 2 * rank(n×m1×p1) + 2 * rank(n×m2×p2) if we use independent decompositions.
    But we can do better by sharing structure.
    
    Actually, let's use: C = A[:,:m1]*B[:m1,:] + A[:,m1:]*B[m1:,:]
    If we have decompositions for (n,m1,p) and (n,m2,p), we can combine them.
    """
    m1 = m_split
    m2 = m - m_split
    p1 = p_split
    p2 = p - p_split
    
    # This approach doesn't directly help unless we have good sub-decompositions
    pass

def _try_rank_am(n, m, p, rank, max_iter=500, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _construct_strassen_3x4x6():
    """
    Try to construct a low-rank decomposition using Strassen-like ideas.
    
    For 3×4×6, we can split:
    - The 4 columns of A into two groups of 2
    - The 6 rows of B into two groups of 3
    
    C = A[:,:2]*B[:2,:] + A[:,2:]*B[2:,:]
    
    Each term is 3×2×6. For 3×2×6, we can further split:
    3×2×6: C = A[:,:2]*B[:2,:3] + A[:,:2]*B[:2,3:]
    Each is 3×2×3.
    
    So total: 4 blocks of 3×2×3.
    If rank(3×2×3) = r, total rank = 4r.
    
    Known: rank(3×2×3) ≤ 14, so total ≤ 56.
    But we can do better by sharing structure.
    
    Let me try a direct construction for rank 24.
    """
    n, m, p = 3, 4, 6
    
    # Try to use the structure: split m=4 into 2+2, p=6 into 3+3
    # This gives 4 sub-problems of 3×2×3
    # If we can find rank(3×2×3) = 6, total = 24
    
    # For 3×2×3, schoolbook rank is 18.
    # Known minimum is 14, but let's see if we can do 6 with some structure.
    
    # Actually, let's try a different approach: use the fact that 6 = 2*3
    # and 4 = 2*2, so we can use Kronecker product structure.
    
    # Let me try to find a rank 24 decomposition by search.
    pass

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a low-rank decomposition by searching from low to high rank
    # Start with a reasonable lower bound.
    # For 3×4×6, the rank is at least max(3*4, 4*6, 3*6) = max(12, 24, 18) = 24
    # (This is a weak lower bound, but let's try 24 first)
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Try ranks from 24 upward
    for rank in range(24, best_rank + 1):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > _DEADLINE - 5.0:
                break
            
            result = _try_rank_am(n, m, p, rank, max_iter=200, tol=1e-8, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_result = (U, V, W)
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
