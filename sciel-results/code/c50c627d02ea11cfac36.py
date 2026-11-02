import time
import numpy as np

def _find_3x3_decomposition(rank, max_attempts=500, seed=42):
    """
    Try to find a rank-r decomposition for 3x3x3 matrix multiplication.
    Uses randomized approach: fix U and V randomly, solve for W.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        # Generate random U and V
        U = rng.randn(9, rank)
        V = rng.randn(9, rank)
        
        # The equation: for all (i,j,k), sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*3+i,c] = 1
        # This is linear in W. Let's set up the system.
        # There are 27 equations (for each i,j,k) and 9*rank unknowns (W has 9 rows, rank cols)
        # But the equations couple the W entries.
        
        # Let's build the linear system. For each (i,j,k), the equation is:
        # sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*3+i,c] = 1
        # This involves only W[k*3+i, :] for fixed (k,i), and the coefficient for W[k*3+i,c] is U[i*3+j,c]*V[j*3+k,c]
        
        # So for fixed (k,i), we have 3 equations (one for each j) that involve W[k*3+i, :]:
        # For j=0: sum_c U[i*3+0,c]*V[0*3+k,c]*W[k*3+i,c] = 1
        # For j=1: sum_c U[i*3+1,c]*V[1*3+k,c]*W[k*3+i,c] = 1
        # For j=2: sum_c U[i*3+2,c]*V[2*3+k,c]*W[k*3+i,c] = 1
        
        # So for each (k,i), we have a 3 x rank system: A_{k,i} * W[k*3+i, :]^T = [1,1,1]^T
        # where A_{k,i}[j,c] = U[i*3+j,c] * V[j*3+k,c]
        
        W = np.zeros((9, rank))
        valid = True
        for k in range(3):
            for i in range(3):
                # Build 3 x rank matrix
                A = np.zeros((3, rank))
                for j in range(3):
                    for c in range(rank):
                        A[j, c] = U[i*3+j, c] * V[j*3+k, c]
                
                # Solve A * W_row^T = [1,1,1]^T
                # This is underdetermined if rank > 3
                try:
                    # Use least squares
                    W_row, residuals, rank_val, sv = np.linalg.lstsq(A, np.ones(3), rcond=None)
                    # Check if solution is exact
                    if np.allclose(A @ W_row, np.ones(3), atol=1e-10):
                        W[k*3+i, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify the full decomposition
            ok = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0
                        for c in range(rank):
                            s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert to lists of lists
                U_list = [[float(U[r, c]) for c in range(rank)] for r in range(9)]
                V_list = [[float(V[r, c]) for c in range(rank)] for r in range(9)]
                W_list = [[float(W[r, c]) for c in range(rank)] for r in range(9)]
                return U_list, V_list, W_list
    
    return None

def _try_rank_decomposition(n, m, p, rank, start_time, deadline_margin=30):
    """
    Try to find a rank-r decomposition for n x m * m x p matrix multiplication.
    Returns (U, V, W) or None.
    """
    # For the general case, we need to handle the full tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations with (n*m + m*p + p*n) * rank unknowns
    # For small n,m,p and rank, we can try randomized approaches
    
    # For 3x3x6, we can split into two 3x3x3 blocks
    if n == 3 and m == 3 and p == 6:
        # Try to find rank-23 decomposition for 3x3x3, then duplicate
        # First try rank 23 for 3x3x3
        result = _find_3x3_decomposition(23, max_attempts=200, seed=1)
        if result is not None:
            U3, V3, W3 = result
            rank_total = 46
            U_full = [[0.0] * rank_total for _ in range(9)]
            V_full = [[0.0] * rank_total for _ in range(18)]
            W_full = [[0.0] * rank_total for _ in range(18)]
            
            # First 23 terms: for B1 (columns 0-2)
            for c in range(23):
                for i in range(3):
                    for j in range(3):
                        U_full[i*3+j][c] = U3[i*3+j][c]
                for j in range(3):
                    for k1 in range(3):
                        V_full[j*3+k1][c] = V3[j*3+k1][c]
                for k1 in range(3):
                    for i in range(3):
                        W_full[k1*3+i][c] = W3[k1*3+i][c]
            
            # Next 23 terms: for B2 (columns 3-5)
            for c in range(23):
                c_full = 23 + c
                for i in range(3):
                    for j in range(3):
                        U_full[i*3+j][c_full] = U3[i*3+j][c]
                for j in range(3):
                    for k2 in range(3):
                        V_full[j*6 + 3 + k2][c_full] = V3[j*3+k2][c]
                for k2 in range(3):
                    for i in range(3):
                        W_full[(3+k2)*3 + i][c_full] = W3[k2*3+i][c]
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank_total):
                            s += U_full[i*m+j][c] * V_full[j*p+k][c] * W_full[k*n+i][c]
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return U_full, V_full, W_full
        
        # Try lower ranks for 3x3x3: 22, 21, 20, 19
        for r in [22, 21, 20, 19]:
            if time.time() > start_time + (150 - deadline_margin):
                break
            result = _find_3x3_decomposition(r, max_attempts=100, seed=r*100+1)
            if result is not None:
                U3, V3, W3 = result
                rank_total = 2 * r
                U_full = [[0.0] * rank_total for _ in range(9)]
                V_full = [[0.0] * rank_total for _ in range(18)]
                W_full = [[0.0] * rank_total for _ in range(18)]
                
                for c in range(r):
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][c] = U3[i*3+j][c]
                    for j in range(3):
                        for k1 in range(3):
                            V_full[j*3+k1][c] = V3[j*3+k1][c]
                    for k1 in range(3):
                        for i in range(3):
                            W_full[k1*3+i][c] = W3[k1*3+i][c]
                
                for c in range(r):
                    c_full = r + c
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][c_full] = U3[i*3+j][c]
                    for j in range(3):
                        for k2 in range(3):
                            V_full[j*6 + 3 + k2][c_full] = V3[j*3+k2][c]
                    for k2 in range(3):
                        for i in range(3):
                            W_full[(3+k2)*3 + i][c_full] = W3[k2*3+i][c]
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank_total):
                                s += U_full[i*m+j][c] * V_full[j*p+k][c] * W_full[k*n+i][c]
                            if abs(s - 1) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return U_full, V_full, W_full
        
        # Try direct rank search for 3x3x6
        for r in [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
            if time.time() > start_time + (150 - deadline_margin):
                break
            # This would require a different approach for the full 3x3x6 tensor
            # Skip for now, too complex
            pass
        
        return None
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Try to find low-rank decompositions for 3x3x3, then combine
    for 3x3x6. Fall back to schoolbook (rank 54) if no better found.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to find improved decomposition
    result = _try_rank_decomposition(n, m, p, 54, start_time, deadline_margin=30)
    
    if result is not None:
        U, V, W = result
        rank = len(U[0])
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
