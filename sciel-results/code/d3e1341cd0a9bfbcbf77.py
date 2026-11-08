"""Program for 3x4x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, safety margin

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0

def _als_decompose(n, m, p, rank, n_restarts=10, max_iter=500, tol=1e-9):
    """
    Alternating Least Squares for finding a rank-rank decomposition
    of the 3x4x6 matrix multiplication tensor.
    
    Returns (U, V, W) if found, else None.
    """
    for restart in range(n_restarts):
        if _time_left() < 10.0:
            return None
        
        # Random initialization with some structure
        np.random.seed(42 + restart * 137)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Ensure no all-zero columns
        for c in range(rank):
            if np.allclose(U[:, c], 0):
                U[np.random.randint(n * m), c] = 1.0
            if np.allclose(V[:, c], 0):
                V[np.random.randint(m * p), c] = 1.0
            if np.allclose(W[:, c], 0):
                W[np.random.randint(p * n), c] = 1.0
        
        prev_err = float('inf')
        
        for iteration in range(max_iter):
            if _time_left() < 5.0:
                return None
            
            # Update U: for each (i, j), solve least squares
            for i in range(n):
                for j in range(m):
                    # A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b_vec = np.ones(p)
                    try:
                        U[i * m + j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Update V: for each (j, k), solve least squares
            for j in range(m):
                for k in range(p):
                    # A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b_vec = np.ones(n)
                    try:
                        V[j * p + k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Update W: for each (k, i), solve least squares
            for k in range(p):
                for i in range(n):
                    # A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b_vec = np.ones(m)
                    try:
                        W[k * n + i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if abs(err - prev_err) < tol or err < tol:
                if err < 1e-6:
                    return (U.copy(), V.copy(), W.copy())
                break
            prev_err = err
        
        # If we exited the loop without finding a solution, check one more time
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (val - 1.0) ** 2
        if np.sqrt(err) < 1e-6:
            return (U.copy(), V.copy(), W.copy())
    
    return None

def _strassen_2x2_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns (U, V, W) for multiplying (2x2) by (2x2) -> (2x2).
    Rank 7 decomposition.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Indexing: A is 2x2 with A[i,j], B is 2x2 with B[j,k], C is 2x2 with C[i,k]
    # U index: i*m+j = i*2+j, V index: j*p+k = j*2+k, W index: k*n+i = k*2+i
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11  (M1 contributes to c00 and c11)
    # Wait, W index is k*n+i = k*2+i
    # c00: k=0,i=0 -> W[0,0]
    # c11: k=1,i=1 -> W[1*2+1, 0] = W[3,0]
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1               # b00
    W[2, 1] = 1               # c10: k=0,i=1 -> W[0*2+1,1]=W[1,1]? No, W[k*n+i] = W[k*2+i]
    # c10: i=1,k=0 -> W[0*2+1, 1] = W[1,1]
    W[1, 1] = 1
    # c11: i=1,k=1 -> W[1*2+1, 1] = W[3,1]
    W[3, 1] = 1
    
    # M3: a00(b10-b11)
    U[0, 2] = 1  # a00
    V[2, 2] = 1; V[3, 2] = -1  # b10, -b11
    W[1, 2] = 1  # c01: i=0,k=1 -> W[1*2+0,2]=W[2,2]? No, W[k*n+i]=W[k*2+i]
    # c01: i=0,k=1 -> W[1*2+0, 2] = W[2,2]
    W[2, 2] = 1
    # c11: i=1,k=1 -> W[1*2+1, 2] = W[3,2]
    W[3, 2] = 1
    
    # M4: a11(b01-b00)
    U[3, 3] = 1  # a11
    V[1, 3] = 1; V[0, 3] = -1  # b01, -b00
    W[0, 3] = 1  # c00: i=0,k=0 -> W[0,3]
    W[2, 3] = 1  # c10: i=1,k=0 -> W[0*2+1,3]=W[1,3]
    W[1, 3] = 1
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1               # b11
    W[2, 4] = -1  # c01: i=0,k=1 -> W[2,4]
    W[3, 4] = 1   # c11: i=1,k=1 -> W[3,4]
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1  # a10, -a00
    V[0, 5] = 1; V[1, 5] = 1   # b00, b01
    W[1, 5] = 1   # c10: i=1,k=0 -> W[1,5]
    W[3, 5] = 1   # c11: i=1,k=1 -> W[3,5]
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
    V[2, 6] = 1; V[3, 6] = 1   # b10, b11
    W[0, 6] = 1   # c00: i=0,k=0 -> W[0,6]
    W[3, 6] = 1   # c11: i=1,k=1 -> W[3,6]
    
    return (U, V, W)

def _construct_3x4x6_from_blocks():
    """
    Try to construct a 3x4x6 decomposition using block structure.
    Split into 2x2 blocks where possible.
    
    A is 3x4, B is 4x6, C is 3x6.
    We can think of this as multiplying a 3x4 by 4x6.
    
    One approach: use Strassen for the 2x2 blocks and handle the extra row/col.
    """
    n, m, p = 3, 4, 6
    
    # Start with schoolbook as baseline
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return (U, V, W)

def construct():
    n, m, p = 3, 4, 6
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_solution = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Try ranks from 24 upward
    for rank in range(24, rank_sb + 1):
        if _time_left() < 20.0:
            break
        
        # Try ALS with multiple restarts
        n_restarts = min(20, int(max(1, _time_left() / 5.0)))
        result = _als_decompose(n, m, p, rank, n_restarts=n_restarts, max_iter=300)
        
        if result is not None:
            U, V, W = result
            # Round to half-integers
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, n, m, p):
                best_solution = (U_r, V_r, W_r)
                best_rank = rank
                break
            elif _verify(U, V, W, n, m, p):
                # Use unrounded if rounded fails
                best_solution = (U, V, W)
                best_rank = rank
                break
    
    U_best, V_best, W_best = best_solution
    
    # Convert to lists
    U_list = [[float(U_best[i, j]) for j in range(U_best.shape[1])] for i in range(U_best.shape[0])]
    V_list = [[float(V_best[i, j]) for j in range(V_best.shape[1])] for i in range(V_best.shape[0])]
    W_list = [[float(W_best[i, j]) for j in range(W_best.shape[1])] for i in range(W_best.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
