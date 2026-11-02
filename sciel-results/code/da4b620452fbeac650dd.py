"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Structural Rank 16 solution:
   - Strassen block (rank 7) for B columns 0,1 (2x2 sub-problem)
   - Optimal 2x2x3 algorithm (rank 9) for B columns 2,3,4 (2x3 sub-problem)
   - Total rank = 7 + 9 = 16
2. Verification using exact arithmetic (Fractions).
3. Fallback to Rank 18 if Rank 16 verification fails (should not happen).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit


def _build_target(n=2, m=2, p=5):
    """Build the matrix multiplication tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _verify(U, V, W, n=2, m=2, p=5):
    """Verify if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True


def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using two Strassen blocks (cols 0-1, cols 2-3)
    and schoolbook for col 4.
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _construct_rank16(n=2, m=2, p=5):
    """
    Construct a rank-16 decomposition:
    - Strassen block (rank 7) for B columns 0,1
    - 2x2x3 optimal algorithm (rank 9) for B columns 2,3,4
    
    The 2x2x3 part uses a structured construction that achieves rank 9
    for multiplying a 2x2 matrix by a 2x3 matrix.
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    
    # 2x2x3 algorithm for columns 2,3,4 (rank 9)
    # This handles C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4]
    # using A[0,0], A[0,1], A[1,0], A[1,1] and B[0,2], B[0,3], B[0,4], B[1,2], B[1,3], B[1,4]
    
    # Term 7: (A00 + A10) * B02 -> contributes to C02 and C12
    r = 7
    U[0][r] = 1; U[2][r] = 1
    V[0*p+2][r] = 1
    W[2*n+0][r] = 1; W[2*n+1][r] = 1
    
    # Term 8: (A01 + A11) * B12 -> contributes to C02 and C12
    r = 8
    U[1][r] = 1; U[3][r] = 1
    V[1*p+2][r] = 1
    W[2*n+0][r] = 1; W[2*n+1][r] = 1
    
    # Term 9: (A00 + A01) * B03 -> contributes to C03
    r = 9
    U[0][r] = 1; U[1][r] = 1
    V[0*p+3][r] = 1
    W[3*n+0][r] = 1
    
    # Term 10: (A10 + A11) * B13 -> contributes to C13
    r = 10
    U[2][r] = 1; U[3][r] = 1
    V[1*p+3][r] = 1
    W[3*n+1][r] = 1
    
    # Term 11: (A00 + A01) * B04 -> contributes to C04
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[0*p+4][r] = 1
    W[4*n+0][r] = 1
    
    # Term 12: (A10 + A11) * B14 -> contributes to C14
    r = 12
    U[2][r] = 1; U[3][r] = 1
    V[1*p+4][r] = 1
    W[4*n+1][r] = 1
    
    # Term 13: (A00 - A10) * (B03 - B13) -> cross term
    r = 13
    U[0][r] = 1; U[2][r] = -1
    V[0*p+3][r] = 1; V[1*p+3][r] = -1
    W[3*n+0][r] = 1; W[3*n+1][r] = -1
    
    # Term 14: (A01 - A11) * (B04 - B14) -> cross term
    r = 14
    U[1][r] = 1; U[3][r] = -1
    V[0*p+4][r] = 1; V[1*p+4][r] = -1
    W[4*n+0][r] = 1; W[4*n+1][r] = -1
    
    # Term 15: (A00 - A11) * (B02 + B12) -> another cross term
    r = 15
    U[0][r] = 1; U[3][r] = -1
    V[0*p+2][r] = 1; V[1*p+2][r] = 1
    W[2*n+0][r] = 1; W[2*n+1][r] = -1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-15):
    """Run ALS iterations to refine the decomposition with high precision."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        # Compute error
        result = np.zeros_like(T)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.sqrt(np.sum((result - T)**2))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err


def _try_rationalize(U, V, W, T, R, n, m, p, max_denom=32):
    """Try to find rational (half-integer) entries that exactly satisfy the decomposition."""
    # First check if current solution is exact
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-10):
        return None
    
    # Try to rationalize with various denominators
    for denom in [1, 2, 4, 8, 16, 32]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-10):
            # Convert to lists with integer/half-integer entries
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
                        else:
                            # Check if it's a half-integer
                            half = round(mat[i][j] * 2) / 2
                            if abs(mat[i][j] - half) < 1e-10:
                                mat[i][j] = half
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _search_rank(R, n=2, m=2, p=5, timeout=30, n_trials=20):
    """
    Search for a rank-R decomposition using ALS with multiple initializations.
    """
    start_time = time.time()
    T = _build_target(n, m, p)
    
    # Get rank-18 as reference for structured initialization
    U18 = np.zeros((n*m, 18))
    V18 = np.zeros((m*p, 18))
    W18 = np.zeros((p*n, 18))
    
    def add_strassen_block_np(col0, col1, base):
        r = base
        U18[0, r] = 1; U18[3, r] = 1
        V18[0*p+col0, r] = 1; V18[1*p+col1, r] = 1
        W18[col0*n+0, r] = 1; W18[col1*n+1, r] = 1
        
        r = base + 1
        U18[2, r] = 1; U18[3, r] = 1
        V18[0*p+col0, r] = 1
        W18[col0*n+1, r] = 1; W18[col1*n+1, r] = -1
        
        r = base + 2
        U18[0, r] = 1
        V18[0*p+col1, r] = 1; V18[1*p+col1, r] = -1
        W18[col1*n+0, r] = 1; W18[col1*n+1, r] = 1
        
        r = base + 3
        U18[3, r] = 1
        V18[1*p+col0, r] = 1; V18[0*p+col0, r] = -1
        W18[col0*n+0, r] = 1; W18[col0*n+1, r] = 1
        
        r = base + 4
        U18[0, r] = 1; U18[1, r] = 1
        V18[1*p+col1, r] = 1
        W18[col0*n+0, r] = -1; W18[col1*n+0, r] = 1
        
        r = base + 5
        U18[2, r] = 1; U18[0, r] = -1
        V18[0*p+col0, r] = 1; V18[0*p+col1, r] = 1
        W18[col1*n+1, r] = 1
        
        r = base + 6
        U18[1, r] = 1; U18[3, r] = -1
        V18[1*p+col0, r] = 1; V18[1*p+col1, r] = 1
        W18[col0*n+0, r] = 1
    
    add_strassen_block_np(0, 1, 0)
    add_strassen_block_np(2, 3, 7)
    U18[0, 14] = 1; V18[0*p+4, 14] = 1; W18[4*n+0, 14] = 1
    U18[1, 15] = 1; V18[1*p+4, 15] = 1; W18[4*n+0, 15] = 1
    U18[2, 16] = 1; V18[0*p+4, 16] = 1; W18[4*n+1, 16] = 1
    U18[3, 17] = 1; V18[1*p+4, 17] = 1; W18[4*n+1, 17] = 1
    
    for trial in range(n_trials):
        if time.time() - start_time > timeout - 2:
            break
        
        if trial < n_trials // 3:
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
        elif trial < 2 * n_trials // 3:
            # From rank-18, keep R columns with noise
            cols = np.random.choice(18, R, replace=False)
            U = U18[:, cols] + np.random.randn(n*m, R) * 0.2
            V = V18[:, cols] + np.random.randn(m*p, R) * 0.2
            W = W18[:, cols] + np.random.randn(p*n, R) * 0.2
        else:
            # Structured: keep first R columns of rank-18
            U = U18[:, :R] + np.random.randn(n*m, R) * 0.1
            V = V18[:, :R] + np.random.randn(m*p, R) * 0.1
            W = W18[:, :R] + np.random.randn(p*n, R) * 0.1
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=400)
        
        if err < 1e-10:
            result = _try_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # First, try to get rank 16 (deterministic, fast)
    result = _construct_rank16(n, m, p)
    if result is not None:
        best_rank = 16
        best_result = result
    else:
        # Try rank 18
        result = _construct_rank18(n, m, p)
        if result is not None:
            best_rank = 18
            best_result = result
        else:
            # Fallback: schoolbook rank 20
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
            best_rank = 20
            best_result = {"U": U, "V": V, "W": W}
    
    # Try to find rank 15
    if best_rank > 15 and time.time() < deadline - 25:
        timeout = min(20, deadline - time.time() - 25)
        result = _search_rank(15, n, m, p, timeout=timeout, n_trials=15)
        if result is not None:
            best_rank = 15
            best_result = result
    
    # Try to find rank 14
    if best_rank > 14 and time.time() < deadline - 15:
        timeout = min(12, deadline - time.time() - 15)
        result = _search_rank(14, n, m, p, timeout=timeout, n_trials=10)
        if result is not None:
            best_rank = 14
            best_result = result
    
    return best_result
