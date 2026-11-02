"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
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

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    return U, V, W

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
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
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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

def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-14):
    """Run ALS iterations to refine the decomposition."""
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
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _rationalize(U, V, W, T, R, n, m, p):
    """Try to rationalize entries and verify."""
    # Try various denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: just verify as floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _find_rank9_for_2x2x3(n=2, m=2, p=3, timeout=30, max_trials=30):
    """
    Find a rank-9 decomposition for 2x2 by 2x3 matrix multiplication.
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}.
    Uses structured initialization + ALS + rationalization.
    """
    start_time = time.time()
    
    # Build the 2x2x3 tensor
    T = np.ones((n*m, m*p, p*n), dtype=np.float64)
    # Actually T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    R = 9
    
    for trial in range(max_trials):
        if time.time() - start_time > timeout:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Different initialization strategies
        if trial % 5 == 0:
            # Random init
            U[:, :] = np.random.randn(n*m, R) * 0.5
            V[:, :] = np.random.randn(m*p, R) * 0.5
            W[:, :] = np.random.randn(p*n, R) * 0.5
        elif trial % 5 == 1:
            # Structured: use partial Strassen-like patterns
            # First 7 terms: Strassen-like for cols 0,1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[6, 0] = 1
            
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[5, 1] = 1; W[6, 1] = -1
            
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[6, 2] = 1
            
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[5, 3] = 1
            
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[6, 5] = 1
            
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            
            # Last 2 terms: random for col 2
            U[:, 7:] = np.random.randn(n*m, 2) * 0.5
            V[:, 7:] = np.random.randn(m*p, 2) * 0.5
            W[:, 7:] = np.random.randn(p*n, 2) * 0.5
        elif trial % 5 == 2:
            # Syndrome-based: exploit the linear structure
            # The 2x2x3 tensor has specific linear dependencies
            # Use a basis that respects the tensor structure
            for r in range(R):
                for i in range(n*m):
                    U[i, r] = np.cos(2 * np.pi * i * r / (R + 1)) * 0.5 + 0.1 * np.random.randn()
                for i in range(m*p):
                    V[i, r] = np.cos(2 * np.pi * i * r / (R + 1)) * 0.5 + 0.1 * np.random.randn()
                for i in range(p*n):
                    W[i, r] = np.cos(2 * np.pi * i * r / (R + 1)) * 0.5 + 0.1 * np.random.randn()
        elif trial % 5 == 3:
            # Block-structured: 3 groups of 3 terms for each column
            for col in range(3):
                base = col * 3
                for sub in range(3):
                    r = base + sub
                    # Each group handles one column with rank-3 (minimum for 2x2x1 is 2, but 3 allows sharing)
                    # Actually for 2x2x1, minimum rank is 2. Let's use 2+2+2=6 for 3 columns... no, that's 6 not 9.
                    # Let's use a different structure: 3 terms per column but with cross-column sharing
                    U[0, r] = 1 if sub == 0 else (0.5 if sub == 1 else 0)
                    U[1, r] = 1 if sub == 1 else (0.5 if sub == 0 else 0)
                    U[2, r] = 1 if sub == 2 else (0.5 if sub == 1 else 0)
                    U[3, r] = 1 if sub == 0 else (0.5 if sub == 2 else 0)
                    
                    V[0*p+col, r] = 1 if sub < 2 else 0
                    V[1*p+col, r] = 1 if sub >= 1 else 0
                    
                    W[col*n+0, r] = 1 if sub == 0 else (0.5 if sub == 1 else 0)
                    W[col*n+1, r] = 1 if sub == 1 else (0.5 if sub == 2 else 0)
                    W[col*n+2, r] = 0 if col == 2 else (0.5 if sub == 2 else 0)
        else:
            # Perturbed structured
            U[:, :] = np.random.randn(n*m, R) * 0.3
            V[:, :] = np.random.randn(m*p, R) * 0.3
            W[:, :] = np.random.randn(p*n, R) * 0.3
            # Add some structure
            U[0, 0] += 1; U[3, 0] += 1
            V[0, 0] += 1; V[3, 0] += 1
            W[0, 0] += 1; W[6, 0] += 1
        
        # Run ALS
        max_iter = 400
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=max_iter)
        
        if err < 1e-8:
            result = _rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    
    return None

def _construct_rank16(n=2, m=2, p=5, timeout=60):
    """
    Construct a rank-16 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Rank-9 decomposition for B columns 2,3,4
    """
    start_time = time.time()
    
    # Find rank-9 for the 2x2x3 sub-problem
    rank9_result = _find_rank9_for_2x2x3(n=2, m=2, p=3, timeout=min(timeout - 5, 45), max_trials=25)
    
    if rank9_result is None:
        return None
    
    U9 = np.array(rank9_result["U"], dtype=np.float64)
    V9 = np.array(rank9_result["V"], dtype=np.float64)
    W9 = np.array(rank9_result["W"], dtype=np.float64)
    
    # Now combine: Strassen for cols 0,1 (7 terms) + rank-9 for cols 2,3,4 (9 terms)
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    # Rank-9 block for cols 2,3,4
    # U9 has shape (4, 9), V9 has shape (6, 9), W9 has shape (6, 9)
    # But we need to map the 2x2x3 indices to the full 2x2x5 indices
    # In the 2x2x3 sub-problem:
    #   U index: i*m + j where i in {0,1}, j in {0,1} -> same as full
    #   V index: j*p3 + k where p3=3, j in {0,1}, k in {0,1,2} -> maps to j*5 + (k+2) in full
    #   W index: k*p3 + i where p3=3, k in {0,1,2}, i in {0,1} -> maps to (k+2)*2 + i in full
    
    for r in range(9):
        # U: same indices (i*2+j)
        U[:, 7+r] = U9[:, r]
        
        # V: map from (j*3+k) to (j*5+(k+2))
        for j in range(2):
            for k in range(3):
                V[j*5 + (k+2), 7+r] = V9[j*3 + k, r]
        
        # W: map from (k*3+i) to ((k+2)*2+i)
        for k in range(3):
            for i in range(2):
                W[(k+2)*2 + i, 7+r] = W9[k*3 + i, r]
    
    # Verify
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    if _verify(U_list, V_list, W_list, n, m, p):
        # Try to rationalize
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        
        result = _rationalize(U, V, W, T, rank, n, m, p)
        if result is not None:
            return result
        
        # If rationalization failed, return floats
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _get_rank18_np(n=2, m=2, p=5):
    """Get rank-18 construction as numpy arrays."""
    U = np.zeros((n*m, 18))
    V = np.zeros((m*p, 18))
    W = np.zeros((p*n, 18))
    
    # Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    # Strassen block for cols 2,3
    U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
    U[:, 7:14] = U_str2
    V[:, 7:14] = V_str2
    W[:, 7:14] = W_str2
    
    # Schoolbook for col 4
    U[0, 14] = 1; V[0*p+4, 14] = 1; W[4*n+0, 14] = 1
    U[1, 15] = 1; V[1*p+4, 15] = 1; W[4*n+0, 15] = 1
    U[2, 16] = 1; V[0*p+4, 16] = 1; W[4*n+1, 16] = 1
    U[3, 17] = 1; V[1*p+4, 17] = 1; W[4*n+1, 17] = 1
    
    return U, V, W

def _search_rank_R(n=2, m=2, p=5, R=17, timeout=40, max_trials=20):
    """
    Search for rank-R decomposition using random initialization + ALS.
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    U18, V18, W18 = _get_rank18_np(n, m, p)
    
    for trial in range(max_trials):
        if time.time() - start_time > timeout - 3:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Different initialization strategies
        if trial % 4 == 0:
            # Pure random
            U[:, :] = np.random.randn(n*m, R) * 0.5
            V[:, :] = np.random.randn(m*p, R) * 0.5
            W[:, :] = np.random.randn(p*n, R) * 0.5
        elif trial % 4 == 1 and R >= 7:
            # Perturbed rank-18 structure
            cols_to_keep = list(range(R))
            if R < 18:
                # Remove random columns from rank-18
                removed = random.sample(range(18), 18 - R)
                cols_to_keep = [c for c in range(18) if c not in removed]
            U[:, :] = U18[:, cols_to_keep[:R]]
            V[:, :] = V18[:, cols_to_keep[:R]]
            W[:, :] = W18[:, cols_to_keep[:R]]
            noise_scale = 0.1 + 0.05 * (trial % 3)
            U[:, :] += np.random.randn(n*m, R) * noise_scale
            V[:, :] += np.random.randn(m*p, R) * noise_scale
            W[:, :] += np.random.randn(p*n, R) * noise_scale
        elif trial % 4 == 2:
            # Structured: Strassen for first 7, random for rest
            if R >= 7:
                U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
                U[:, :7] = U_str
                V[:, :7] = V_str
                W[:, :7] = W_str
                if R > 7:
                    U[:, 7:] = np.random.randn(n*m, R-7) * 0.5
                    V[:, 7:] = np.random.randn(m*p, R-7) * 0.5
                    W[:, 7:] = np.random.randn(p*n, R-7) * 0.5
            else:
                U[:, :] = np.random.randn(n*m, R) * 0.5
                V[:, :] = np.random.randn(m*p, R) * 0.5
                W[:, :] = np.random.randn(p*n, R) * 0.5
        else:
            # Fourier-like structured initialization
            for r in range(R):
                for i in range(n*m):
                    U[i, r] = np.cos(2 * np.pi * i * r / (R + n*m)) + 0.1 * np.random.randn()
                for i in range(m*p):
                    V[i, r] = np.cos(2 * np.pi * i * r / (R + m*p)) + 0.1 * np.random.randn()
                for i in range(p*n):
                    W[i, r] = np.cos(2 * np.pi * i * r / (R + p*n)) + 0.1 * np.random.randn()
        
        # Run ALS refinement
        max_iter = 300
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=max_iter)
        
        # Check and rationalize
        if err < 1e-8:
            result = _rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # First try rank 16 (Strassen + rank-9 for 2x2x3)
    if time.time() < deadline - 30:
        result = _construct_rank16(n, m, p, timeout=min(50, deadline - time.time() - 30))
        if result is not None:
            return result
    
    # Try rank 17
    if time.time() < deadline - 15:
        result = _search_rank_R(n, m, p, R=17, timeout=min(20, deadline - time.time() - 15), max_trials=15)
        if result is not None:
            return result
    
    # Try rank 18 (constructive, guaranteed)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
