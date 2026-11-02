import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    """
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, -1]
    ], dtype=float)
    
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _build_block(n, m, p):
    """
    Builds decomposition for n x m x p using best known block strategies.
    """
    if n == 1 and m == 1 and p == 1:
        return _schoolbook_block(1, 1, 1)
    if n == 1 and m == 1 and p == 2:
        return _schoolbook_block(1, 1, 2)
    if n == 1 and m == 2 and p == 2:
        return _schoolbook_block(1, 2, 2)
    if n == 2 and m == 1 and p == 2:
        return _schoolbook_block(2, 1, 2)
    if n == 2 and m == 2 and p == 1:
        return _schoolbook_block(2, 2, 1)
    if n == 1 and m == 2 and p == 1:
        return _schoolbook_block(1, 2, 1)
    if n == 1 and m == 1 and p == 3:
        return _schoolbook_block(1, 1, 3)
    if n == 1 and m == 3 and p == 1:
        return _schoolbook_block(1, 3, 1)
    if n == 1 and m == 3 and p == 2:
        return _schoolbook_block(1, 3, 2)
    if n == 1 and m == 2 and p == 3:
        return _schoolbook_block(1, 2, 3)
    if n == 2 and m == 1 and p == 3:
        return _schoolbook_block(2, 1, 3)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Recursive or specific splits
    if n == 2 and m == 3 and p == 2:
        # Split m into 2 and 1
        U1, V1, W1, r1 = _build_block(2, 2, 2)
        U2, V2, W2, r2 = _build_block(2, 1, 2)
        total_rank = r1 + r2
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Embed 2x2x2 (j=0,1)
        for i in range(n):
            for j in range(2):
                for c in range(r1):
                    U[i * m + j, c] = U1[i * 2 + j, c]
        for j in range(2):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k, c] = V1[j * p + k, c]
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i, c] = W1[k * n + i, c]
                    
        # Embed 2x1x2 (j=2)
        offset = r1
        for i in range(n):
            for c in range(r2):
                U[i * m + 2, offset + c] = U2[i * 1 + 0, c]
        for k in range(p):
            for c in range(r2):
                V[2 * p + k, offset + c] = V2[0 * p + k, c]
        for k in range(p):
            for i in range(n):
                for c in range(r2):
                    W[k * n + i, offset + c] = W2[k * n + i, c]
                    
        return U, V, W, total_rank

    if n == 2 and m == 2 and p == 3:
        # Split p into 2 and 1
        U1, V1, W1, r1 = _build_block(2, 2, 2)
        U2, V2, W2, r2 = _build_block(2, 2, 1)
        total_rank = r1 + r2
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Embed 2x2x2 (k=0,1)
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j, c] = U1[i * m + j, c]
        for j in range(m):
            for k in range(2):
                for c in range(r1):
                    V[j * p + k, c] = V1[j * 2 + k, c]
        for k in range(2):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i, c] = W1[k * n + i, c]
                    
        # Embed 2x2x1 (k=2)
        offset = r1
        for i in range(n):
            for j in range(m):
                for c in range(r2):
                    U[i * m + j, offset + c] = U2[i * m + j, c]
        for j in range(m):
            for c in range(r2):
                V[j * p + 2, offset + c] = V2[j * 1 + 0, c]
        for i in range(n):
            for c in range(r2):
                W[2 * n + i, offset + c] = W2[0 * n + i, c]
                
        return U, V, W, total_rank

    if n == 3 and m == 2 and p == 2:
        # Split n into 2 and 1
        U1, V1, W1, r1 = _build_block(2, 2, 2)
        U2, V2, W2, r2 = _build_block(1, 2, 2)
        total_rank = r1 + r2
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Embed 2x2x2 (i=0,1)
        for i in range(2):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j, c] = U1[i * m + j, c]
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k, c] = V1[j * p + k, c]
        for k in range(p):
            for i in range(2):
                for c in range(r1):
                    W[k * n + i, c] = W1[k * 2 + i, c]
                    
        # Embed 1x2x2 (i=2)
        offset = r1
        for j in range(m):
            for c in range(r2):
                U[2 * m + j, offset + c] = U2[0 * m + j, c]
        for j in range(m):
            for k in range(p):
                for c in range(r2):
                    V[j * p + k, offset + c] = V2[j * p + k, c]
        for k in range(p):
            for c in range(r2):
                W[k * n + 2, offset + c] = W2[k * 1 + 0, c]
                
        return U, V, W, total_rank

    if n == 3 and m == 3 and p == 2:
        # Split n into 2 and 1
        U1, V1, W1, r1 = _build_block(2, 3, 2)
        U2, V2, W2, r2 = _build_block(1, 3, 2)
        total_rank = r1 + r2
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Embed 2x3x2 (i=0,1)
        for i in range(2):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j, c] = U1[i * 3 + j, c]
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k, c] = V1[j * p + k, c]
        for k in range(p):
            for i in range(2):
                for c in range(r1):
                    W[k * n + i, c] = W1[k * 2 + i, c]
                    
        # Embed 1x3x2 (i=2)
        offset = r1
        for j in range(m):
            for c in range(r2):
                U[2 * m + j, offset + c] = U2[0 * m + j, c]
        for j in range(m):
            for k in range(p):
                for c in range(r2):
                    V[j * p + k, offset + c] = V2[j * p + k, c]
        for k in range(p):
            for c in range(r2):
                W[k * n + 2, offset + c] = W2[k * 1 + 0, c]
                
        return U, V, W, total_rank

    # Fallback schoolbook
    return _schoolbook_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0

    try:
        # Partition n=3 into 1, 2
        # Partition m=5 into 2, 3
        # Partition p=6 into 2, 2, 2
        
        # Blocks:
        # I=1, J=2, K=2 (3 blocks)
        # I=1, J=3, K=2 (3 blocks)
        # I=2, J=2, K=2 (3 blocks)
        # I=2, J=3, K=2 (3 blocks)
        
        U_full = np.zeros((n * m, 0), dtype=float)
        V_full = np.zeros((m * p, 0), dtype=float)
        W_full = np.zeros((p * n, 0), dtype=float)
        total_rank = 0
        
        # Helper to append block
        def append_block(n_b, m_b, p_b, offset_i, offset_j, offset_k):
            nonlocal U_full, V_full, W_full, total_rank
            U_b, V_b, W_b, r_b = _build_block(n_b, m_b, p_b)
            new_rank = total_rank + r_b
            U_new = np.zeros((n * m, new_rank), dtype=float)
            V_new = np.zeros((m * p, new_rank), dtype=float)
            W_new = np.zeros((p * n, new_rank), dtype=float)
            
            if total_rank > 0:
                U_new[:, :total_rank] = U_full
                V_new[:, :total_rank] = V_full
                W_new[:, :total_rank] = W_full
                
            # Embed U_b
            for i in range(n_b):
                for j in range(m_b):
                    for c in range(r_b):
                        U_new[(offset_i + i) * m + (offset_j + j), total_rank + c] = U_b[i * m_b + j, c]
                        
            # Embed V_b
            for j in range(m_b):
                for k in range(p_b):
                    for c in range(r_b):
                        V_new[(offset_j + j) * p + (offset_k + k), total_rank + c] = V_b[j * p_b + k, c]
                        
            # Embed W_b
            for k in range(p_b):
                for i in range(n_b):
                    for c in range(r_b):
                        W_new[(offset_k + k) * n + (offset_i + i), total_rank + c] = W_b[k * n_b + i, c]
                        
            U_full = U_new
            V_full = V_new
            W_full = W_new
            total_rank = new_rank

        # I=1 (size 1), offset_i=0
        # J=2 (size 2), offset_j=0
        for k_idx in range(3):
            append_block(1, 2, 2, 0, 0, k_idx * 2)
            
        # I=1 (size 1), offset_i=0
        # J=3 (size 3), offset_j=2
        for k_idx in range(3):
            append_block(1, 3, 2, 0, 2, k_idx * 2)
            
        # I=2 (size 2), offset_i=1
        # J=2 (size 2), offset_j=0
        for k_idx in range(3):
            append_block(2, 2, 2, 1, 0, k_idx * 2)
            
        # I=2 (size 2), offset_i=1
        # J=3 (size 3), offset_j=2
        for k_idx in range(3):
            append_block(2, 3, 2, 1, 2, k_idx * 2)

        if time.time() < deadline:
            if _check_exact(U_full, V_full, W_full, n, m, p, tol=1e-6):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    except Exception as e:
        pass
    
    # Fallback schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
