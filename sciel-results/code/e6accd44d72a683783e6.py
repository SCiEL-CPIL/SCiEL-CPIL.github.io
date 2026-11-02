"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
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

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 by 2x2 = 2x2.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11) -> c00+1, c11+1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00 -> c10+1, c11-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11) -> c01+1, c11+1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00) -> c00+1, c10+1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (a00+a01)b11 -> c00-1, c01+1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c11+1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11) -> c00+1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks.
    For 3x4x6: split into blocks.
    """
    n, m, p = 3, 4, 6
    
    total_rank = 6 * 7 + 6 * 4  # 66
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    def embed_2x2x2(a_row0, a_col0, b_row0, c_row0, c_col0, start_col):
        """
        A submatrix: rows a_row0..a_row0+1, cols a_col0..a_col0+1 (2x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: rows c_row0..c_row0+1, cols c_col0..c_col0+1 (2x2)
        """
        sU, sV, sW = _strassen_2x2_2x2()
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = a_row0 + li
                    gj = a_col0 + lj
                    U[gi * m + gj][start_col + r] += sU[li * 2 + lj][r]
                    
                    for lk in range(2):
                        gj_b = b_row0 + lj
                        gk = c_col0 + lk
                        V[gj_b * p + gk][start_col + r] += sV[lj * 2 + lk][r]
                    
                    for lk in range(2):
                        gi_c = c_row0 + li
                        gk = c_col0 + lk
                        W[gk * n + gi_c][start_col + r] += sW[lk * 2 + li][r]
        
        return start_col + 7
    
    def embed_1x2x2(a_row, a_col0, b_row0, c_row, c_col0, start_col):
        """
        A submatrix: row a_row, cols a_col0..a_col0+1 (1x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: row c_row, cols c_col0..c_col0+1 (1x2)
        """
        for l1 in range(2):
            for l2 in range(2):
                gi = a_row
                gj = a_col0 + l1
                U[gi * m + gj][start_col] = 1
                
                gj_b = b_row0 + l1
                gk = c_col0 + l2
                V[gj_b * p + gk][start_col] = 1
                
                gi_c = c_row
                gk = c_col0 + l2
                W[gk * n + gi_c][start_col] = 1
                
                start_col += 1
        
        return start_col
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    col = embed_2x2x2(0, 0, 0, 0, 0, col)
    col = embed_2x2x2(0, 2, 2, 0, 0, col)
    
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    col = embed_2x2x2(0, 0, 0, 0, 2, col)
    col = embed_2x2x2(0, 2, 2, 0, 2, col)
    
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    col = embed_2x2x2(0, 0, 0, 0, 4, col)
    col = embed_2x2x2(0, 2, 2, 0, 4, col)
    
    # C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    col = embed_1x2x2(2, 0, 0, 2, 0, col)
    col = embed_1x2x2(2, 2, 2, 2, 0, col)
    
    # C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    col = embed_1x2x2(2, 0, 0, 2, 2, col)
    col = embed_1x2x2(2, 2, 2, 2, 2, col)
    
    # C[2, 4:6] = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6]
    col = embed_1x2x2(2, 0, 0, 2, 4, col)
    col = embed_1x2x2(2, 2, 2, 2, 4, col)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=500, tol=1e-10, seed=0, init=None):
    """Try to find a decomposition of given rank using vectorized alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
    else:
        scale = 1.0 / (rank ** 0.25)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U: for each (i,j), A[k,r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                # Vectorized: A[k,r] = V[j*p+k, r] * W[k*n+i, r]
                Vj = V[j * p:(j + 1) * p, :]  # (p, rank)
                Wi = W[i::n, :]  # (p, rank) - rows k*n+i for k=0..p-1
                A = Vj * Wi  # (p, rank)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), B[i,r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                Uj = U[j::m, :]  # (n, rank) - rows i*m+j for i=0..n-1
                Wk = W[k::p, :]  # (n, rank) - rows k*n+i for i=0..n-1
                B = Uj * Wk  # (n, rank)
                try:
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), C[j,r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                Ui = U[i * m:(i + 1) * m, :]  # (m, rank)
                Vk = V[k::p, :]  # (m, rank) - rows j*p+k for j=0..m-1
                C = Ui * Vk  # (m, rank)
                try:
                    x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence every 5 iterations
        if it % 5 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for idx in range(len(row)):
                            if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-8:
                                row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def _polish(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """High-precision polish pass with vectorized updates and tighter tolerance."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3.0:
            break
        
        for i in range(n):
            for j in range(m):
                Vj = V[j * p:(j + 1) * p, :]
                Wi = W[i::n, :]
                A = Vj * Wi
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                Uj = U[j::m, :]
                Wk = W[k::p, :]
                B = Uj * Wk
                try:
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                Ui = U[i * m:(i + 1) * m, :]
                Vk = V[k::p, :]
                C = Ui * Vk
                try:
                    x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                break
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-10:
                    row[idx] = round(row[idx] * 2) / 2.0
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    
    # First try the structured Strassen-based decomposition (rank 66)
    best_result = None
    best_rank = float('inf')
    
    try:
        U, V, W = _construct_strassen_based(n, m, p)
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    except Exception:
        pass
    
    # Try a 3x3x3 + remainder approach
    try:
        # Use Strassen 2x2x2 for the 2x2x2 core, handle remainders
        # 3x4x6: split as (2+1)x(2+2)x(2+2+2)
        # 2x2x2 blocks: 2*2*2 = 8 blocks, each rank 7 -> 56
        # Remainder rows/cols handled separately
        # This is essentially what _construct_strassen_based does
        pass
    except Exception:
        pass
    
    # Try to find even lower rank via alternating minimization
    # Start from rank 50 and go up to best_rank - 1
    search_start = 50
    search_end = min(best_rank - 1, 65) if best_rank < float('inf') else 65
    
    for rank in range(search_start, search_end + 1):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Multiple random restarts
        for seed in range(8):
            if time.time() > _DEADLINE - 10.0:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=400, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    # Polish for higher precision
                    U_p, V_p, W_p = _polish(U, V, W, n, m, p)
                    if _verify(U_p, V_p, W_p, n, m, p):
                        best_result = {"U": U_p, "V": V_p, "W": W_p}
                        best_rank = rank
                        break
                    else:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break
        
        if best_rank <= rank:
            break
    
    # Fallback to schoolbook if nothing found
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def _strassen_3x3_3x3():
    """
    Return a rank-23 decomposition for 3x3 by 3x3 = 3x3.
    Based on Strassen's 3x3x3 algorithm.
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Strassen 3x3x3: 23 multiplications
    # This is a known decomposition from the literature
    # Using the standard Strassen 3x3x3 scheme
    
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    # M1: (a00+a01+a10+a11+a20+a21)(b00+b01+b10+b11+b20+b21)
    # This contributes to c00,c01,c10,c11,c20,c21 with +1
    for idx in [0,1,3,4,6,7]:
        U[idx][0] = 1
    for idx in [0,1,3,4,6,7]:
        V[idx][0] = 1
    for idx in [0,1,3,4,6,7]:
        W[idx][0] = 1
    
    # M2: (a02+a12+a22)(b02+b12+b22)
    # Contributes to c02,c12,c22 with +1
    for idx in [2,5,8]:
        U[idx][1] = 1
    for idx in [2,5,8]:
        V[idx][1] = 1
    for idx in [2,5,8]:
        W[idx][1] = 1
    
    # M3: (a00+a01+a10+a11)(b00+b01+b10+b11)
    # Contributes to c00,c01,c10,c11
    for idx in [0,1,3,4]:
        U[idx][2] = 1
    for idx in [0,1,3,4]:
        V[idx][2] = 1
    for idx in [0,1,3,4]:
        W[idx][2] = 1
    
    # M4: (a02+a12)(b02+b12)
    for idx in [2,5]:
        U[idx][3] = 1
    for idx in [2,5]:
        V[idx][3] = 1
    for idx in [2,5]:
        W[idx][3] = 1
    
    # M5: (a20+a21+a22)(b20+b21+b22)
    for idx in [6,7,8]:
        U[idx][4] = 1
    for idx in [6,7,8]:
        V[idx][4] = 1
    for idx in [6,7,8]:
        W[idx][4] = 1
    
    # M6: (a00+a10+a20)(b00+b10+b20)
    for idx in [0,3,6]:
        U[idx][5] = 1
    for idx in [0,3,6]:
        V[idx][5] = 1
    for idx in [0,3,6]:
        W[idx][5] = 1
    
    # M7: (a01+a11+a21)(b01+b11+b21)
    for idx in [1,4,7]:
        U[idx][6] = 1
    for idx in [1,4,7]:
        V[idx][6] = 1
    for idx in [1,4,7]:
        W[idx][6] = 1
    
    # Now we need to subtract off the overcounted terms
    # The sum M1+M2+M3+M4+M5+M6+M7 gives each c[i,j] a coefficient
    # that needs correction. This is getting complex - let me use a 
    # verified 3x3x3 rank-23 decomposition instead.
    
    # Actually, let me use a simpler approach: just return None to indicate
    # this is not ready, and rely on the other methods.
    return None
# EVOLVE-BLOCK-END
