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

def _strassen_2x2_decomposition():
    """
    Returns rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Tensors: U (2*2 x 7), V (2*2 x 7), W (2*2 x 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # Result:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a11 (i=0,j=0)
    U[0][0] = 1; U[0][1] = 0; U[0][2] = 1; U[0][3] = 0; U[0][4] = 0; U[0][5] = -1; U[0][6] = 0
    # a12 (i=0,j=1)
    U[1][0] = 0; U[1][1] = 0; U[1][2] = 0; U[1][3] = 0; U[1][4] = 1; U[1][5] = 0; U[1][6] = 1
    # a21 (i=1,j=0)
    U[2][0] = 0; U[2][1] = 1; U[2][2] = 0; U[2][3] = 0; U[2][4] = 0; U[2][5] = 1; U[2][6] = 0
    # a22 (i=1,j=1)
    U[3][0] = 1; U[3][1] = 1; U[3][2] = 0; U[3][3] = 1; U[3][4] = 0; U[3][5] = 0; U[3][6] = -1
    
    # b11 (j=0,k=0)
    V[0][0] = 1; V[0][1] = 1; V[0][2] = 0; V[0][3] = -1; V[0][4] = 0; V[0][5] = 1; V[0][6] = 0
    # b12 (j=0,k=1)
    V[1][0] = 0; V[1][1] = 0; V[1][2] = 1; V[1][3] = 0; V[1][4] = 0; V[1][5] = 1; V[1][6] = 0
    # b21 (j=1,k=0)
    V[2][0] = 0; V[2][1] = 0; V[2][2] = 0; V[2][3] = 0; V[2][4] = 0; V[2][5] = 0; V[2][6] = 1
    # b22 (j=1,k=1)
    V[3][0] = 1; V[3][1] = 0; V[3][2] = -1; V[3][3] = 0; V[3][4] = 1; V[3][5] = 0; V[3][6] = 1
    
    # c11 (k=0,i=0)
    W[0][0] = 1; W[0][1] = 0; W[0][2] = 0; W[0][3] = 1; W[0][4] = -1; W[0][5] = 0; W[0][6] = 1
    # c12 (k=1,i=0)
    W[1][0] = 0; W[1][1] = 0; W[1][2] = 1; W[1][3] = 0; W[1][4] = 1; W[1][5] = 0; W[1][6] = 0
    # c21 (k=0,i=1)
    W[2][0] = 0; W[2][1] = 1; W[2][2] = 0; W[2][3] = 1; W[2][4] = 0; W[2][5] = 0; W[2][6] = 0
    # c22 (k=1,i=1)
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][3] = 0; W[3][4] = 0; W[3][5] = 1; W[3][6] = 0
    
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=1000, tol=1e-8, seed=0):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
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
                # Round to nearest half-integer if close
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def _construct_3x4x6_strassen():
    """
    Construct a rank-21 decomposition for 3x4x6 using Strassen on 2x2 blocks.
    
    Idea: Split the 4x6 matrix into two 2x3 blocks vertically.
    Use Strassen-like decomposition for each 2x3, then combine.
    
    Actually, let's use a more direct approach:
    Split 4 into 2+2, and use Strassen for 2x2x2 blocks.
    
    For 3x4x6:
    - Split middle dimension 4 into two 2s
    - Split output dimension 6 into three 2s
    - Use Strassen (rank 7) for each 2x2x2 block
    - Total rank: 3 * 7 = 21? No, that's not quite right.
    
    Better approach: Use the fact that 4x6 can be computed with fewer than 24 products.
    Actually, let's just try to find a good decomposition via optimization.
    """
    n, m, p = 3, 4, 6
    
    # Try rank 24 first (which is 3 * 8, using 8-product schemes)
    # Or try rank 21 (3 * 7, using Strassen)
    
    best_result = None
    best_rank = n * m * p
    
    # Try various ranks
    for target_rank in [21, 24, 27, 30, 33, 36]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        for seed in range(10):
            if time.time() > _DEADLINE - 5.0:
                break
            
            result = _try_rank(n, m, p, target_rank, max_iter=800, tol=1e-9, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                    break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p
    
    # Strategy 1: Try to find a low-rank decomposition using optimization
    result, rank_found = _construct_3x4x6_strassen()
    if result is not None:
        best_result = result
        best_rank = rank_found
    
    # Strategy 2: If we didn't find a good one, try direct search
    if best_result is None or best_rank > 40:
        if time.time() < _DEADLINE - 10.0:
            for rank in range(30, 60):
                if time.time() > _DEADLINE - 5.0:
                    break
                for seed in range(3):
                    if time.time() > _DEADLINE - 5.0:
                        break
                    result = _try_rank(n, m, p, rank, max_iter=600, seed=seed)
                    if result is not None:
                        U, V, W = result
                        if _verify(U, V, W, n, m, p):
                            if rank < best_rank:
                                best_rank = rank
                                best_result = {"U": U, "V": V, "W": W}
                            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
