import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

def _validate_numpy(U, V, W, n, m, p):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _validate(U, V, W, n, m, p):
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    return _validate_numpy(U, V, W, n, m, p)

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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)"""
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    U[0][0] = 1; U[1][0] = 1  # M1: A11+A22
    U[1][1] = 1; U[1][0] = 1  # M2: A21+A22
    U[0][2] = 1                # M3: A11
    U[1][3] = 1                # M4: A22
    U[0][4] = 1; U[0][1] = 1   # M5: A11+A12
    U[1][5] = 1; U[0][5] = -1  # M6: A21-A11
    U[0][6] = 1; U[1][6] = -1  # M7: A12-A22
    
    V[0][0] = 1; V[1][0] = 1  # M1: B11+B22
    V[0][1] = 1                # M2: B11
    V[0][2] = 1; V[1][2] = -1  # M3: B12-B22
    V[1][3] = 1; V[0][3] = -1  # M4: B21-B11
    V[1][4] = 1                # M5: B22
    V[0][5] = 1; V[0][1] = 1   # M6: B11+B12
    V[1][6] = 1; V[1][1] = 1   # M7: B21+B22
    
    W[0][0] = 1; W[0][4] = -1; W[0][6] = 1  # C11 = M1 - M5 + M7
    W[1][0] = 1; W[1][4] = 1                # C12 = M3 + M5
    W[0][1] = 1; W[0][3] = 1                # C21 = M2 + M4
    W[1][1] = -1; W[1][3] = 1               # C22 = M1 - M2 + M4
    
    return U, V, W

def _build_332_decomposition(rank):
    """
    Build a 3x3x2 decomposition with given rank using structured random search.
    Returns (U, V, W) as lists or None.
    """
    n, m, p = 3, 3, 2
    num_eqs = n * m * p  # 18
    
    for attempt in range(200):
        # Structured random: use {-1, 0, 1} with bias toward sparsity
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for r in range(rank):
            # Each rank-1 term should have some structure
            # Randomly select which (i,j) and (j,k) positions are active
            active_U = set()
            active_V = set()
            for _ in range(random.randint(2, 5)):
                active_U.add(random.randint(0, n*m-1))
            for _ in range(random.randint(2, 5)):
                active_V.add(random.randint(0, m*p-1))
            
            for idx in active_U:
                U[idx, r] = random.choice([-1, 1])
            for idx in active_V:
                V[idx, r] = random.choice([-1, 1])
        
        # Build A matrix for W
        A = np.zeros((num_eqs, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        col_idx = k * n * rank + i * rank + r
                        A[eq_idx, col_idx] = coeff
        
        b = np.ones(num_eqs)
        
        try:
            W_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if solution is exact
        if np.allclose(A @ W_sol, b, atol=1e-10):
            # Try rounding
            for mult in [1, 2]:
                W_rounded = np.round(W_sol * mult) / mult
                if np.allclose(A @ W_rounded, b, atol=1e-9):
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_rounded.reshape(p * n, rank).tolist()
                    if _validate(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
    
    return None

def _construct_from_332(U_332, V_332, W_332):
    n, m, p = 3, 3, 8
    rank_332 = len(U_332[0])
    total_rank = 4 * rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_332):
            for block in range(4):
                U[i][block * rank_332 + r] = U_332[i][r]
                
    for j in range(m):
        for k in range(p):
            block = k // 2
            k_local = k % 2
            for r in range(rank_332):
                V[j * p + k][block * rank_332 + r] = V_332[j * 2 + k_local][r]
                
    for k in range(p):
        block = k // 2
        k_local = k % 2
        for i in range(n):
            for r in range(rank_332):
                W[k * n + i][block * rank_332 + r] = W_332[k_local * 3 + i][r]
                
    return {"U": U, "V": V, "W": W}

def _try_direct_338(rank, time_limit=20.0):
    """
    Try to find a direct rank-rank decomposition for 3x3x8.
    Uses structured random search.
    """
    n, m, p = 3, 3, 8
    num_eqs = n * m * p  # 72
    start_time = time.time()
    
    for attempt in range(300):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for r in range(rank):
            active_U = set()
            active_V = set()
            for _ in range(random.randint(2, 6)):
                active_U.add(random.randint(0, n*m-1))
            for _ in range(random.randint(2, 8)):
                active_V.add(random.randint(0, m*p-1))
            
            for idx in active_U:
                U[idx, r] = random.choice([-1, 1])
            for idx in active_V:
                V[idx, r] = random.choice([-1, 1])
        
        A = np.zeros((num_eqs, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        col_idx = k * n * rank + i * rank + r
                        A[eq_idx, col_idx] = coeff
        
        b = np.ones(num_eqs)
        
        try:
            W_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        if np.allclose(A @ W_sol, b, atol=1e-10):
            for mult in [1, 2]:
                W_rounded = np.round(W_sol * mult) / mult
                if np.allclose(A @ W_rounded, b, atol=1e-9):
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_rounded.reshape(p * n, rank).tolist()
                    if _validate(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    schoolbook = _schoolbook(n, m, p)
    
    # Strategy 1: Try to find rank-17 3x3x2 decomposition -> rank 68 for 3x3x8
    if time.time() < deadline - 40:
        result_332 = _build_332_decomposition(17)
        if result_332:
            U, V, W = result_332
            decomp = _construct_from_332(U, V, W)
            if _validate(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                return decomp
    
    # Strategy 2: Try direct 3x3x8 with rank 71
    if time.time() < deadline - 30:
        result = _try_direct_338(71, time_limit=25)
        if result:
            return result
    
    # Strategy 3: Try rank 70
    if time.time() < deadline - 20:
        result = _try_direct_338(70, time_limit=15)
        if result:
            return result
    
    # Fallback: schoolbook
    return schoolbook
