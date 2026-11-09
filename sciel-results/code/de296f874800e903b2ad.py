"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    if r == 0:
        return False
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_28():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4, C is 2x4.
    
    Split into 4 subproblems of size 2x2x2:
    Total rank: 4 * 7 = 28
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 7 terms for 2x2x2:
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0: C00+=1, C11+=1
        1: [(1,0,1), (1,1,-1)],   # M1: C10+=1, C11+=-1
        2: [(0,1,1), (1,1,1)],    # M2: C01+=1, C11+=1
        3: [(0,0,1), (1,0,1)],    # M3: C00+=1, C10+=1
        4: [(0,0,-1), (0,1,1)],   # M4: C00+=-1, C01+=1
        5: [(1,1,1)],             # M5: C11+=1
        6: [(0,0,1)],             # M6: C00+=1
    }
    
    col = 0
    for jp in range(2):  # j_pair: 0={0,1}, 1={2,3}
        for kp in range(2):  # k_pair: 0={0,1}, 1={2,3}
            for t in range(7):
                # U coefficients
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                # V coefficients
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                # W coefficients
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """Try to reduce rank by removing terms that are linear combinations of others."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for iteration in range(max_iterations):
        if _remaining() < 15.0:
            break
        if best_rank <= 14:
            break
        
        removed_any = False
        r = best_U.shape[1]
        
        for c in range(r):
            if _remaining() < 15.0:
                break
            
            # Try removing column c
            mask = [i for i in range(r) if i != c]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            if _verify(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return best_U, best_V, best_W, best_rank

def _block_strassen_2x4x4():
    """
    Structural approach: Treat the 2x4x4 multiplication as two independent 2x2x4 
    multiplications (splitting the 4x4 matrix into two 2x2 column blocks).
    
    For each 2x2x4 subproblem, we can use a rank-14 decomposition derived from 
    Strassen's algorithm applied to two 2x2x2 blocks with shared structure.
    
    Total rank: 2 * 14 = 28 (same as naive Strassen, but different structure 
    that may allow further reduction)
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # For 2x2x4: A is 2x2, B is 2x4 (two 2x2 blocks), C is 2x4
    # C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    # Each A*Bt is 2x2x2, solved with Strassen (rank 7)
    # But we can share some computations between the two blocks
    
    # Actually, let's use the same structure as _strassen_28 but organized differently
    # to potentially find different linear dependencies
    
    # Use the standard Strassen for each 2x2x2 block
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0: C00+=1, C11+=1
        1: [(1,0,1), (1,1,-1)],   # M1: C10+=1, C11+=-1
        2: [(0,1,1), (1,1,1)],    # M2: C01+=1, C11+=1
        3: [(0,0,1), (1,0,1)],    # M3: C00+=1, C10+=1
        4: [(0,0,-1), (0,1,1)],   # M4: C00+=-1, C01+=1
        5: [(1,1,1)],             # M5: C11+=1
        6: [(0,0,1)],             # M6: C00+=1
    }
    
    col = 0
    for jp in range(2):  # j_pair: 0={0,1}, 1={2,3}
        for kp in range(2):  # k_pair: 0={0,1}, 1={2,3}
            for t in range(7):
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _random_search(U_init, V_init, W_init, n, m, p, max_time=100.0):
    """
    Try to reduce rank by randomly perturbing columns and checking if the 
    decomposition still holds. This is a heuristic search for lower-rank 
    decompositions.
    """
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    best_rank = U_init.shape[1]
    
    # First, try simple column removal
    U, V, W = best_U, best_V, best_W
    rank = U.shape[1]
    
    # Greedy removal
    for _ in range(100):
        if _remaining() < 20.0:
            break
        if rank <= 14:
            break
        
        removed = False
        for c in range(rank):
            mask = [i for i in range(rank) if i != c]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if _verify(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank = len(mask)
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank
                removed = True
                break
        
        if not removed:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 4 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based decomposition (rank 28)
    U, V, W = _strassen_28()
    
    # Try to reduce rank via greedy column removal
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p)
    
    # If we still have time, try the block-based approach
    if _remaining() > 30.0 and rank > 20:
        U2, V2, W2 = _block_strassen_2x4x4()
        U2, V2, W2, rank2 = _try_reduce_rank(U2, V2, W2, n, m, p)
        if rank2 < rank:
            U, V, W, rank = U2, V2, W2, rank2
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                val_rounded = round(val * 2) / 2
                if abs(val_rounded) < 1e-9:
                    row_list.append(0)
                elif abs(val_rounded - round(val_rounded)) < 1e-9:
                    row_list.append(int(round(val_rounded)))
                else:
                    row_list.append(val_rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
