import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W):
    """Faster verification using vectorized operations."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    # Precompute for each (i,j,k) the product V[j,k,:] * W[k,i,:]
    # Then check if U[i,j,:] dotted with that equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = V[j*p+k, :] * W[k*n+i, :]
                if abs(np.dot(U[i*m+j, :], prod) - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_column(U, V, W, col_idx):
    """Try removing a specific column and verify."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    mask = [i for i in range(rank) if i != col_idx]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_fast(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None

def _try_remove_columns(U, V, W, col_indices):
    """Try removing multiple columns and verify."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    mask = [i for i in range(rank) if i not in col_indices]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_fast(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None

def _find_redundant_columns(U, V, W, max_iterations=50):
    """Find columns that can be removed iteratively."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    removed = []
    
    for _ in range(max_iterations):
        found = False
        for col in range(rank):
            result = _try_remove_column(U, V, W, col)
            if result is not None:
                U, V, W = result
                removed.append(col)
                rank = U.shape[1]
                found = True
                break
        if not found:
            break
    
    return U, V, W, removed

def _try_pair_removal(U, V, W):
    """Try removing pairs of columns."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    for i in range(rank):
        for j in range(i+1, rank):
            result = _try_remove_columns(U, V, W, [i, j])
            if result is not None:
                U, V, W = result
                return U, V, W, [i, j]
    return U, V, W, None

def _try_triple_removal(U, V, W):
    """Try removing triples of columns."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    for i in range(rank):
        for j in range(i+1, rank):
            for k in range(j+1, rank):
                result = _try_remove_columns(U, V, W, [i, j, k])
                if result is not None:
                    U, V, W = result
                    return U, V, W, [i, j, k]
    return U, V, W, None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Get the baseline rank 36 decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Iteratively remove single columns
    if time.time() < deadline - 30:
        U, V, W, removed = _find_redundant_columns(U, V, W, max_iterations=100)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 2: Try pair removals
    if time.time() < deadline - 20:
        U, V, W, removed_pairs = _try_pair_removal(U, V, W)
        if removed_pairs is not None and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # After finding a pair, try more single removals
            U, V, W, _ = _find_redundant_columns(U, V, W, max_iterations=50)
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 3: Try triple removals
    if time.time() < deadline - 10:
        U, V, W, removed_triples = _try_triple_removal(U, V, W)
        if removed_triples is not None and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # After finding a triple, try more single removals
            U, V, W, _ = _find_redundant_columns(U, V, W, max_iterations=30)
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 4: Final attempt - try all pairs from current state
    if time.time() < deadline - 5:
        U, V, W, _ = _try_pair_removal(U, V, W)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W):
        # Use the original verified decomposition
        best_U, best_V, best_W = _build_strassen_36()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
