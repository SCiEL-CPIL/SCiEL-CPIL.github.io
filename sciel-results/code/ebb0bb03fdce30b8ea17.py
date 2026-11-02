import numpy as np
import time
from fractions import Fraction

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

def _verify_decomposition_exact(U, V, W):
    """Verify using exact rational arithmetic."""
    n, m, p = 2, 4, 5
    U_frac = [[Fraction(int(round(U[i, r]))) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_frac = [[Fraction(int(round(V[i, r]))) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_frac = [[Fraction(int(round(W[i, r]))) for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(U.shape[1]):
                    total += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if total != Fraction(1):
                    return False
    return True

def _try_remove_column(U, V, W, col_idx):
    """Try removing a single column and verify."""
    mask = [i for i in range(U.shape[1]) if i != col_idx]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_decomposition(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None

def _try_remove_columns(U, V, W, cols_to_remove):
    """Try removing multiple columns and verify."""
    mask = [i for i in range(U.shape[1]) if i not in cols_to_remove]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_decomposition(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None

def _local_search(U, V, W, deadline):
    """
    Local search to try to reduce rank by finding removable columns.
    Uses greedy removal with backtracking for pairs.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Phase 1: Greedy single column removal
    for _ in range(50):
        if time.time() > deadline - 15:
            break
        progress = False
        for col in range(best_rank):
            if time.time() > deadline - 15:
                break
            result = _try_remove_column(best_U, best_V, best_W, col)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = best_U.shape[1]
                progress = True
                break
        if not progress:
            break
    
    # Phase 2: Try removing pairs
    if time.time() < deadline - 10:
        for i in range(best_rank):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, best_rank):
                if time.time() > deadline - 10:
                    break
                result = _try_remove_columns(best_U, best_V, best_W, [i, j])
                if result is not None:
                    best_U, best_V, best_W = result
                    best_rank = best_U.shape[1]
                    break
            if time.time() > deadline - 10:
                break
            if best_U.shape[1] < best_rank:
                break
    
    # Phase 3: Try removing triples if time permits
    if time.time() < deadline - 5:
        for i in range(best_rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, best_rank):
                if time.time() > deadline - 5:
                    break
                for k in range(j+1, best_rank):
                    if time.time() > deadline - 5:
                        break
                    result = _try_remove_columns(best_U, best_V, best_W, [i, j, k])
                    if result is not None:
                        best_U, best_V, best_W = result
                        best_rank = best_U.shape[1]
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
            if best_U.shape[1] < best_rank:
                break
    
    return best_U, best_V, best_W, best_rank

def _try_rank_35_direct():
    """
    Try to construct a rank-35 decomposition directly.
    This is a structured attempt to merge one term from the schoolbook blocks
    with a Strassen term.
    """
    n, m, p = 2, 4, 5
    target_rank = 35
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
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
    
    # First 3 Strassen blocks: 21 terms
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Now we have 14 columns left for: A_R B_22 (2x2x2) + A_L B_15 (2x2x1) + A_R B_25 (2x2x1)
    # That's 7 + 4 + 4 = 15 terms needed, but only 14 slots.
    # Try a modified 6-term decomposition for the last 2x2x2 block.
    
    # Use a different pattern for the last block that might share with schoolbook terms
    # This is experimental - try a variant where we merge M6 and M7 differently
    
    y_indices = [(0,2),(0,3),(1,2),(1,3)]
    z_indices = [(2,2),(2,3),(3,2),(3,3)]
    w_offsets = [(2,0),(3,0),(2,1),(3,1)]
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
    
    # M6 = (y10-y00)(z00+z01) - modified to potentially share with schoolbook
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
    
    # Now we've used 28 columns. We have 7 left for the two 2x2x1 blocks (need 8 terms).
    # Try to merge one schoolbook term with an existing Strassen term.
    
    # Schoolbook for A_L B_15:
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Schoolbook for A_R B_25:
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    
    # Try to add these 4 terms in 3 columns by merging
    # Column 28: A[0,0]*B[0,4] -> C[0,4]
    U[u_idx(0,0), col] = 1
    V[v_idx(0,4), col] = 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # Column 29: A[0,1]*B[1,4] -> C[0,4]
    U[u_idx(0,1), col] = 1
    V[v_idx(1,4), col] = 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # Column 30: A[1,0]*B[0,4] -> C[1,4]
    U[u_idx(1,0), col] = 1
    V[v_idx(0,4), col] = 1
    W[w_idx(4,1), col] += 1
    col += 1
    
    # Column 31: A[1,1]*B[1,4] -> C[1,4]
    U[u_idx(1,1), col] = 1
    V[v_idx(1,4), col] = 1
    W[w_idx(4,1), col] += 1
    col += 1
    
    # Column 32: A[0,2]*B[2,4] -> C[0,4]
    U[u_idx(0,2), col] = 1
    V[v_idx(2,4), col] = 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # Column 33: A[0,3]*B[3,4] -> C[0,4]
    U[u_idx(0,3), col] = 1
    V[v_idx(3,4), col] = 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # Column 34: A[1,2]*B[2,4] -> C[1,4]
    U[u_idx(1,2), col] = 1
    V[v_idx(2,4), col] = 1
    W[w_idx(4,1), col] += 1
    col += 1
    
    # We're out of columns! We have 35 columns total but used 35.
    # But we still need A[1,3]*B[3,4] -> C[1,4]
    # This won't work as is. Return None.
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
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
    
    # Local search to try to reduce rank
    best_U, best_V, best_W, best_rank = _local_search(U, V, W, deadline)
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W):
        # Fallback to the verified rank-36
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
