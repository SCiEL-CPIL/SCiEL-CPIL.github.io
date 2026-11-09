"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

def _build_rank46():
    """
    Decomposition: two independent j-blocks (j in {0,1}).
    Each j-block covers k in {0..6}, split into three k-chunks of size 2 (Strassen 2x2x2)
    plus one k of size 1 (schoolbook 2x2x1).
    Rank per j-block = 3*7 + 2 = 23.  Total rank = 2*23 = 46.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7

    # Strassen 2x2x2 factors (rank 7).
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    # Term 0: M0 = A00*(B01+B11)
    U2[0][0] = 1; V2[1][0] = 1; V2[3][0] = 1; W2[1][0] = 1; W2[3][0] = 1
    # Term 1: M1 = A10*(B00+B01)
    U2[2][1] = 1; V2[0][1] = 1; V2[1][1] = 1; W2[0][1] = 1; W2[2][1] = 1
    # Term 2: M2 = (A00+A10)*B00
    U2[0][2] = 1; U2[2][2] = 1; V2[0][2] = 1; W2[0][2] = 1
    # Term 3: M3 = A11*(B10+B11)
    U2[3][3] = 1; V2[2][3] = 1; V2[3][3] = 1; W2[1][3] = 1; W2[3][3] = 1
    # Term 4: M4 = (A01-A11)*B10
    U2[1][4] = 1; U2[3][4] = -1; V2[2][4] = 1; W2[1][4] = 1
    # Term 5: M5 = (A10+A11)*(B11-B10)
    U2[2][5] = 1; U2[3][5] = 1; V2[2][5] = -1; V2[3][5] = 1; W2[2][5] = 1; W2[3][5] = 1
    # Term 6: M6 = (A00-A01)*B01
    U2[0][6] = 1; U2[1][6] = -1; V2[1][6] = 1; W2[2][6] = 1; W2[3][6] = 1

    # Schoolbook 2x2x1 factors (rank 2).
    U1 = [[0]*2 for _ in range(4)]
    V1 = [[0]*2 for _ in range(4)]
    W1 = [[0]*2 for _ in range(4)]
    # Term 0: A00*B00
    U1[0][0] = 1; V1[0][0] = 1; W1[0][0] = 1
    # Term 1: A10*B10
    U1[2][1] = 1; V1[2][1] = 1; W1[0][1] = 1

    rank = 2 * (3*7 + 2)  # 46
    U = [[0]*rank for _ in range(n*m)]  # 8 rows
    V = [[0]*rank for _ in range(m*p)]  # 28 rows
    W = [[0]*rank for _ in range(p*n)]  # 14 rows

    def col_idx(jblock, kchunk, t):
        per_jblock = 3*7 + 2  # 23
        base = jblock * per_jblock
        if kchunk < 3:
            return base + kchunk*7 + t
        else:
            return base + 3*7 + t

    for jblock in range(2):
        jj = jblock
        for i in range(n):
            for kchunk in range(3):
                for t in range(7):
                    col = col_idx(jblock, kchunk, t)
                    U[i*m + jj][col] += U2[i*2 + jj][t]
                    for kk in range(2):
                        k_val = kchunk*2 + kk
                        V[jj*p + k_val][col] += V2[jj*2 + kk][t]
                        W[k_val*n + i][col] += W2[kk*2 + i][t]
            for t in range(2):
                col = col_idx(jblock, 3, t)
                U[i*m + jj][col] += U1[i*2 + jj][t]
                V[jj*p + 6][col] += V1[jj*2 + 0][t]
                W[6*n + i][col] += W1[0*2 + i][t]

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if s != 1:
                    return False
    return True


def _build_schoolbook(n, m, p):
    rank = n * m * p
    Us = [[0]*rank for _ in range(n*m)]
    Vs = [[0]*rank for _ in range(m*p)]
    Ws = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j][column] = 1
                Vs[j*p+k][column] = 1
                Ws[k*n+i][column] = 1
                column += 1
    return {"U": Us, "V": Vs, "W": Ws}


def _try_rank_reduction(base_U, base_V, base_W, n, m, p, max_attempts=500):
    """
    Try to reduce rank by merging columns using linear algebra over rationals.
    For each pair of columns (a, b), check if we can find alpha, beta such that
    the tensor equation still holds when column a is replaced by alpha*col_a + beta*col_b
    and column b is removed.
    """
    rank = len(base_U[0])
    if rank <= 40:
        return None
    
    U = [row[:] for row in base_U]
    V = [row[:] for row in base_V]
    W = [row[:] for row in base_W]
    
    current_rank = rank
    attempts = 0
    deadline = time.time() + 120
    
    while current_rank > 40 and attempts < max_attempts and time.time() < deadline:
        attempts += 1
        found = False
        
        # Try to find two columns that can be merged
        for a in range(current_rank):
            for b in range(a+1, current_rank):
                # Check if columns a and b are "close" in some sense
                # We'll try to find alpha such that col_a - alpha*col_b has small support
                # and then check if the remaining system is consistent.
                
                # Quick heuristic: if the columns have similar patterns, try merging
                # Compute the "distance" between columns
                dist = 0
                for i in range(n*m):
                    if U[i][a] != U[i][b]:
                        dist += 1
                for j in range(m*p):
                    if V[j][a] != V[j][b]:
                        dist += 1
                for k in range(p*n):
                    if W[k][a] != W[k][b]:
                        dist += 1
                
                if dist > 10:
                    continue
                
                # Try to find alpha such that the merged column works
                # We need to solve: for all (i,j,k),
                # sum_c' U'[i*m+j,c'] * V'[j*p+k,c'] * W'[k*n+i,c'] = 1
                # where c' ranges over current_rank-1 columns after merging a into b
                
                # Simplified: try alpha = 0, 1, -1, 0.5, -0.5
                for alpha in [0, 1, -1, 0.5, -0.5]:
                    # Create new matrices with column a merged into column b
                    U_new = [row[:] for row in U]
                    V_new = [row[:] for row in V]
                    W_new = [row[:] for row in W]
                    
                    # Remove column a, add alpha*col_a to col_b
                    for i in range(n*m):
                        U_new[i][b] += alpha * U_new[i][a]
                        del U_new[i][a]
                    for j in range(m*p):
                        V_new[j][b] += alpha * V_new[j][a]
                        del V_new[j][a]
                    for k in range(p*n):
                        W_new[k][b] += alpha * W_new[k][a]
                        del W_new[k][a]
                    
                    # Verify
                    if _verify(U_new, V_new, W_new, n, m, p):
                        U, V, W = U_new, V_new, W_new
                        current_rank -= 1
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
        
        if not found:
            break
    
    if current_rank < rank:
        return {"U": U, "V": V, "W": W}
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with rank-46 decomposition
    result = _build_rank46()
    
    # Verify correctness
    try:
        ok = _verify(result["U"], result["V"], result["W"], n, m, p)
        if not ok:
            return _build_schoolbook(n, m, p)
    except Exception:
        return _build_schoolbook(n, m, p)
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        reduced = _try_rank_reduction(result["U"], result["V"], result["W"], n, m, p)
        if reduced is not None:
            try:
                if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    result = reduced
            except Exception:
                pass
    
    return result
# EVOLVE-BLOCK-END
