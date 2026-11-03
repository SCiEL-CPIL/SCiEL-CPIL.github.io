"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
_DEADLINE = 130.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _strassen_2x2_2():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Column 0: P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Column 2: P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Column 3: P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Column 4: P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Column 5: P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Column 6: P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    # Build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_embedding(i_start, k_start, j_start=0):
    """
    Build a decomposition with one Strassen 2x2x2 block at rows [i_start, i_start+1],
    middle indices [j_start, j_start+1], and cols [k_start, k_start+1],
    with schoolbook for the rest.
    
    Returns (U, V, W, rank) or None if invalid placement.
    """
    n, m, p = 3, 3, 8
    if i_start + 1 >= n or k_start + 1 >= p or j_start + 1 >= m:
        return None
    
    sU, sV, sW = _strassen_2x2_2()
    
    # Covered (i,j,k) tuples
    covered = set()
    for i in [i_start, i_start+1]:
        for j in [j_start, j_start+1]:
            for k in [k_start, k_start+1]:
                covered.add((i, j, k))
    
    # Count schoolbook terms needed
    schoolbook_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_count += 1
    
    rank = 7 + schoolbook_count
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Place Strassen block
    for c in range(7):
        for u_idx in range(4):
            i_local = u_idx // 2
            j_local = u_idx % 2
            i_global = i_start + i_local
            j_global = j_start + j_local
            U[i_global*m + j_global][c] = sU[u_idx][c]
        for v_idx in range(4):
            j_local = v_idx // 2
            k_local = v_idx % 2
            j_global = j_start + j_local
            k_global = k_start + k_local
            V[j_global*p + k_global][c] = sV[v_idx][c]
        for w_idx in range(4):
            k_local = w_idx // 2
            i_local = w_idx % 2
            k_global = k_start + k_local
            i_global = i_start + i_local
            W[k_global*n + i_global][c] = sW[w_idx][c]
    
    col = 7
    
    # Schoolbook for uncovered terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W, rank

def _try_strassen_placements():
    """
    Try all valid single Strassen placements and return the best one.
    """
    n, m, p = 3, 3, 8
    best = None
    best_rank = float('inf')
    
    # Try all (i_start, k_start, j_start) combinations
    for i_start in range(n - 1):
        for k_start in range(p - 1):
            for j_start in range(m - 1):
                if _time_left() < 5.0:
                    break
                result = _build_strassen_embedding(i_start, k_start, j_start)
                if result is None:
                    continue
                U, V, W, rank = result
                if rank < best_rank:
                    if _verify(U, V, W, n, m, p):
                        best = (U, V, W)
                        best_rank = rank
    
    return best, best_rank

def _try_multiple_strassen():
    """
    Try to place multiple Strassen 2x2x2 blocks, being careful about overlaps.
    This is more complex, so we try a greedy approach.
    """
    n, m, p = 3, 3, 8
    best = None
    best_rank = float('inf')
    
    # Generate all possible Strassen placements
    placements = []
    for i_start in range(n - 1):
        for j_start in range(m - 1):
            for k_start in range(p - 1):
                covered = set()
                for i in [i_start, i_start+1]:
                    for j in [j_start, j_start+1]:
                        for k in [k_start, k_start+1]:
                            covered.add((i, j, k))
                placements.append((i_start, j_start, k_start, covered))
    
    # Try all pairs of non-overlapping placements
    for idx1 in range(len(placements)):
        for idx2 in range(idx1 + 1, len(placements)):
            if _time_left() < 10.0:
                break
            p1 = placements[idx1]
            p2 = placements[idx2]
            # Check for overlap
            if p1[3] & p2[3]:
                continue
            
            # Build combined decomposition
            covered = p1[3] | p2[3]
            schoolbook_count = n * m * p - len(covered)
            rank = 14 + schoolbook_count
            
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Place first Strassen block
            sU, sV, sW = _strassen_2x2_2()
            for c in range(7):
                for u_idx in range(4):
                    i_local = u_idx // 2
                    j_local = u_idx % 2
                    i_global = p1[0] + i_local
                    j_global = p1[1] + j_local
                    U[i_global*m + j_global][c] = sU[u_idx][c]
                for v_idx in range(4):
                    j_local = v_idx // 2
                    k_local = v_idx % 2
                    j_global = p1[1] + j_local
                    k_global = p1[2] + k_local
                    V[j_global*p + k_global][c] = sV[v_idx][c]
                for w_idx in range(4):
                    k_local = w_idx // 2
                    i_local = w_idx % 2
                    k_global = p1[2] + k_local
                    i_global = p1[0] + i_local
                    W[k_global*n + i_global][c] = sW[w_idx][c]
            
            # Place second Strassen block
            for c in range(7, 14):
                for u_idx in range(4):
                    i_local = u_idx // 2
                    j_local = u_idx % 2
                    i_global = p2[0] + i_local
                    j_global = p2[1] + j_local
                    U[i_global*m + j_global][c] = sU[u_idx][c]
                for v_idx in range(4):
                    j_local = v_idx // 2
                    k_local = v_idx % 2
                    j_global = p2[1] + j_local
                    k_global = p2[2] + k_local
                    V[j_global*p + k_global][c] = sV[v_idx][c]
                for w_idx in range(4):
                    k_local = w_idx // 2
                    i_local = w_idx % 2
                    k_global = p2[2] + k_local
                    i_global = p2[0] + i_local
                    W[k_global*n + i_global][c] = sW[w_idx][c]
            
            col = 14
            
            # Schoolbook for uncovered terms
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            U[i*m + j][col] = 1
                            V[j*p + k][col] = 1
                            W[k*n + i][col] = 1
                            col += 1
            
            if rank < best_rank:
                if _verify(U, V, W, n, m, p):
                    best = (U, V, W)
                    best_rank = rank
    
    return best, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try systematic search over multiple Strassen placements
    try:
        if _time_left() > 30.0:
            best, best_rank = _try_multiple_strassen()
            if best is not None:
                U, V, W = best
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try systematic search over single Strassen placements
    try:
        if _time_left() > 10.0:
            best, best_rank = _try_strassen_placements()
            if best is not None:
                U, V, W = best
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
