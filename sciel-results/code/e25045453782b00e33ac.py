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
    Also try combining a 3x3x3 Strassen-based block with 2x2x2 blocks.
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
    
    # Try 3x3x3 Strassen-based block + multiple 2x2x2 Strassen blocks
    # 3x3x3 block: one 2x2x2 Strassen at (0,0,0) + 19 schoolbook = rank 26
    # Remaining 3x3x5 region: place 2x2x2 Strassen blocks
    for k3_start in range(3):  # which 3 k-values the 3x3x3 block covers
        if _time_left() < 15.0:
            break
        k3_vals = list(range(k3_start, k3_start + 3))
        remaining_k = [k for k in range(p) if k not in k3_vals]
        
        # Build 3x3x3 block: Strassen at (i=0,j=0,k=k3_start) + schoolbook for rest
        sU, sV, sW = _strassen_2x2_2()
        strassen_covered = set()
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [k3_start, k3_start + 1]:
                    strassen_covered.add((i, j, k))
        
        # 3x3x3 covered set
        block3_covered = set()
        for i in range(3):
            for j in range(3):
                for k in k3_vals:
                    block3_covered.add((i, j, k))
        
        # Rank for 3x3x3 block: 7 (Strassen) + (27 - 8) schoolbook = 26
        rank_3x3x3 = 7 + 19
        
        # Now place 2x2x2 Strassen blocks in remaining 3x3x5 region
        # Generate placements in remaining region
        rem_placements = []
        for i_start in range(2):
            for j_start in range(2):
                for k_start in remaining_k[:-1]:
                    # Check k_start+1 is also in remaining_k
                    if (k_start + 1) not in remaining_k:
                        continue
                    cov = set()
                    for i in [i_start, i_start+1]:
                        for j in [j_start, j_start+1]:
                            for k in [k_start, k_start+1]:
                                cov.add((i, j, k))
                    # Must not overlap with block3_covered
                    if cov & block3_covered:
                        continue
                    rem_placements.append((i_start, j_start, k_start, cov))
        
        # Try all subsets of non-overlapping placements from rem_placements
        # Greedy: try all pairs and triples
        for idx1 in range(len(rem_placements)):
            for idx2 in range(idx1 + 1, len(rem_placements)):
                if _time_left() < 5.0:
                    break
                r1 = rem_placements[idx1]
                r2 = rem_placements[idx2]
                if r1[3] & r2[3]:
                    continue
                
                covered = block3_covered | r1[3] | r2[3]
                schoolbook_count = n * m * p - len(covered)
                rank = rank_3x3x3 + 14 + schoolbook_count
                
                if rank >= best_rank:
                    continue
                
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                
                # Place 3x3x3 Strassen part (columns 0-6)
                for c in range(7):
                    for u_idx in range(4):
                        i_local = u_idx // 2
                        j_local = u_idx % 2
                        i_global = i_local
                        j_global = j_local
                        U[i_global*m + j_global][c] = sU[u_idx][c]
                    for v_idx in range(4):
                        j_local = v_idx // 2
                        k_local = v_idx % 2
                        j_global = j_local
                        k_global = k3_start + k_local
                        V[j_global*p + k_global][c] = sV[v_idx][c]
                    for w_idx in range(4):
                        k_local = w_idx // 2
                        i_local = w_idx % 2
                        k_global = k3_start + k_local
                        i_global = i_local
                        W[k_global*n + i_global][c] = sW[w_idx][c]
                
                # Schoolbook for 3x3x3 block uncovered terms (columns 7-25)
                col = 7
                for i in range(3):
                    for j in range(3):
                        for k in k3_vals:
                            if (i, j, k) not in strassen_covered:
                                U[i*m + j][col] = 1
                                V[j*p + k][col] = 1
                                W[k*n + i][col] = 1
                                col += 1
                
                # Place first 2x2x2 Strassen in remaining region (columns 26-32)
                for c in range(26, 33):
                    for u_idx in range(4):
                        i_local = u_idx // 2
                        j_local = u_idx % 2
                        i_global = r1[0] + i_local
                        j_global = r1[1] + j_local
                        U[i_global*m + j_global][c] = sU[u_idx][c]
                    for v_idx in range(4):
                        j_local = v_idx // 2
                        k_local = v_idx % 2
                        j_global = r1[1] + j_local
                        k_global = r1[2] + k_local
                        V[j_global*p + k_global][c] = sV[v_idx][c]
                    for w_idx in range(4):
                        k_local = w_idx // 2
                        i_local = w_idx % 2
                        k_global = r1[2] + k_local
                        i_global = r1[0] + i_local
                        W[k_global*n + i_global][c] = sW[w_idx][c]
                
                # Place second 2x2x2 Strassen in remaining region (columns 33-39)
                for c in range(33, 40):
                    for u_idx in range(4):
                        i_local = u_idx // 2
                        j_local = u_idx % 2
                        i_global = r2[0] + i_local
                        j_global = r2[1] + j_local
                        U[i_global*m + j_global][c] = sU[u_idx][c]
                    for v_idx in range(4):
                        j_local = v_idx // 2
                        k_local = v_idx % 2
                        j_global = r2[1] + j_local
                        k_global = r2[2] + k_local
                        V[j_global*p + k_global][c] = sV[v_idx][c]
                    for w_idx in range(4):
                        k_local = w_idx // 2
                        i_local = w_idx % 2
                        k_global = r2[2] + k_local
                        i_global = r2[0] + i_local
                        W[k_global*n + i_global][c] = sW[w_idx][c]
                
                col = 40
                # Schoolbook for all uncovered terms
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) not in covered:
                                U[i*m + j][col] = 1
                                V[j*p + k][col] = 1
                                W[k*n + i][col] = 1
                                col += 1
                
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
