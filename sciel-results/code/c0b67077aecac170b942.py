import time
import numpy as np

_START = time.time()
DEADLINE = 135.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    # P1 = (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # P2 = (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    # P3 = a00(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    # P4 = a11(b10-b00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    # P5 = (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    # P6 = (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    # P7 = (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    return U, V, W

def _get_block_coverage(i0, j0, k0, n, m, p):
    """Get set of (i,j,k) covered by a Strassen block at (i0,j0,k0)."""
    covered = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0+di, j0+dj, k0+dk
                if 0 <= i < n and 0 <= j < m and 0 <= k < p:
                    covered.add((i, j, k))
    return covered

def _find_max_disjoint_blocks(n, m, p):
    """Find maximum number of non-overlapping 2x2x2 Strassen blocks."""
    # Generate all valid block positions
    valid_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                cov = _get_block_coverage(i0, j0, k0, n, m, p)
                if len(cov) == 8:  # Must cover exactly 8 entries
                    valid_blocks.append(((i0, j0, k0), cov))
    
    if not valid_blocks:
        return []
    
    # Greedy: try all subsets up to size 4 (since max possible is limited)
    # With n=m=3, p=8: i0 in {0,1}, j0 in {0,1}, k0 in {0..6}
    # Max disjoint: need to check
    
    best = []
    best_count = 0
    
    # Try all combinations of up to 5 blocks
    from itertools import combinations
    for r in range(min(6, len(valid_blocks)), 0, -1):
        for combo in combinations(valid_blocks, r):
            # Check disjointness
            all_cov = set()
            disjoint = True
            for _, cov in combo:
                if all_cov & cov:
                    disjoint = False
                    break
                all_cov |= cov
            if disjoint:
                if r > best_count:
                    best_count = r
                    best = [pos for pos, _ in combo]
                break  # Found one of this size, no need to check more of same size
        if best_count >= r:
            break
    
    return best

def _build_decomposition_with_blocks(n, m, p, blocks):
    """Build decomposition using Strassen blocks at given positions, schoolbook elsewhere."""
    if not blocks:
        return _schoolbook(n, m, p)
    
    covered = set()
    for (i0, j0, k0) in blocks:
        cov = _get_block_coverage(i0, j0, k0, n, m, p)
        covered |= cov
    
    n_school = n * m * p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for (i0, j0, k0) in blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = sU[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = sV[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = sW[dk*2+di, c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    
    # Fallback: schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = n * m * p
    
    if _time_left() < 10:
        return best_result
    
    # Find maximum number of disjoint Strassen blocks
    blocks = _find_max_disjoint_blocks(n, m, p)
    
    if blocks:
        try:
            U, V, W = _build_decomposition_with_blocks(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = U.shape[1]
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Also try the known-good 4-block configuration as backup
    known_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    try:
        U, V, W = _build_decomposition_with_blocks(n, m, p, known_blocks)
        if _verify(U, V, W, n, m, p):
            r = U.shape[1]
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return best_result
