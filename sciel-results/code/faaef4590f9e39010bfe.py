"""Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication."""
import time
import random
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 125.0  # safety margin below 150s kill

def _check_validity(U, V, W, n, m, p, max_sample=200):
    """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == sum_l A[i,l]*B[l,k]."""
    r = len(U[0]) if U else 0
    if len(U) != n*m: return False
    if len(V) != m*p: return False
    if len(W) != p*n: return False
    for row in U + V + W:
        if len(row) != r: return False
    
    # Use small integer values for speed
    for trial in range(min(max_sample, 100)):
        rng = random.Random(trial * 98765 + 42)
        A = [[rng.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        # True product
        C_true = [[0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for l in range(m):
                    s += A[i][l] * B[l][k]
                C_true[i][k] = s
        # Decoded product
        C_dec = [[0 for _ in range(p)] for _ in range(n)]
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    uj = U[i*m + j][c]
                    if uj == 0: continue
                    for k in range(p):
                        vk = V[j*p + k][c]
                        if vk == 0: continue
                        wval = W[k*n + i][c]
                        if wval != 0:
                            C_dec[i][k] += uj * vk * wval
        for i in range(n):
            for k in range(p):
                if C_dec[i][k] != C_true[i][k]:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_222():
    """Return (U,V,W) for 2x2x2 with rank 7 (Strassen)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, -1, 1, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return {"U": U, "V": V, "W": W}

def _construct_rank_39():
    """
    Construct rank 39 decomposition:
    - Split middle dimension l into {0} and {1,2}
    - For l=0: R(2,1,7) = 14 (schoolbook)
    - For l in {1,2}: R(2,2,7), split k into (0,1),(2,3),(4,5),(6)
      - Three 2x2x2 Strassen blocks: 7*3 = 21
      - One 2x2x1 schoolbook block: 4
    Total: 14 + 21 + 4 = 39
    """
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0]*rank for _ in range(n*m)]  # 6 x 39
    V = [[0]*rank for _ in range(m*p)]  # 21 x 39
    W = [[0]*rank for _ in range(p*n)]  # 14 x 39
    
    col = 0
    
    # Part 1: l=0, all k=0..6 (schoolbook 2x1x7)
    for i in range(2):
        for k in range(7):
            U[i*3 + 0][col] = 1
            V[0*7 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    # Part 2: l in {1,2}, k split into pairs and single
    def strassen_222_block(col_offset, k_vals, l_base):
        """Add Strassen block for 2x2 (l in {l_base, l_base+1}) x 2 (k in k_vals)."""
        s = _construct_strassen_222()
        sU, sV, sW = s["U"], s["V"], s["W"]
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    val = sU[i*2 + j][c]
                    if val != 0:
                        U[i*3 + (l_base + j)][col_offset + c] = val
            for j in range(2):
                for k in range(2):
                    val = sV[j*2 + k][c]
                    if val != 0:
                        V[(l_base + j)*7 + (k_vals[0] + k)][col_offset + c] = val
            for k in range(2):
                for i in range(2):
                    val = sW[k*2 + i][c]
                    if val != 0:
                        W[(k_vals[0] + k)*2 + i][col_offset + c] = val
        return col_offset + 7
    
    def schoolbook_221_block(col_offset, k_val, l_base):
        """Add schoolbook block for 2x2 (l) x 1 (k)."""
        for i in range(2):
            for j in range(2):
                U[i*3 + (l_base + j)][col_offset] = 1
                V[(l_base + j)*7 + k_val][col_offset] = 1
                W[k_val*2 + i][col_offset] = 1
                col_offset += 1
        return col_offset
    
    # k-pairs with l in {1,2}
    col = strassen_222_block(col, (0, 1), 1)
    col = strassen_222_block(col, (2, 3), 1)
    col = strassen_222_block(col, (4, 5), 1)
    col = schoolbook_221_block(col, 6, 1)
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}

def _try_rank_38_via_column_parallel():
    """
    Attempt to find rank 38 decomposition using column-parallel approach.
    Treat the 7 output columns as independent linear maps.
    
    The idea: For each output column k, the multiplication map is a bilinear form
    in A and B. We want to find a shared set of rank-1 tensors that span all 7 columns.
    
    This is equivalent to finding a tensor of format 2x3x7 with rank 38.
    
    Strategy: Start from rank 39 and try to merge one term via linear dependency.
    """
    # For now, return None to indicate this approach is not yet implemented
    # A full implementation would require solving a system of polynomial equations
    # over the rationals, which is computationally expensive.
    return None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n, m, p = 2, 3, 7
    elapsed = time.time() - _START_TIME
    
    # Try rank 38 first (most ambitious)
    if elapsed < TIME_LIMIT - 30:
        try:
            result38 = _try_rank_38_via_column_parallel()
            if result38 is not None:
                if _check_validity(result38["U"], result38["V"], result38["W"], n, m, p):
                    elapsed = time.time() - _START_TIME
                    if elapsed < TIME_LIMIT:
                        return result38
        except Exception:
            pass
    
    # Try rank 39 (block decomposition with Strassen)
    if elapsed < TIME_LIMIT - 10:
        try:
            result39 = _construct_rank_39()
            if _check_validity(result39["U"], result39["V"], result39["W"], n, m, p):
                elapsed = time.time() - _START_TIME
                if elapsed < TIME_LIMIT:
                    return result39
        except Exception:
            pass
    
    # Fall back to schoolbook rank 42
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT:
        result42 = _schoolbook(n, m, p)
        return result42
    
    # Last resort
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
