"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication.
    Each factor is (U_vec, V_vec, W_dict) where:
      U_vec: dict mapping (i,j) -> coefficient for A[i,j]
      V_vec: dict mapping (i,j) -> coefficient for B[i,j]
      W_dict: dict mapping (i,j) -> coefficient for C[i,j]
    
    Strassen:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B11)
      M4 = A22(B22-B12)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Using 0-indexed: A00,A01,A10,A11 and B00,B01,B10,B11, C00,C01,C10,C11
    factors = []
    
    # M1 = (A00+A11)(B00+B11)
    factors.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10+A11)B00
    factors.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # M3 = A00(B10-B00)
    factors.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A11(B11-B01)
    factors.append({
        'U': {(1,1): 1},
        'V': {(1,1): 1, (0,1): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00+A01)B11
    factors.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): -1}
    })
    
    # M6 = (A10-A00)(B00+B01)
    factors.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (A01-A11)(B10+B11)
    factors.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    return factors

def _build_strassen_4x4_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 * 4x4.
    Two levels of Strassen: 7*7 = 49 rank-1 terms.
    
    Level 1: Split 4x4 into 2x2 blocks of 2x2.
    Level 2: Each 2x2*2x2 product uses 7 Strassen scalar products.
    """
    # 4x4 * 4x4: indices i,j,k in 0..3
    # Block structure:
    # A blocks: A[0,0]=A[0:2,0:2], A[0,1]=A[0:2,2:4], A[1,0]=A[2:4,0:2], A[1,1]=A[2:4,2:4]
    # B blocks: B[0,0]=B[0:2,0:2], B[0,1]=B[0:2,2:4], B[1,0]=B[2:4,0:2], B[1,1]=B[2:4,2:4]
    # C blocks: C[0,0]=C[0:2,0:2], C[0,1]=C[0:2,2:4], C[1,0]=C[2:4,0:2], C[1,1]=C[2:4,2:4]
    
    # Level 1 Strassen factors (for the 2x2 block matrix):
    # Each factor has U_coeffs, V_coeffs, W_coeffs at the block level.
    # Then each block-level product is expanded using level 2 Strassen.
    
    level1_factors = _strassen_2x2_factors()
    
    U = [[0] * 49 for _ in range(16)]  # 4*4 = 16 entries
    V = [[0] * 49 for _ in range(16)]  # 4*4 = 16 entries
    W = [[0] * 49 for _ in range(16)]  # 4*4 = 16 entries
    
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            # Level 1 factor l1 gives block-level coefficients
            # Level 2 factor l2 gives scalar-level coefficients within each block
            
            l1_factor = level1_factors[l1]
            l2_factor = level1_factors[l2]
            
            # For each block (bi, bj) in the A-block matrix:
            # The coefficient for block (bi, bj) is l1_factor['U'][(bi, bj)]
            # Within that block, the scalar (si, sj) has coefficient l2_factor['U'][(si, sj)]
            # So A[(bi*2+si), (bj*2+sj)] gets coefficient l1_factor['U'][(bi,bj)] * l2_factor['U'][(si,sj)]
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['U']:
                        continue
                    c1 = l1_factor['U'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['U']:
                                continue
                            c2 = l2_factor['U'][(si, sj)]
                            ai = bi * 2 + si
                            aj = bj * 2 + sj
                            U[ai * 4 + aj][col] += c1 * c2
            
            # Similarly for B
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['V']:
                        continue
                    c1 = l1_factor['V'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['V']:
                                continue
                            c2 = l2_factor['V'][(si, sj)]
                            # B block (bi, bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
                            # B index: row = bi*2+si, col = bj*2+sj
                            # V index: row * 4 + col
                            bri = bi * 2 + si
                            bcj = bj * 2 + sj
                            V[bri * 4 + bcj][col] += c1 * c2
            
            # Similarly for C
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['W']:
                        continue
                    c1 = l1_factor['W'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['W']:
                                continue
                            c2 = l2_factor['W'][(si, sj)]
                            # C block (bi, bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
                            # C index: row = bi*2+si, col = bj*2+sj
                            # W index: col * 4 + row (trace convention: W[k*n+i, r] where k=col, i=row)
                            cri = bi * 2 + si
                            ccj = bj * 2 + sj
                            W[ccj * 4 + cri][col] += c1 * c2
            
            col += 1
    
    return U, V, W

def _build_strassen_tiled(n, m, p):
    """
    Build a Strassen-tiled decomposition for 4x4 * 4x6.
    Split into 4x4*4x4 (Strassen, rank 49) + 4x4*4x2 (schoolbook, rank 32).
    Total rank: 81.
    """
    n, m, p = 4, 4, 6
    rank_strassen = 49
    rank_schoolbook = n * m * (p - 4)  # 4*4*2 = 32
    rank_total = rank_strassen + rank_schoolbook
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B), using Strassen rank 49
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    for i in range(n * m):
        for c in range(rank_strassen):
            U[i][c] = U_s[i][c]
    for i in range(16):  # m * 4 = 16
        for c in range(rank_strassen):
            V[i][c] = V_s[i][c]
    for i in range(16):  # 4 * n = 16
        for c in range(rank_strassen):
            W[i][c] = W_s[i][c]
    
    # Part 2: 4x4 * 4x2 (last 2 columns of B), using schoolbook rank 32
    col = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4 and 5 of B
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p
    
    # Try Strassen-tiled approach
    strassen = _build_strassen_tiled(n, m, p)
    if strassen is not None:
        if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            rank = len(strassen["U"][0])
            if rank < best_rank:
                best = strassen
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
