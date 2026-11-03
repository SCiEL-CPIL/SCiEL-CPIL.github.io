"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

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
    """
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
    """
    level1_factors = _strassen_2x2_factors()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            l1_factor = level1_factors[l1]
            l2_factor = level1_factors[l2]
            
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
                            bri = bi * 2 + si
                            bcj = bj * 2 + sj
                            V[bri * 4 + bcj][col] += c1 * c2
            
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
    rank_schoolbook = n * m * (p - 4)
    rank_total = rank_strassen + rank_schoolbook
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    for i in range(n * m):
        for c in range(rank_strassen):
            U[i][c] = U_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            V[i][c] = V_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            W[i][c] = W_s[i][c]
    
    col = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_grobner_search(n, m, p, start_rank, max_rank, time_limit):
    """
    Attempt to find a decomposition of rank r using Gröbner basis / linear algebra.
    This is a heuristic search for the minimal rank.
    """
    # For 4x4x6, the rank is likely 96 (schoolbook) or 81 (strassen tiled).
    # Let's try to find if rank < 81 is possible.
    # The number of variables is r * (nm + mp + pn).
    # The number of equations is n*p * m = 4*6*4 = 96.
    # This is a system of 96 quadratic equations in 3*r variables.
    
    # We will try to solve for r starting from a lower bound.
    # Lower bound for 4x4x6: max(n*m, m*p, p*n) = max(16, 24, 24) = 24.
    # Actually, a better lower bound is ceil(n*m*p / max(n,m,p)) = 96/6 = 16.
    # Known rank for 4x4x4 is 49.
    # Rank for 4x4x6 is at most 49 + 32 = 81.
    
    # We'll try to find a solution for r in [start_rank, max_rank]
    # This is computationally heavy, so we only do it if we have time.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
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
    
    # If we have time, try to search for a lower rank using a more advanced method.
    # For this specific task, 81 is likely the best known or close to it.
    # A full Gröbner basis computation for rank 80 would be very slow.
    # We will stick with the Strassen-tiled result as it is a significant improvement over schoolbook (96 -> 81).
    
    return best
# EVOLVE-BLOCK-END
