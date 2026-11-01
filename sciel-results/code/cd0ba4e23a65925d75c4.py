import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _strassen_block(col0, col1, n, p):
    """
    Return the 7 Strassen terms for a 2x2 * 2x2 block covering columns col0, col1.
    Each term is a dict with U, V, W contributions.
    """
    terms = []
    
    # M1 = (A00 + A11) * (B0,col0 + B1,col1)
    terms.append({
        'U': {0: 1, 3: 1},
        'V': {0*p + col0: 1, 1*p + col1: 1},
        'W': {col0*n + 0: 1, col1*n + 1: 1}
    })
    
    # M2 = (A10 + A11) * B0,col0
    terms.append({
        'U': {2: 1, 3: 1},
        'V': {0*p + col0: 1},
        'W': {col0*n + 1: 1, col1*n + 1: -1}
    })
    
    # M3 = A00 * (B0,col1 - B1,col1)
    terms.append({
        'U': {0: 1},
        'V': {0*p + col1: 1, 1*p + col1: -1},
        'W': {col1*n + 0: 1, col1*n + 1: 1}
    })
    
    # M4 = A11 * (B1,col0 - B0,col0)
    terms.append({
        'U': {3: 1},
        'V': {1*p + col0: 1, 0*p + col0: -1},
        'W': {col0*n + 0: 1, col0*n + 1: 1}
    })
    
    # M5 = (A00 + A01) * B1,col1
    terms.append({
        'U': {0: 1, 1: 1},
        'V': {1*p + col1: 1},
        'W': {col0*n + 0: -1, col1*n + 0: 1}
    })
    
    # M6 = (A10 - A00) * (B0,col0 + B0,col1)
    terms.append({
        'U': {2: 1, 0: -1},
        'V': {0*p + col0: 1, 0*p + col1: 1},
        'W': {col1*n + 1: 1}
    })
    
    # M7 = (A01 - A11) * (B1,col0 + B1,col1)
    terms.append({
        'U': {1: 1, 3: -1},
        'V': {1*p + col0: 1, 1*p + col1: 1},
        'W': {col0*n + 0: 1}
    })
    
    return terms

def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 6 blocks of 2 columns. Each block uses Strassen (7 terms) for the 2x2*2x2 sub-problem.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank_per_pair = 7
    rank = num_pairs * rank_per_pair  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * rank_per_pair
        col0 = 2 * t
        col1 = 2 * t + 1
        
        terms = _strassen_block(col0, col1, n, p)
        
        for s, term in enumerate(terms):
            c = base + s
            for idx, val in term['U'].items():
                U[idx][c] = val
            for idx, val in term['V'].items():
                V[idx][c] = val
            for idx, val in term['W'].items():
                W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
