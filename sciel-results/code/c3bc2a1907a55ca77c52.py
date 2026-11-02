import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 Strassen blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A, B, C
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: j*6 + k
    # C indices: k*2 + i
    
    # Define the 7 Strassen terms for a pair of columns (ka, kb)
    # Each term is a list of (index, coeff) for U, V, W
    # U: (idx, coeff)
    # V: (idx, coeff)
    # W: (idx, coeff)
    
    def get_strassen_terms(ka, kb):
        terms = []
        
        # M1
        terms.append({
            'U': [(0, 1), (3, 1)],
            'V': [(0*6 + ka, 1), (1*6 + kb, 1)],
            'W': [(ka*2 + 0, 1), (kb*2 + 1, 1)]
        })
        # M2
        terms.append({
            'U': [(2, 1), (3, 1)],
            'V': [(0*6 + ka, 1)],
            'W': [(ka*2 + 1, 1)]
        })
        # M3
        terms.append({
            'U': [(0, 1)],
            'V': [(0*6 + kb, 1), (1*6 + kb, -1)],
            'W': [(kb*2 + 0, 1)]
        })
        # M4
        terms.append({
            'U': [(3, 1)],
            'V': [(1*6 + ka, 1), (0*6 + kb, -1)],
            'W': [(ka*2 + 0, 1), (ka*2 + 1, 1)]
        })
        # M5
        terms.append({
            'U': [(0, 1), (1, 1)],
            'V': [(1*6 + kb, 1)],
            'W': [(ka*2 + 0, -1), (kb*2 + 0, 1)]
        })
        # M6
        terms.append({
            'U': [(2, 1), (0, -1)],
            'V': [(0*6 + ka, 1), (0*6 + kb, 1)],
            'W': [(kb*2 + 1, 1)]
        })
        # M7
        terms.append({
            'U': [(1, 1), (3, -1)],
            'V': [(1*6 + ka, 1), (1*6 + kb, 1)],
            'W': [(ka*2 + 0, 1)]
        })
        
        return terms

    # 3 pairs of columns: (0,1), (2,3), (4,5)
    pairs = [(0, 1), (2, 3), (4, 5)]
    
    col = 0
    for ka, kb in pairs:
        terms = get_strassen_terms(ka, kb)
        for t in terms:
            for idx, coeff in t['U']:
                U[idx][col] += coeff
            for idx, coeff in t['V']:
                V[idx][col] += coeff
            for idx, coeff in t['W']:
                W[idx][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
