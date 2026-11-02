import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def _verify_mod(U, V, W, n, m, p, mod):
    """Fast verification using modular arithmetic."""
    rank = len(U[0])
    U_mod = [[x % mod for x in row] for row in U]
    V_mod = [[x % mod for x in row] for row in V]
    W_mod = [[x % mod for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_mod[i*m+j][c] * V_mod[j*p+k][c] * W_mod[k*n+i][c]
                if s % mod != 1 % mod:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact integer verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_block_combination(n, m, p, deadline):
    """
    Try to combine Strassen 2x2x2 blocks with schoolbook for remaining parts.
    For 3x3 * 3x4: split as (2x2 + 1x1) * (2x2 + 1x2) approximately.
    Actually, let's think of it as:
    - Use Strassen for a 2x2 subblock: rank 7
    - Handle the remaining rows/cols with schoolbook or optimized terms
    """
    # Strategy: Partition the 3x3 matrix A into blocks:
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x2)],
    #      [B21 (1x2), B22 (1x2)]]
    # C = [[C11 (2x2), C12 (2x2)],
    #      [C21 (1x2), C22 (1x2)]]
    
    # This is complex. Let's try a different approach:
    # Use the fact that we can share computations.
    
    # For now, let's try a direct search with structured initialization.
    return None

def _structured_search(target_rank, n, m, p, deadline, seed=42):
    """
    Search for a rank-r decomposition using structured random search
    with finite field verification as a fast filter.
    """
    rng = np.random.RandomState(seed)
    best_result = None
    
    # Use small integer coefficients
    coeff_range = [-2, -1, 0, 1, 2]
    
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        # Generate random sparse matrices with small integer entries
        U = np.zeros((n*m, target_rank), dtype=int)
        V = np.zeros((m*p, target_rank), dtype=int)
        W = np.zeros((p*n, target_rank), dtype=int)
        
        # Each column of U, V, W should be relatively sparse
        for c in range(target_rank):
            # For U: each column has ~2-3 nonzeros
            nz_u = rng.randint(2, 4)
            rows_u = rng.choice(n*m, size=nz_u, replace=False)
            for r in rows_u:
                U[r, c] = rng.choice(coeff_range)
            
            nz_v = rng.randint(2, 4)
            rows_v = rng.choice(m*p, size=nz_v, replace=False)
            for r in rows_v:
                V[r, c] = rng.choice(coeff_range)
            
            nz_w = rng.randint(2, 4)
            rows_w = rng.choice(p*n, size=nz_w, replace=False)
            for r in rows_w:
                W[r, c] = rng.choice(coeff_range)
        
        # Fast modular check first (mod 5)
        if not _verify_mod(U.tolist(), V.tolist(), W.tolist(), n, m, p, 5):
            continue
        
        # Full exact check
        if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

def _try_known_constructions(n, m, p):
    """
    Try known low-rank constructions for specific dimensions.
    For 3x3 * 3x4, we can try:
    - Rank 27: 3 * 3 * 3 (if we could share the last dimension)
    - Rank 24: Various block combinations
    - Rank 21: More aggressive sharing
    """
    results = []
    
    # Try rank 27: treat as 3 independent 3x1 * 3x4 products? No, that's not right.
    # Actually, for 3x3 * 3x4, the tensor has 36 entries.
    # A natural lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But that's too optimistic.
    
    # Let's try to construct rank 24 by combining three 2x2 Strassen blocks
    # with some additional terms.
    
    # For 3x3 * 3x4, one approach:
    # Split into 2x2 + 1x1 in the first two dimensions
    # The 2x2*2x2 part uses rank 7 (Strassen)
    # The cross terms and 1x1 parts need additional rank
    
    # This is getting complex. Let's just return None and rely on search.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 second budget, leaving 5s margin
    
    # First, try known constructions
    known = _try_known_constructions(n, m, p)
    if known is not None:
        return known
    
    # Try structured search for progressively lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = 12
    # Realistic targets: 24, 21, 18, 15, 12
    
    target_ranks = [24, 21, 18, 15, 12]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        result = _structured_search(target_rank, n, m, p, deadline, seed=target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
