# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is a dict with keys 'U', 'V', 'W' mapping (i,j) -> coeff.
    """
    terms = []
    
    # Term 1: m1 = (a+d)(e+h)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Term 2: m2 = (c+d)(e)
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # Term 3: m3 = a(f-h)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1}
    })
    
    # Term 4: m4 = d(g-e)
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # Term 5: m5 = (a+b)(h)
    # C00 gets -m5, C01 gets +m5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # Term 6: m6 = (c-a)(e+f)
    # C11 gets +m6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # Term 7: m7 = (b-d)(g+h)
    # C00 gets +m7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    return terms

def _construct_block_decomposition(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2 blocks.
    
    For n=2, m=4, p=6:
    Split A (2x4) into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    Split B (4x6) into six 2x2 blocks:
      B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
      B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
    
    C = A @ B
    C[0:2, 0:2] = A0 @ B[0:2,0:2] + A1 @ B[2:4,0:2]
    C[0:2, 2:4] = A0 @ B[0:2,2:4] + A1 @ B[2:4,2:4]
    C[0:2, 4:6] = A0 @ B[0:2,4:6] + A1 @ B[2:4,4:6]
    
    For each 2x2 @ 2x2 multiplication, use Strassen (rank 7)
    Total: 6 * 7 = 42
    """
    n, m, p = 2, 4, 6
    
    strassen_terms = _strassen_2x2_terms()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for a_block in range(2):
        for b_block in range(3):
            for t_idx, term in enumerate(strassen_terms):
                r = a_block * 21 + b_block * 7 + t_idx
                
                for (i, j_local), coeff in term['U'].items():
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global][r] += coeff
                
                for (j_local, k_local), coeff in term['V'].items():
                    j_global = a_block * 2 + j_local
                    k_global = b_block * 2 + k_local
                    V[j_global * p + k_global][r] += coeff
                
                for (i, k_local), coeff in term['W'].items():
                    k_global = b_block * 2 + k_local
                    W[k_global * n + i][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, max_samples=100):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i * m + j][r] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j * p + k][r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += alpha[i] * beta[k] * W[k * n + i][r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Use the block Strassen decomposition (rank 42)
    result = _construct_block_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, max_samples=100):
        if time.time() > deadline:
            return result
        return result
    
    # If verification fails, fall back to schoolbook (rank 48)
    result = _schoolbook_decomposition(n, m, p)
    return result

# EVOLVE-BLOCK-END
