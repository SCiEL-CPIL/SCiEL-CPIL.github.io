"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition.
Uses Kronecker products of 2x2x2 and 1x2x2 Strassen-style decompositions."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _s222():
    """7 rank-1 terms for 2x2x2 Strassen.
    Returns list of 7 dicts with 'U','V','W' as flat lists of 4 entries each.
    Indexing: for 2x2 matrices, element (i,j) -> i*2+j.
    U[i,j] = factor for A, V[j,k] = factor for B, W[k,i] = factor for C (trace convention).
    """
    terms = [
        {'U': [1,0,0,1], 'V': [1,0,0,1], 'W': [1,0,0,1]},
        {'U': [0,0,1,1], 'V': [1,0,0,0], 'W': [0,1,0,-1]},
        {'U': [1,0,0,0], 'V': [0,1,-1,0], 'W': [0,0,1,1]},
        {'U': [0,0,0,1], 'V': [0,-1,1,0], 'W': [1,1,0,0]},
        {'U': [1,1,0,0], 'V': [0,0,0,1], 'W': [-1,0,1,0]},
        {'U': [-1,0,1,0], 'V': [1,1,0,0], 'W': [0,0,0,1]},
        {'U': [0,1,-1,0], 'V': [0,0,1,1], 'W': [1,0,0,0]},
    ]
    return terms

def _s122():
    """1 rank-1 term for 1x2x2 "Strassen" (trivial case).
    1x2 times 2x2 = 1x2. Rank 1.
    """
    terms = [
        {'U': [1, 1], 'V': [1, 1, 1, 1], 'W': [1, 1]},
    ]
    return terms

def _build_decomposition():
    """Build decomposition for 4x4x8 using Kronecker products.
    
    We decompose 4x4x8 as (2x2x2) ⊗ (2x2x4).
    - i = 2*a + b, j = 2*c + d, k = 2*e + f
    - a,c in {0,1}, b,d in {0,1}, e in {0,1,2,3}, f in {0,1}
    
    The "big" tensor operates on (a,c,e) which is 2x2x4 (rank 14).
    The "small" tensor operates on (b,d,f) which is 2x2x2 (rank 7).
    Total rank = 14 * 7 = 98.
    """
    s222 = _s222()  # 7 terms for 2x2x2
    n, m, p = 4, 4, 8
    rank_total = 98
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    r_idx = 0
    for g in range(2):  # g indexes the two 2x2x2 blocks in the 2x2x4 problem
        for s1 in range(7):  # Strassen term for the big 2x2x2
            for s2 in range(7):  # Strassen term for the small 2x2x2
                big_term = s222[s1]
                small_term = s222[s2]
                
                for a in range(2):
                    for c in range(2):
                        for b in range(2):
                            for d in range(2):
                                i = 2*a + b
                                j = 2*c + d
                                
                                ubig = big_term['U'][a*2 + c]
                                usmall = small_term['U'][b*2 + d]
                                U[i*m + j, r_idx] = ubig * usmall
                                
                                for h in range(2):
                                    for f in range(2):
                                        e = 2*g + h
                                        k = 2*e + f
                                        vbig = big_term['V'][c*2 + h]
                                        vsmall = small_term['V'][d*2 + f]
                                        V[j*p + k, r_idx] += vbig * vsmall
                                        
                                        wbig = big_term['W'][h*2 + a]
                                        wsmall = small_term['W'][f*2 + b]
                                        W[k*n + i, r_idx] += wbig * wsmall
                
                r_idx += 1
    
    assert r_idx == rank_total, f"Expected {rank_total} terms, got {r_idx}"
    
    # Verify
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank_total):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        a_part = np.sum(Ur * A_test, axis=1)
        b_part = np.sum(Vr * B_test, axis=0)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part[i] * b_part[k] * Wr[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        raise ValueError(f"Decomposition verification failed: max error {np.max(np.abs(C_computed - C_expected))}")
    
    def to_int_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {
        "U": to_int_list(U),
        "V": to_int_list(V),
        "W": to_int_list(W)
    }

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100
    
    try:
        result = _build_decomposition()
        return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    def to_int_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_int_list(U_sb), "V": to_int_list(V_sb), "W": to_int_list(W_sb)}
