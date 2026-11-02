"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using tensor structure."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check zeros for invalid combinations
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x3():
    """Known rank-5 decomposition for 2x2 * 2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 5
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
    ]
    
    for s in range(5):
        for (ai,aj), coeff in products[s][0]:
            U[ai*m+aj, s] = coeff
        for (bj,bk), coeff in products[s][1]:
            V[bj*p+bk, s] = coeff
        for (ci,ck), coeff in output_contribs[s]:
            W[ck*n+ci, s] = coeff
    
    return U, V, W

def _strassen_2x2x4():
    """Strassen-based rank-14 decomposition for 2x2 * 2x4."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(2):
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj, col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk, col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci, col] = coeff
            col += 1
    return U, V, W

def _find_decomposition(n, m, p, rank, max_iter=3000, seed=42):
    """Find a rank-r decomposition of the (n,m,p) matrix multiplication tensor via ALS."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    def get_target():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    target = get_target()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update U
        for a in range(n*m):
            i, j = a // m, a % m
            M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
            U[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
    
    return best_U, best_V, best_W, best_err

def _strassen_2x2x2_block(base_col, U, V, W, n, m, p):
    """Add one 2x2x2 Strassen block (rank 7) starting at column base_col."""
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    for s in range(7):
        col = base_col + s
        for (ai,aj), coeff in products[s][0]:
            U[ai*m+aj, col] = coeff
        for (bj,bk), coeff in products[s][1]:
            V[bj*p+base_col+bk, col] = coeff
        for (ci,ck), coeff in output_contribs[s]:
            W[(base_col+ck)*n+ci, col] = coeff

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy: 
    1. Try rank-20 via 4 groups of 2x2x3 (rank 5 each)
    2. Try rank-24 via 6 groups of 2x2x2 (rank 4 each) - if rank-4 exists for 2x2x2
    3. Try rank-28 via 4 groups of 2x2x3 (rank 7 each) - Strassen on pairs
    4. Fallback to rank-42 (Strassen on 6 pairs)
    """
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy 1: Rank-20 using 4 blocks of 2x2x3 (rank 5 each)
    U3, V3, W3 = _strassen_2x2x3()
    if _verify_decomposition_fast(U3, V3, W3, 2, 2, 3):
        rank = 20  # 4 * 5
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(4):
            base_col = 3*t
            for r in range(5):
                U[:, t*5+r] = U3[:, r]
                for j in range(2):
                    for k in range(3):
                        V[j*p + base_col + k, t*5+r] = V3[j*3+k, r]
                for k in range(3):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*5+r] = W3[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try to find rank-4 for 2x2x2, then tile 6 times for rank-24
    # This would be a breakthrough if it works
    found_4 = False
    U2 = V2 = W2 = None
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        U2_t, V2_t, W2_t, err = _find_decomposition(2, 2, 2, 4, max_iter=500, seed=seed)
        if err < 1e-6:
            if _verify_decomposition_fast(U2_t, V2_t, W2_t, 2, 2, 2):
                U2, V2, W2 = U2_t, V2_t, W2_t
                found_4 = True
                break
    
    if found_4:
        rank = 24  # 6 * 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(6):
            base_col = 2*t
            for r in range(4):
                U[:, t*4+r] = U2[:, r]
                for j in range(2):
                    for k in range(2):
                        V[j*p + base_col + k, t*4+r] = V2[j*2+k, r]
                for k in range(2):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*4+r] = W2[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 3: Rank-28 using 4 blocks of 2x2x3 with Strassen rank-7
    # Wait, 2x2x3 has rank 5, not 7. Let me use 2x2x2 blocks with rank 7 each
    # 12 = 6 * 2, so 6 blocks of 2x2x2, each rank 7, total 42. That's the fallback.
    
    # Actually, let me try a different tiling: 2x2x6 blocks
    # If rank(2x2x6) <= 14, then 2 blocks give rank 28
    found_14 = False
    U6 = V6 = W6 = None
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        U6_t, V6_t, W6_t, err = _find_decomposition(2, 2, 6, 14, max_iter=800, seed=seed)
        if err < 1e-6:
            if _verify_decomposition_fast(U6_t, V6_t, W6_t, 2, 2, 6):
                U6, V6, W6 = U6_t, V6_t, W6_t
                found_14 = True
                break
    
    if found_14:
        rank = 28  # 2 * 14
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(2):
            base_col = 6*t
            for r in range(14):
                U[:, t*14+r] = U6[:, r]
                for j in range(2):
                    for k in range(6):
                        V[j*p + base_col + k, t*14+r] = V6[j*6+k, r]
                for k in range(6):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*14+r] = W6[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-42 Strassen on 6 pairs
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(6):
        if time.time() > deadline:
            break
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj][col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk][col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
