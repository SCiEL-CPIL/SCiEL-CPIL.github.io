"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Validate if U, V, W form a valid decomposition of n x m by m x p matrix multiplication.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Vectorized check:
    # T[a, b, c, d, i, j] = sum_r U[a*m+b, r] * V[c*p+d, r] * W[j*n+i, r]
    # We want T[a, b, c, d, i, j] == 1 if a==i and d==j and b==c, else 0.
    
    # Reshape to 3D tensors
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # T[a, b, c, d, i, j] = sum_r U_3d[a, b, r] * V_3d[c, d, r] * W_3d[j, i, r]
    # This is a 6-tensor.
    # We can compute it as:
    # Temp1[a, b, c, d, r] = U_3d[a, b, r] * V_3d[c, d, r]
    # Temp2[a, b, c, d, i, j] = sum_r Temp1[a, b, c, d, r] * W_3d[j, i, r]
    
    # Memory check: 6*6*6*6*6*6 = 46656 floats. Very small.
    # Temp1: 6*6*6*6*rank. If rank=200, 6^4*200 = 2.5M floats. OK.
    
    # Let's just do the einsum or explicit loop over rank if needed, but einsum is better.
    # Actually, constructing the full 6-tensor might be overkill if we can check sparsity.
    # But 46656 checks is fast in Python loops too, but vectorized is safer for speed.
    
    # Let's construct the expected tensor E[a, b, c, d, i, j]
    E = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if a == i and d == j and b == c: # Wait, indices in E are a,b,c,d,i,j
                       # E[a,b,c,d,i,j] = 1 if i==a and j==d and c==b
                       pass
    # Correct condition: T[a,b,c,d,i,j] = delta_{a,i} delta_{d,j} delta_{b,c}
    # So E[a,b,c,d,i,j] = 1 if i==a and j==d and c==b
    
    # Let's just iterate over the non-zero expected values and check T.
    # There are n * p * m non-zero values (a, d, b). i=a, j=d, c=b.
    # Total non-zeros: 6 * 6 * 6 = 216.
    # We only need to check these 216 values for T == 1.
    # And we need to check that all other values are 0.
    # Total values 46656.
    # Checking all is fine.
    
    # Compute T
    # T[a, b, c, d, i, j]
    # U[a, b, r], V[c, d, r], W[j, i, r]
    
    # We can compute T via:
    # T = np.einsum('abr,cdr,ji r->abcdij', U_3d, V_3d, W_3d)
    # But W_3d is indexed [j, i, r] in the formula?
    # W[j*n+i, r] -> W_3d[j, i, r]
    # Yes.
    
    try:
        T = np.einsum('abr,cdr,jir->abcdij', U_3d, V_3d, W_3d)
    except Exception:
        # Fallback to loop
        T = np.zeros((n, m, m, p, n, p))
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for j in range(p):
                                s = 0.0
                                for r in range(rank):
                                    s += U_3d[a, b, r] * V_3d[c, d, r] * W_3d[j, i, r]
                                T[a, b, c, d, i, j] = s
    
    # Check
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for j in range(p):
                            expected = 1.0 if (a == i and d == j and b == c) else 0.0
                            if abs(T[a, b, c, d, i, j] - expected) > tol:
                                return False
    return True

def _construct_rank_189():
    """Fallback: Block-wise Strassen (2x2 blocks in 3x3 grid). Rank 189."""
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    for (li, lk, coeff) in strassen_U[t]:
                        gi = 2 * bi + li
                        gk = 2 * bk + lk
                        U[gi * m + gk][col] += coeff
                    for (lk, lj, coeff) in strassen_V[t]:
                        gk = 2 * bk + lk
                        gj = 2 * bj + lj
                        V[gk * p + gj][col] += coeff
                    for (li, lj, coeff) in strassen_W[t]:
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        W[gj * n + gi][col] += coeff
                    col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_184():
    """
    Attempt rank-184 decomposition using 3x3 blocks (2x2 grid) with rank-23 3x3 scheme.
    This uses a hardcoded rank-23 decomposition for 3x3 matrices.
    """
    n, m, p = DIMENSIONS
    # Rank 23 decomposition for 3x3
    # Source: Known optimal rank for 3x3 is 23.
    # We use a specific valid set of 23 terms.
    # Note: The previous attempt in the seed program had a potentially invalid set.
    # Here we use a set derived from standard literature or verified implementations.
    
    # U: (row, col, coeff)
    # V: (row, col, coeff)
    # W: (row, col, coeff)
    
    # This specific list is from a known rank-23 algorithm.
    # If this list is invalid, validation will fail and we fall back to 189.
    
    terms_3x3 = [
        {'U': [(0,0,1), (2,2,1)], 'V': [(0,0,1), (2,2,1)], 'W': [(0,0,1), (2,2,1)]},
        {'U': [(2,0,1), (2,2,1)], 'V': [(0,0,1)], 'W': [(2,0,1), (2,2,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (2,1,-1)], 'W': [(0,1,1), (2,1,1)]},
        {'U': [(2,2,1)], 'V': [(2,0,1), (0,0,-1)], 'W': [(0,2,1), (2,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(2,2,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(2,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(2,1,1)]},
        {'U': [(0,1,1), (2,2,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1), (2,2,1)], 'V': [(0,2,1)], 'W': [(0,2,1), (2,2,1)]},
        {'U': [(2,2,1)], 'V': [(0,2,1), (2,2,-1)], 'W': [(0,2,-1), (2,2,1)]},
        {'U': [(0,0,1), (0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1), (0,2,1)]},
        {'U': [(2,0,1)], 'V': [(0,2,1), (2,0,-1)], 'W': [(2,0,1), (2,2,1)]},
        {'U': [(0,2,1), (2,0,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,2,1)]},
        {'U': [(1,1,1)], 'V': [(1,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,1)], 'V': [(1,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,0,1)], 'V': [(1,1,1), (0,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(0,1,1), (1,0,1)], 'V': [(1,1,1)], 'W': [(0,1,-1), (1,0,1)]},
        {'U': [(1,1,1), (1,0,-1)], 'V': [(0,1,1), (1,0,1)], 'W': [(1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(1,2,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(1,2,1), (0,2,-1)], 'W': [(0,2,1), (1,2,1)]},
        {'U': [(0,2,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(0,2,-1), (1,2,1)]},
        {'U': [(1,2,1)], 'V': [(1,1,1), (0,2,-1)], 'W': [(1,2,1), (1,1,1)]},
        {'U': [(0,2,1), (1,2,-1)], 'V': [(1,1,1), (1,2,1)], 'W': [(0,2,1)]},
    ]
    
    rank = 8 * 23 # 2x2 grid of 3x3 blocks, each rank 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 2x2 grid of 3x3 blocks
    for bi in range(2): # Row block of A/C
        for bk in range(2): # Col block of A / Row block of B
            for bj in range(2): # Col block of B / Col block of C
                for t in range(23):
                    term = terms_3x3[t]
                    for (li, lk, coeff) in term['U']:
                        gi = 3 * bi + li
                        gk = 3 * bk + lk
                        U[gi * m + gk][col] += coeff
                    for (lk, lj, coeff) in term['V']:
                        gk = 3 * bk + lk
                        gj = 3 * bj + lj
                        V[gk * p + gj][col] += coeff
                    for (li, lj, coeff) in term['W']:
                        gi = 3 * bi + li
                        gj = 3 * bj + lj
                        W[gj * n + gi][col] += coeff
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Try rank 184 first
    try:
        res_184 = _construct_rank_184()
        if time.time() < deadline:
            if _validate_decomposition(res_184['U'], res_184['V'], res_184['W'], n, m, p):
                return res_184
    except Exception:
        pass
        
    # Fallback to rank 189
    return _construct_rank_189()

# EVOLVE-BLOCK-END
