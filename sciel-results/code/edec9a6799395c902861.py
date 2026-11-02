"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct using the tensor convention.
    
    The multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    """
    # Check that each output element gets exactly 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that all other entries are 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if not (j == jj and k == kk and i == ii):
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _build_strassen_block(U, V, W, block_idx, p, n, m, col_offset):
    """Build one Strassen 2x2x2 block starting at column block_idx (each block covers 2 cols)."""
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
    
    base_col = 2 * block_idx
    for s in range(7):
        col = col_offset + s
        for (ai, aj), coeff in products[s][0]:
            U[ai*m + aj, col] = coeff
        for (bj, bk), coeff in products[s][1]:
            V[bj*p + base_col + bk, col] = coeff
        for (ci, ck), coeff in output_contribs[s]:
            W[(base_col + ck)*n + ci, col] = coeff

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses 6 independent Strassen 2x2x2 blocks, one per pair of output columns.
    Total rank = 6 * 7 = 42.
    
    This is believed to be optimal: the bilinear complexity of (2,2,p) matrix
    multiplication for even p is 7p/2, achieved by tiling p/2 Strassen blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    if time.time() > _DEADLINE:
        # Emergency: return a valid (but possibly non-optimal) decomposition
        # Fall back to schoolbook (rank 48) which is always correct
        rank = n * m * p  # 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-42 Strassen decomposition
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(6):
        _build_strassen_block(U, V, W, t, p, n, m, col_offset=7*t)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails (should not happen), fall back to schoolbook
    rank = n * m * p  # 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
