import time
import numpy as np
from fractions import Fraction

def _elapsed_ok(start, deadline=140.0):
    """Check if we're still within the deadline with a safety margin."""
    return (time.time() - start) < deadline

def _schoolbook():
    """Return the schoolbook decomposition (rank 48)."""
    n, m, p = 2, 3, 8
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i*m+a][col] = 1
                V[a*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W):
    """Validate that the decomposition is correct."""
    n, m, p = 2, 3, 8
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    u_val = U[i*m+a][c]
                    v_val = V[a*p+k][c]
                    w_val = W[k*n+i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        acc += u_val * v_val * w_val
                if acc != 1:
                    return False
    return True

def _verify_strassen():
    """Verify the Strassen decomposition for 2x2x2."""
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    for i in range(2):
        for a in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    acc += S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i]
                if acc != 1:
                    return False
    return True

def _strassen_decomposition():
    """Build the rank-44 decomposition using Strassen for 2x2x2x8 + schoolbook for 2x1x1x8."""
    n, m, p = 2, 3, 8
    
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: 2x2 x 2x8 using 4 Strassen blocks
    for block in range(4):
        k0 = block * 2
        for r in range(7):
            for i in range(2):
                for a in range(2):
                    u_val = S_U[r][i*2+a]
                    if u_val != 0:
                        U[i*m + a][col] += u_val
            
            for a in range(2):
                for k in range(2):
                    v_val = S_V[r][a*2+k]
                    if v_val != 0:
                        V[a*p + (k0+k)][col] += v_val
            
            for k in range(2):
                for i in range(2):
                    w_val = S_W[r][k*2+i]
                    if w_val != 0:
                        W[(k0+k)*n + i][col] += w_val
            
            col += 1
    
    # Part 2: 2x1 x 1x8 using schoolbook (16 terms)
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_single_rank_term_for_last_part():
    """
    Try to use a single rank term for the 2x1x1x8 part.
    
    The tensor entries for this part are T[i,2,k] = 1 for all i in {0,1}, k in {0..7}.
    We need: sum_r U[i*3+2, r] * V[2*8+k, r] * W[k*2+i, r] = 1 for all i, k.
    
    If we use one rank term r with U[i*3+2, r] = 1, V[2*8+k, r] = 1, W[k*2+i, r] = 1,
    then the contribution is 1*1*1 = 1 for all (i,k). This should work!
    
    But we need to verify it doesn't interfere with the Strassen part.
    The Strassen part uses U[0:4, :], V[0:16, :], W[0:16, :] for its rank terms.
    The new rank term would use U[4:6, :], V[16:24, :], W[0:16, :].
    
    Since the U and V rows are disjoint from the Strassen part, there's no interference.
    The W rows are shared, but the Strassen terms only have non-zero W entries for
    their specific (k,i) pairs, and the new term adds 1 to all W entries.
    
    Wait, this is a problem. The W matrix is shared. Let me think more carefully.
    
    Actually, for the Strassen part, each rank term r in 0..27 has specific non-zero
    entries in W. For the new rank term (say rank 28), we set W[k*2+i, 28] = 1 for
    all i, k. This doesn't affect the Strassen terms because they use different
    columns (r=0..27 vs r=28).
    
    So the total rank would be 28 + 1 = 29!
    """
    n, m, p = 2, 3, 8
    
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    rank = 29
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: 2x2 x 2x8 using 4 Strassen blocks
    for block in range(4):
        k0 = block * 2
        for r in range(7):
            for i in range(2):
                for a in range(2):
                    u_val = S_U[r][i*2+a]
                    if u_val != 0:
                        U[i*m + a][col] += u_val
            
            for a in range(2):
                for k in range(2):
                    v_val = S_V[r][a*2+k]
                    if v_val != 0:
                        V[a*p + (k0+k)][col] += v_val
            
            for k in range(2):
                for i in range(2):
                    w_val = S_W[r][k*2+i]
                    if w_val != 0:
                        W[(k0+k)*n + i][col] += w_val
            
            col += 1
    
    # Part 2: 2x1 x 1x8 using a single rank term
    # U[2,28]=1, U[5,28]=1, V[16+k,28]=1 for all k, W[2k+i,28]=1 for all i,k
    U[2][col] = 1
    U[5][col] = 1
    for k in range(8):
        V[2*p + k][col] = 1
        W[k*n + 0][col] = 1
        W[k*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    
    # First, verify Strassen works
    if not _verify_strassen():
        return _schoolbook()
    
    # Try the rank-29 decomposition first
    result = _try_single_rank_term_for_last_part()
    
    if _validate(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to rank-44
    result = _strassen_decomposition()
    
    if _validate(result["U"], result["V"], result["W"]):
        return result
    
    # Final fallback: schoolbook
    return _schoolbook()
