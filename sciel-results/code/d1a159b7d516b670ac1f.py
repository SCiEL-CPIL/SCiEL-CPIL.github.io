"""
Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication.
Strategy: Start with schoolbook (rank 32), then search for better decompositions
using structured patterns based on Strassen-like block decompositions.
"""
import time
import random
from fractions import Fraction

N, M, P = 2, 2, 8
NM, MP, PN = N * M, M * P, P * N  # 4, 16, 16
SCHOOBOOK_RANK = N * M * P  # 32


def verify(U, V, W, n=N, m=M, p=P):
    """Verify exact tensor decomposition using integer arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    # Check the 32 required entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # Check all zero entries
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W


def try_strassen_column(k):
    """
    For a given column k (0..7), build the Strassen decomposition
    for computing A * B[:, k] where A is 2x2 and B[:,k] is 2x1.
    Returns (U_part, V_part, W_part) each as list of 7 entries.
    U_part: 4 entries (coefficients on A)
    V_part: 16 entries (coefficients on B)  
    W_part: 16 entries (coefficients on C)
    """
    # Strassen for 2x2 * 2x1 (B has only column k, so b01=b11=0 effectively)
    # Actually for 2x2 * 2x1:
    # C0 = a00*b00 + a01*b10
    # C1 = a10*b00 + a11*b10
    # This is just 4 multiplications, or we can use 3:
    # M1 = a00*b00
    # M2 = a10*b00  
    # M3 = (a00+a10-a11)*b10 ... no, simpler:
    # Actually 2x2*2x1 has rank 4 (it's just 4 independent products)
    # Or can we do it in 3? C0 = a00*b0 + a01*b1, C1 = a10*b0 + a11*b1
    # This is a 2x2*2x1 = rank 4 tensor (no known rank 3)
    # So for each column, we need rank 4, total 4*8=32 = schoolbook.
    # This confirms schoolbook is optimal for this simple approach.
    pass


def random_search(target_rank, n_trials, rng, deadline):
    """Random search for a decomposition of given rank."""
    vals = [0, 0, 0, 1, -1]  # Keep it simple with integers first
    for trial in range(n_trials):
        if time.time() > deadline:
            return None
        U = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(NM)]
        V = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(MP)]
        W = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(PN)]
        if verify(U, V, W):
            return (U, V, W)
    return None


def structured_search(deadline):
    """
    Try structured decompositions that exploit the 2x2 * 2x8 structure.
    Key insight: C[i,k] = a00*b0k + a01*b1k for i=0
                  C[i,k] = a10*b0k + a11*b1k for i=1
    where b0k = B[0,k], b1k = B[1,k]
    
    So C[0,k] = a00*B[0,k] + a01*B[1,k] for all k
    And C[1,k] = a10*B[0,k] + a11*B[1,k] for all k
    
    We can factor this as:
    C[0,:] = a00 * B[0,:] + a01 * B[1,:]
    C[1,:] = a10 * B[0,:] + a11 * B[1,:]
    
    This suggests using 4 'basis' patterns for B and 4 for A,
    but with the constraint that the same B pattern is used for all 8 columns.
    """
    # The structure is: for each (i, j, k), T[i*2+j, j*8+k, k*2+i] = 1
    # We want to find U (4xR), V (16xR), W (16xR) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Try rank 16-31 with structured patterns
    # Pattern: U has only 4 nonzero rows effectively (it's 4xR)
    # V has structure: V[j*8+k, r] depends on j and k
    # W has structure: W[k*2+i, r] depends on k and i
    
    # For a given rank R, try:
    # U[i*2+j, r] = u_{i,j,r}
    # V[j*8+k, r] = v_{j,k,r}  
    # W[k*2+i, r] = w_{k,i,r}
    
    # The constraint: for all i,j,k:
    # sum_r u_{i,j,r} * v_{j,k,r} * w_{k,i,r} = 1
    # And for invalid (a,b,c): sum = 0
    
    # This is still a hard problem. Let's try small structured searches.
    
    best = None
    rng = random.Random(12345)
    
    # Try ranks from 16 to 31
    for R in range(16, 32):
        if time.time() > deadline:
            break
        
        # Number of trials based on remaining time
        remaining = deadline - time.time()
        n_trials = min(500, max(10, int(remaining * 50)))
        
        result = random_search(R, n_trials, rng, deadline)
        if result is not None:
            best = result
            break  # Found one, try to go lower next iteration won't happen since we break
    
    return best


def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U0, V0, W0 = schoolbook()
    if verify(U0, V0, W0):
        best = (U0, V0, W0)
        best_rank = SCHOOBOOK_RANK
    else:
        best = None
        best_rank = SCHOOBOOK_RANK
    
    # Attempt to find a better decomposition
    if time.time() < deadline:
        result = structured_search(deadline)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best = result
                best_rank = len(U[0])
    
    if best is not None:
        return {"U": best[0], "V": best[1], "W": best[2]}
    else:
        return {"U": U0, "V": V0, "W": W0}
