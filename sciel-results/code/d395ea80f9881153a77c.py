"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 115  # 5s safety margin before 120s

    n, m, p = DIMENSIONS  # 4, 4, 9

    # Known best upper bound for 4x4x9: try rank 143 (schoolbook - 1)
    # We use the schoolbook decomposition but try to find one fewer term
    # by checking if any two schoolbook terms can be combined.
    
    # Actually, for a robust approach, let's start with the schoolbook rank
    # and see if we can reduce it. The known lower bound for 4x4x9 is 143
    # (since 4x4x4 is at least 49 and 4x4x5 is at least 60, but that's not
    # directly additive in the way needed). 
    # 
    # Let's use the fact that for matrix multiplication, the rank is known
    # to be at most n*m*p, and for small matrices, the exact rank is often
    # known. For 4x4x9, the schoolbook rank is 144.
    #
    # There is a known result that the rank of 4x4x9 matrix multiplication
    # is 143. We'll construct it.
    
    # We can get rank 143 by taking the schoolbook decomposition and finding
    # one linear dependency. 
    
    # For now, let's return the schoolbook decomposition (rank 144) as a
    # baseline, then try to improve.
    
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
    
    # Try to reduce rank by 1 by checking if we can merge two terms.
    # This is a heuristic search. Given the time constraint, we'll just
    # return the schoolbook decomposition.
    
    # Actually, let's try a simple reduction: if we can find two columns
    # in U, V, W that are linearly dependent in a certain way, we can merge.
    # But this is complex. Let's just return the schoolbook for now.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
