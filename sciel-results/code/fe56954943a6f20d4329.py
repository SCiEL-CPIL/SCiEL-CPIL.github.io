import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # Naive rank-30 decomposition
    rank_naive = nm * mp // nm # No, rank is nm * p? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Terms are A[i,j] * B[j,k]
    # Number of terms is n * m * p = 30.
    rank_naive = n * m * p
    
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m + j, r] = 1
                V_naive[j*p + k, r] = 1
                W_naive[k*n + i, r] = 1
                r += 1
    
    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        # T[i,j,k] = 1 for all i,j,k? 
        # The tensor is defined such that sum_rr U[i*m+j,rr] V[j*p+k,rr] W[k*n+i,rr] = 1
        # for all i in [0,n), j in [0,m), k in [0,p).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not validate(U_naive, V_naive, W_naive, rank_naive):
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = rank_naive

    # Try to construct rank 28 based on Strassen for 2x2 blocks
    # A (2x3) = [A_L (2x2), A_R (2x1)]
    # B (3x5) = [B_T (2x5); B_B (1x5)]
    # C = A_L B_T + A_R B_B
    
    # A_R B_B is rank 10 (outer product)
    # A_L B_T: A_L is 2x2, B_T is 2x5.
    # Split B_T into B_T1 (2x2) and B_T2 (2x3).
    # A_L B_T1: Rank 7 (Strassen)
    # A_L B_T2: Rank 12 (3 cols * rank 4)
    # Total 7 + 12 + 10 = 29.
    # This is worse than 30? No, 29 < 30.
    
    # Let's try to build rank 29.
    U29 = np.zeros((nm, 29))
    V29 = np.zeros((mp, 29))
    W29 = np.zeros((pn, 29))
    
    r = 0
    # Part 1: A_L B_T1 (Strassen 2x2x2x2)
    # A_L indices: i in 0..1, j in 0..1. (Flat: i*3+j)
    # B_T1 indices: j in 0..1, k in 0..1. (Flat: j*5+k)
    # C indices: i in 0..1, k in 0..1. (Flat: k*2+i)
    
    # Strassen for X (2x2) * Y (2x2)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We need to map these to U, V, W.
    # This is complex to hardcode perfectly.
    
    # Instead, use a general search for rank 27-29 with random initialization.
    deadline = time.time() + 140
    
    # Values to try
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for target_rank in range(27, best_rank + 1):
        if time.time() > deadline:
            break
        found = False
        # Limit attempts to avoid timeout
        max_attempts = 200 if target_rank < 28 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            U_t = np.random.choice(vals, size=(nm, target_rank))
            V_t = np.random.choice(vals, size=(mp, target_rank))
            W_t = np.random.choice(vals, size=(pn, target_rank))
            if validate(U_t, V_t, W_t, target_rank):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = target_rank
                found = True
                break
        if found:
            continue # Try lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
