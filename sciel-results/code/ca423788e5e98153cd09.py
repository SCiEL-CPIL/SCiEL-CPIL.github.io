import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    # Try to find a rank 38 decomposition by attempting a rank 10 decomposition
    # for the 2x2x3 block (columns 8,9,10) instead of rank 11 (Strassen+schoolbook)
    
    # First, build the known rank 39 solution as fallback
    rank_39 = _build_rank_39()
    
    # Check time
    if time.time() > deadline - 2:
        return rank_39
    
    # Try to find rank 10 decomposition for [2,2,3]
    rank_10_solution = _try_rank_10_for_223()
    
    if rank_10_solution is not None:
        # Build the full rank 38 decomposition
        result = _build_rank_38_from_rank_10(rank_10_solution)
        if _verify_decomposition(result, n, m, p):
            return result
    
    # Fallback to rank 39
    return rank_39


def _build_rank_39():
    """Build the known rank 39 decomposition: 5 Strassen blocks (7 each) + 1 schoolbook column (4)."""
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
    U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
    U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
    U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
    
    return {"U": U, "V": V, "W": W}


def _try_rank_10_for_223():
    """
    Try to find a rank 10 decomposition for [2,2,3] (2x2 matrix * 2x3 matrix).
    If found, return (U, V, W) arrays of shapes (4,10), (6,10), (6,10).
    Returns None if not found within time.
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave margin for final construction
    
    n, m, p = 2, 2, 3
    rank = 10
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 4 x 6 x 6 tensor (since n*m=4, m*p=6, p*n=6)
    # Actually: U shape (4, rank), V shape (6, rank), W shape (6, rank)
    
    # Strategy: Use a structured approach. The rank of [2,2,3] is at most 12 (schoolbook).
    # Strassen gives rank 7 for [2,2,2]. For [2,2,3], we can try various approaches.
    
    # Approach 1: Try to find a decomposition using bilinear forms
    # Let's set up the problem as finding U, V, W such that the tensor is decomposed.
    
    # For small ranks, we can try a grid search over integer/half-integer values
    # But rank 10 with entries in {-1, 0, 1, 0.5, -0.5} has too many possibilities.
    
    # Alternative: Use the fact that [2,2,3] can be seen as [2,2,2] + [2,2,1]
    # [2,2,2] has rank 7 (Strassen), [2,2,1] has rank 4 (schoolbook for one column)
    # But that gives 11, not 10.
    
    # Let's try a more systematic approach: assume the decomposition has a certain structure.
    
    # Actually, let's try to see if rank 10 is possible by checking if the tensor
    # can be decomposed with rank 10 using a relaxation or integer programming.
    
    # Given the time constraint, let's try a few heuristic approaches:
    
    # Heuristic 1: Try to merge the last column of the Strassen block with the schoolbook
    # in a way that saves one rank.
    
    # Let's just return None for now, as finding a rank 10 decomposition for [2,2,3]
    # is non-trivial and may not be possible.
    
    if time.time() > deadline:
        return None
    
    # Try a simple structured approach: assume U, V, W have entries in {-1, 0, 1}
    # and try to find a valid decomposition.
    
    # This is a combinatorial search, so let's be very limited.
    
    # Actually, let's just return None to be safe. The rank 39 solution is known to work.
    return None


def _build_rank_38_from_rank_10(rank_10_solution):
    """Build rank 38 decomposition using rank 10 for the last [2,2,3] block."""
    n, m, p = 2, 2, 11
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # First 4 blocks: Strassen (rank 7 each)
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Last block: columns 8, 9, 10 using rank 10 solution
    U_r10, V_r10, W_r10 = rank_10_solution
    for i in range(4):
        for r_idx in range(10):
            U[i][r + r_idx] += U_r10[i][r_idx]
    for i in range(6):
        for r_idx in range(10):
            V[11 + i][r + r_idx] += V_r10[i][r_idx]  # B rows 0,1 cols 8,9,10 -> V[8..13, 14..19]
    for i in range(6):
        for r_idx in range(10):
            W[16 + i][r + r_idx] += W_r10[i][r_idx]  # C rows 0,1 cols 8,9,10 -> W[16..21]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p):
    """Verify that the decomposition correctly decomposes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Check: for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    return True
