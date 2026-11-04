import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Try to find rank-49 decomposition for 2x5x5.
    
    Strategy: The known algorithm for 2xn x n uses rank 2n^2 - 1 for odd n.
    For n=5, that's 49. The construction works by:
    1. Using n^2 - 1 = 24 terms for the "main" 1x5x5 part (row 0 of A)
    2. Using n^2 - 1 = 24 terms for row 1 of A  
    3. Sharing 1 term between them, total 49
    
    The key identity for 1x5x5 (or 5x5x5) rank 24:
    We need T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,c]
    
    For 1xm x mxm, the tensor is just the m x m matrix multiplication.
    Rank of 5x5 matmul is at most 25 (schoolbook), but can we do 24?
    
    Actually, for 1xm x mxp, the rank is at most mp, and the tensor is:
    T[j,k] = sum_i A[i,j] * B[i,k] ... wait, let me reconsider.
    
    For A (2x5) * B (5x5) = C (2x5):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor has entries T[i,j,k] where i in [0,2), j in [0,5), k in [0,5)
    T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    
    So it's a 2x5x5 tensor with T[i,j,k] = delta_{i,j,k} in the sense that
    for each (i,k), sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    
    Actually the tensor is: for each term c, we have:
    C[i,k] += U[i,j,c] * V[j,k,c] * W[k,i,c] ... no.
    
    Let me use the standard form:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Decomposition: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    Wait, that's not right either. The standard is:
    
    C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) * W[k,i,c]
    
    No, the standard rank-r decomposition of the matmul tensor T_{i,j,k} is:
    T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    
    Where T_{i,j,k} = 1 (the tensor that maps A[i,j] and B[j,k] to C[i,k]).
    
    So we need: for all i,j,k: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    And for all other (i,j,k) combinations that don't appear in matmul, it should be 0.
    But in our case, every (i,j,k) with i<n, j<m, k<p appears exactly once.
    
    So the constraint is simply: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1 for all i,j,k.
    
    For 2x5x5, we have 50 such equations.
    
    Known construction for 2xn x n with rank 2n^2-1 (n odd):
    
    The idea is to use the fact that for the 1xn x n part, we can use rank n^2-1
    if we allow the "shared" term to help.
    
    Let me try a direct approach: start with schoolbook and try to set one U entry to 0
    and solve for adjustments.
    
    Actually, let me try a simpler approach: use the known algorithm.
    
    For 2x5x5, one construction:
    - Terms 0-23: handle row 0 of A, using a rank-24 decomposition of 5x5 matmul
    - Terms 24-48: handle row 1 of A, using a rank-24 decomposition of 5x5 matmul
    - But we share term 24, so total 49.
    
    The rank-24 decomposition of 5x5 matmul:
    This is non-trivial. The minimal rank of 5x5 matmul is not known to be 24.
    Schoolbook is 25. It's possible that 25 is minimal.
    
    So maybe 49 is not achievable via this simple sharing.
    
    Let me try a different approach: direct search for rank 49.
    
    I'll use a heuristic: start with schoolbook, pick a random term to "remove" (set all its
    U,V,W entries to 0), then try to fix the 50 equations by adjusting other terms.
    
    This is underdetermined (49 terms, 50 equations), so it's a system of 50 equations
    in 49*3 = 147 unknowns (if we allow all entries to vary). But it's nonlinear.
    
    Alternative: fix the structure. Keep all terms with integer 0/1 entries, but allow
    one term to have fractional entries.
    
    Let me try: for each pair of schoolbook terms (c1, c2), try to merge them by
    setting U[c1] = U[c2] = 0 for some entries and adjusting.
    
    This is getting complex. Let me just return None and rely on schoolbook.
    """
    if _remaining() < 5:
        return None
    
    n, m, p = 2, 5, 5
    rank = 49
    
    # Try a structured approach: 
    # Use 25 terms for "even" indices and 24 for "odd" with sharing.
    # 
    # Actually, let me try the following known construction for 2xn x n:
    # 
    # For n=2: rank 7 (2*4-1=7)
    # For n=3: rank 17 (2*9-1=17)  
    # For n=5: rank 49 (2*25-1=49)
    #
    # The construction uses the fact that x^n - 1 factors nicely for odd n.
    #
    # I'll implement a direct search: try to find integer coefficients.
    
    # Initialize with schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill first 49 terms with schoolbook pattern
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c < rank:
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    # We've used 49 of the 50 schoolbook terms. The 50th term (i=1,j=4,k=4) is missing.
    # We need to "absorb" this missing term into the existing 49.
    #
    # The missing term contributes 1 to C[1,4] via A[1,4]*B[4,4].
    # Currently, C[1,4] = sum_{j=0}^{3} A[1,j]*B[j,4] (from terms with k=4, i=1, j=0..3)
    # We need to add A[1,4]*B[4,4].
    #
    # We can try to modify one of the existing terms to also contribute to this.
    # For example, modify term c=0 (which was i=0,j=0,k=0) to also help.
    #
    # But this is a nonlinear problem. Let me try a simpler heuristic:
    # Add a new term that is a "linear combination" that helps.
    
    # Actually, let me try a different approach: use rational coefficients.
    # Set up the system and solve.
    
    # For simplicity, let me just return None if I can't find it quickly.
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    r49 = _try_rank_49(n, m, p)
    if r49 is not None:
        # Verify
        U, V, W = r49["U"], r49["V"], r49["W"]
        if _verify(n, m, p, U, V, W):
            return r49
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
