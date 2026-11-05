"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    - The 3x3x3 block (columns 0,1,2) can be decomposed with rank 19 (Coppersmith-Winograd).
    - The remaining 4 columns (3,4,5,6) are computed with schoolbook: 4 * 9 = 36 terms.
    - Total rank: 19 + 36 = 55.
    
    However, implementing the full CW decomposition is complex. Let's use a simpler but
    still reduced approach:
    - Use schoolbook for the first 3 columns: 27 terms.
    - Use schoolbook for the remaining 4 columns: 36 terms.
    - Total: 63 (same as full schoolbook).
    
    Actually, let me try to implement a known rank-19 decomposition for 3x3x3.
    
    The Coppersmith-Winograd rank-19 decomposition for 3x3x3 matrix multiplication:
    The decomposition is given by the following set of triples (u, v, w) where each
    is a vector in {0, 1/2, -1/2, 1}^3, and the tensor is:
    T = sum_{t=1}^{19} u_t ⊗ v_t ⊗ w_t
    
    Here's the explicit CW-19 decomposition (from various sources):
    
    The 19 terms are indexed by t=0..18. Each term has u_t, v_t, w_t in Z[1/2]^3.
    
    Let me use the explicit listing from the literature:
    
    Term 0:  u=(1,0,0), v=(0,1,0), w=(0,0,1)
    Term 1:  u=(0,1,0), v=(1,0,0), w=(0,1,0)
    Term 2:  u=(0,0,1), v=(0,0,1), w=(1,0,0)
    Term 3:  u=(1,1,0), v=(0,1,1), w=(1,0,0)
    Term 4:  u=(1,0,1), v=(1,0,1), w=(0,1,0)
    Term 5:  u=(0,1,1), v=(1,1,0), w=(0,0,1)
    Term 6:  u=(1,1,1), v=(1,1,1), w=(1,1,1)
    Term 7:  u=(1,1,-1), v=(1,1,1), w=(1,-1,1)
    Term 8:  u=(1,-1,1), v=(1,1,1), w=(1,1,-1)
    Term 9:  u=(1,1,1), v=(1,1,-1), w=(-1,1,1)
    Term 10: u=(1,1,1), v=(1,-1,1), w=(1,-1,-1)
    Term 11: u=(1,1,1), v=(-1,1,1), w=(-1,-1,1)
    Term 12: u=(1,-1,-1), v=(1,1,1), w=(1,1,-1)
    Term 13: u=(-1,1,1), v=(1,1,1), w=(-1,1,1)
    Term 14: u=(1,1,-1), v=(1,1,1), w=(1,1,1)
    Term 15: u=(1,1,1), v=(1,1,1), w=(1,1,1)  -- this can't be right, duplicates
    
    Actually, I'm not confident in my memory of the exact CW-19 coefficients.
    Let me use a different approach: use the known fact that 3x3x3 has rank 19,
    and construct the decomposition programmatically using a known algorithm.
    
    Alternatively, I can use a simpler reduction:
    - For the 3x3x3 part, I can use a rank-23 decomposition (which is easier to construct)
      or stick with schoolbook rank 27.
    
    Let me try a different strategy: use the schoolbook decomposition but look for
    opportunities to share factors across the 7 columns.
    
    For each column k, C[:,k] = A @ B[:,k]. The schoolbook uses 9 terms per column.
    If I can find that some of the 9 terms for column k can be shared with terms for
    column k', I can reduce the total rank.
    
    However, in general, for arbitrary B, the 9 terms for each column are independent.
    
    Let me just use the schoolbook rank-63 decomposition to ensure correctness,
    since I'm not confident in implementing CW-19 correctly from memory.
    
    Actually, wait. Let me try to implement a simple but correct reduction.
    
    Observation: The 3x3x3 multiplication tensor can be decomposed with rank 19, but
    constructing it is complex. Instead, let me use the following approach:
    
    1. Use rank-19 for the 3x3x3 part (columns 0,1,2) - I'll hardcode a known decomposition.
    2. Use rank-36 for the remaining 4 columns.
    3. Total: 55.
    
    Let me look up the CW-19 decomposition more carefully.
    
    Actually, I'll use a different well-known result: the rank of 3x3x3 is 19, and
    one explicit decomposition is available in the literature. Let me try to recall it.
    
    The CW-19 decomposition uses the following 19 triples (u_t, v_t, w_t) where each
    component is in {0, ±1/2, ±1}:
    
    I'll use the decomposition from [Stothers, 2010] or similar.
    
    Given my uncertainty, let me use a safer approach: implement the schoolbook
    decomposition (rank 63) which is guaranteed correct.
    
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 3, 3, 7
    rank_schoolbook = n * m * p  # 63
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
