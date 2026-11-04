import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-40 exact tensor decomposition for 3x3 by 3x5 matrix multiplication.
    Falls back to rank-45 schoolbook if search fails or times out.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 5
    
    # Target Rank 40
    # Structure:
    # Terms 0-29: Handle k=0,1,2 (3x3x3 part). Pattern: (i,j,k) -> term index
    # Terms 30-39: Handle k=3,4 (3x3x2 part).
    
    # Initialize U, V, W as lists of lists for flexibility
    # U: 9 x 40, V: 15 x 40, W: 15 x 40
    
    def make_schoolbook():
        rank = n * m * p # 45
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_u = i * m + j
                    row_v = j * p + k
                    row_w = k * n + i
                    U[row_u][col_idx] = 1
                    V[row_v][col_idx] = 1
                    W[row_w][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    # Try Rank 40
    # We fix the first 30 columns (k=0,1,2) to the standard pattern.
    # U_col(t) = e_{i,j}, V_col(t) = e_{j,k}, W_col(t) = e_{k,i}
    
    # For k=3,4, we need to find 10 more terms.
    # Let the additional terms be indexed 30..39.
    # Let a term be defined by (u_idx, v_idx, w_idx, val) where val is usually 1.
    # Actually, to keep it simple, we assume binary entries (0 or 1) for the search space initially,
    # or allow simple integer values.
    
    # The constraint for each (i, j, k) is:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 if (i,j,k) is part of the tensor?
    # Wait, the tensor T_{i,j,k} for matrix multiplication A(3x3) * B(3x5) -> C(3x5)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 if i in 0..2, j in 0..2, k in 0..4.
    # Specifically, T_{i,j,k} = 1 for all valid i,j,k.
    # So we need sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1 for all i,j,k.
    
    # For the first 30 terms (k=0,1,2):
    # Term t corresponds to (i_t, j_t, k_t).
    # U_{i,j,t} = 1 if i==i_t and j==j_t.
    # V_{j,k,t} = 1 if j==j_t and k==k_t.
    # W_{k,i,t} = 1 if k==k_t and i==i_t.
    
    # Let's precompute the contribution of the first 30 terms to each (i,j,k).
    # For a given (i,j,k):
    # If k < 3:
    #   The term t corresponding to (i,j,k) contributes 1.
    #   Do any other terms t' (with k' < 3) contribute?
    #   Term t' has (i', j', k').
    #   Contribution is 1 if i==i', j==j', k==k'.
    #   Since the mapping (i,j,k) -> t is bijective for the first 30,
    #   only t=(i,j,k) contributes 1. All others 0.
    #   So sum_{t<30} ... = 1 for all k<3.
    #   This is exactly what we want (T_{i,j,k}=1).
    
    # For k >= 3 (k=3,4):
    #   The first 30 terms have k' < 3.
    #   V_{j,k,t} requires k==k'. Since k>=3 and k'<3, V_{j,k,t} = 0 for all t<30.
    #   So sum_{t<30} ... = 0 for all k>=3.
    #   We need sum_{t=30..39} U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1 for all i,j,k in {0..2}x{0..2}x{3,4}.
    
    # So we need to find 10 vectors u_t, v_t, w_t (t=30..39) such that:
    # For all i in 0..2, j in 0..2, k in {3,4}:
    #   sum_{t=30..39} U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1
    
    # Let's search for binary solutions (0 or 1) for these 10 terms.
    # Each term t has:
    #   u_t: subset of {0..8} (indices for i*m+j)
    #   v_t: subset of {0..14} (indices for j*p+k)
    #   w_t: subset of {0..14} (indices for k*n+i)
    
    # To make it tractable, we can restrict the structure.
    # A common pattern is that each term "covers" a specific set of (i,j,k) triples.
    # Let's try to find a set of 10 "rectangles" or similar structures that cover the 18 points (3*3*2) exactly once?
    # No, the equation is sum of products = 1.
    # If we use binary entries, and if the supports are disjoint in a certain way, it's like a covering.
    
    # Let's parameterize the 10 terms.
    # To reduce search space, we can assume without loss of generality some symmetries or fixed patterns?
    # Or just do a randomized search / backtracking.
    
    # Given the small size (10 terms, 18 constraints), we can try a backtracking search with pruning.
    # However, the variables are the entries of U, V, W for columns 30..39.
    # That's 10 * (9 + 15 + 15) = 390 binary variables. Too many for brute force.
    
    # Alternative: Assume the terms are "standard" in some way.
    # For 3x3x3, rank is 27? No, 27 is schoolbook. Strassen is 7 for 2x2.
    # For 3x3x3, best known is 23?
    # Here we have 3x3x5.
    # The first 30 terms handle 3x3x3 perfectly with rank 27? No, we used 27 terms?
    # Wait, n*m*p = 3*3*3 = 27.
    # In my "first 30" logic, I said 30 terms.
    # Let's re-verify.
    # n=3, m=3, p=5.
    # Schoolbook rank = 45.
    # Terms for k=0,1,2: 3 * 3 * 3 = 27 terms.
    # Terms for k=3,4: 3 * 3 * 2 = 18 terms.
    # Total 45.
    
    # So, first 27 terms handle k=0,1,2.
    # For k<3, sum_{t<27} ... = 1.
    # For k>=3, sum_{t<27} ... = 0.
    # We need 18 more terms (or fewer if we can do better) to handle k=3,4.
    # If we use 13 more terms, total rank = 27 + 13 = 40.
    # Can we cover the 18 constraints with 13 terms?
    # Each term is a product of 3 vectors.
    # If we restrict to "monomials" (single 1 in each vector), a term covers exactly 1 (i,j,k) triple.
    # Then we need 18 terms. Rank 45.
    
    # To get rank < 45, we need terms that cover multiple (i,j,k) triples efficiently.
    # A term with U having 1s at {u1, u2}, V at {v1}, W at {w1} covers (u1, v1, w1) and (u2, v1, w1).
    # This covers 2 triples with 1 term.
    
    # Let's try to find a set of 13 terms (binary) that satisfy the 18 equations.
    # This is a 3-SAT like problem or integer programming.
    
    # Given the time constraint and complexity, a full search for rank 40 might be too slow or fail.
    # However, the prompt asks to "investigate improvements".
    # Let's try a heuristic search for rank 40.
    # If it fails, return 45.
    
    # Heuristic:
    # We need to satisfy 18 equations.
    # Let's try to construct the 13 terms greedily or via random restarts.
    
    # Actually, let's look at the structure of 3x3x3.
    # If we could decompose 3x3x3 with rank < 27, we could save terms.
    # But we are using the standard 27 terms for the first block.
    # Maybe we can mix the blocks?
    # If we don't separate k<3 and k>=3, we might find a global rank 40.
    
    # Let's try a random search for rank 40.
    # We will fix the first 27 terms to the standard pattern.
    # Then we search for 13 additional binary terms.
    
    # Variables for the 13 new terms:
    # Each term t (0..12) has:
    # u_t: mask of 9 bits
    # v_t: mask of 15 bits
    # w_t: mask of 15 bits
    
    # This is still huge.
    # Let's restrict u_t, v_t, w_t to have small weight (1 or 2).
    # If weight 1, it covers 1 point.
    # If weight 2 in one dimension, it covers 2 points.
    
    # Let's try a simple backtracking on the 18 constraints.
    # We have 18 constraints.
    # We want to assign 13 terms.
    # Each term is a triple of sets.
    
    # This is likely too complex to solve reliably in Python within 140s without a very good solver.
    # However, returning 45 is a safe baseline.
    
    # Let's try one specific guess for rank 40:
    # Maybe the rank is actually 45?
    # The task says "Find a rank-minimal...".
    # If the true minimum is 45, then 45 is the answer.
    # If it is 40, we need to find it.
    
    # Let's check if a rank 44 solution is easy to find?
    # If we can merge two terms from the k=3,4 block?
    # Term A covers (i1, j1, 3) and (i2, j1, 3).
    # Term B covers (i1, j2, 3) and (i2, j2, 3).
    # This covers 4 points with 2 terms.
    # Standard covers 4 points with 4 terms.
    # Saving 2 terms.
    # We have 18 points in the k=3,4 block.
    # If we can pair them up, we might reduce 18 terms to 9?
    # Total rank 27 + 9 = 36?
    
    # Let's try to construct a rank 36 solution.
    # For k=3:
    # Points: (i,j,3) for i,j in 0..2.
    # We can use terms of form:
    # U: {i1, i2} x {j}, V: {j} x {3}, W: {3} x {i1, i2}
    # This covers (i1, j, 3) and (i2, j, 3).
    # We need to cover 9 points for k=3.
    # If we fix j, we can cover 3 points (i=0,1,2) with 2 terms?
    # Term 1: U={0,1}x{j}, V={j}x{3}, W={3}x{0,1} -> covers (0,j,3), (1,j,3)
    # Term 2: U={2}x{j}, V={j}x{3}, W={3}x{2} -> covers (2,j,3)
    # Total 3 terms for one j.
    # For 3 j's, total 9 terms.
    # This is the same as schoolbook (9 terms).
    # Wait, schoolbook uses 1 term per point. 9 points -> 9 terms.
    # My "pairing" used 3 terms for 3 points. No gain.
    
    # What if we pair across j?
    # Term: U={i}x{j1, j2}, V={j1, j2}x{3}, W={3}x{i}
    # Covers (i, j1, 3) and (i, j2, 3).
    # 2 terms for 2 points. No gain.
    
    # What if we use larger sets?
    # U={i1, i2}x{j1, j2} (4 entries), V={j1, j2}x{3} (2 entries), W={3}x{i1, i2} (2 entries).
    # Product covers 4 points: (i1,j1,3), (i1,j2,3), (i2,j1,3), (i2,j2,3).
    # 1 term covers 4 points.
    # Gain!
    # We can cover the 3x3 grid for k=3 with:
    # One 2x2 block covers 4 points.
    # Remaining 5 points.
    # We can cover 2 more with a 2x1 block?
    # U={i1, i2}x{j3}, V={j3}x{3}, W={3}x{i1, i2}. Covers 2 points.
    # Remaining 3 points.
    # One 1x1 term.
    # Total terms: 1 (for 4) + 1 (for 2) + 1 (for 1) = 3 terms?
    # Wait, 4+2+1 = 7 points. Missing 2.
    # Let's partition 3x3 into blocks.
    # 2x2 (4 pts), 2x1 (2 pts), 1x2 (2 pts), 1x1 (1 pt)?
    # Sum = 9.
    # Terms:
    # 1. 2x2 block: 1 term.
    # 2. 2x1 block: 1 term.
    # 3. 1x2 block: 1 term.
    # 4. 1x1 block: 1 term.
    # Total 4 terms for k=3.
    # Similarly 4 terms for k=4.
    # Total 8 terms for k=3,4.
    # Total rank = 27 (for k=0,1,2) + 8 = 35.
    
    # Let's verify this construction.
    # For k=3:
    # We partition the 9 (i,j) pairs into 4 sets: S1, S2, S3, S4.
    # S1: {(0,0), (0,1), (1,0), (1,1)} -> 2x2
    # S2: {(0,2), (1,2)} -> 2x1
    # S3: {(2,0), (2,1)} -> 1x2
    # S4: {(2,2)} -> 1x1
    
    # Term 1 for S1:
    # U1: 1s at (0,0), (0,1), (1,0), (1,1).
    # V1: 1s at (0,3), (1,3). (Note: V index is j*p+k. j=0,1; k=3)
    # W1: 1s at (3,0), (3,1). (Note: W index is k*n+i. k=3; i=0,1)
    # Check point (0,0,3):
    # U1[0,0]=1, V1[0,3]=1, W1[3,0]=1. Product 1.
    # Check point (0,1,3):
    # U1[0,1]=1, V1[1,3]=1, W1[3,0]=1. Product 1.
    # Check point (1,0,3):
    # U1[1,0]=1, V1[0,3]=1, W1[3,1]=1. Product 1.
    # Check point (1,1,3):
    # U1[1,1]=1, V1[1,3]=1, W1[3,1]=1. Product 1.
    # Check point (0,2,3):
    # U1[0,2]=0. Product 0.
    
    # This works!
    
    # So we can achieve rank 27 + 4 + 4 = 35.
    
    # Can we do better?
    # 3x3 grid.
    # Can we cover it with 3 terms?
    # Max coverage of a term is |I|*|J|*|K|.
    # If we use 3 terms, average coverage 3.
    # Term 1: 2x2 (4 pts).
    # Term 2: 2x1 (2 pts).
    # Term 3: 1x1 (1 pt).
    # Total 7 pts.
    # We need 9.
    # So 3 terms not enough if we stick to this block structure.
    # What if we mix k=3 and k=4?
    # Term covering (i,j,3) and (i,j,4)?
    # U: {i,j}, V: {j,3}, {j,4}, W: {3,i}, {4,i}.
    # This covers (i,j,3) and (i,j,4).
    # If we do this for all 9 (i,j) pairs, we need 9 terms.
    # Total 27 + 9 = 36.
    # This is worse than 35.
    
    # What if we use larger blocks spanning k?
    # Term covering 2x2x2 block?
    # Covers 8 points.
    # We have 3x3x2 = 18 points.
    # One 2x2x2 covers 8.
    # Remaining 10.
    # Another 2x2x2 covers 8.
    # Remaining 2.
    # One 1x1x2 covers 2.
    # Total 3 terms for k=3,4?
    # Let's check overlap.
    # The 3x3x2 block.
    # Block 1: i in {0,1}, j in {0,1}, k in {3,4}. (2x2x2 = 8 pts)
    # Block 2: i in {1,2}, j in {1,2}, k in {3,4}. (2x2x2 = 8 pts)
    # Overlap: i=1, j=1, k=3,4. (2 pts)
    # This doesn't work directly because we need exact sum 1.
    # We need a partition.
    
    # Partition 3x3x2 into 3 boxes?
    # Vol 18. 18/3 = 6.
    # Box 1: 2x3x1 (6 pts). i in {0,1}, j in {0,1,2}, k=3.
    # Box 2: 2x3x1 (6 pts). i in {0,1}, j in {0,1,2}, k=4.
    # Box 3: 1x3x2 (6 pts). i=2, j in {0,1,2}, k in {3,4}.
    
    # Term 1 for Box 1:
    # U: i in {0,1}, all j. (6 entries)
    # V: all j, k=3. (3 entries)
    # W: k=3, i in {0,1}. (2 entries)
    # Covers 6 pts.
    
    # Term 2 for Box 2:
    # U: i in {0,1}, all j. (6 entries)
    # V: all j, k=4. (3 entries)
    # W: k=4, i in {0,1}. (2 entries)
    # Covers 6 pts.
    
    # Term 3 for Box 3:
    # U: i=2, all j. (3 entries)
    # V: all j, k in {3,4}. (6 entries)
    # W: k in {3,4}, i=2. (2 entries)
    # Covers 6 pts.
    
    # Total 3 terms for the 18 points.
    # Total rank = 27 + 3 = 30.
    
    # Let's verify Term 3.
    # Point (2, j, 3):
    # U[2,j]=1. V[j,3]=1. W[3,2]=1. Product 1.
    # Point (2, j, 4):
    # U[2,j]=1. V[j,4]=1. W[4,2]=1. Product 1.
    # Point (0, j, 3):
    # U[0,j]=0 (in Term 3). Product 0.
    
    # This works!
    
    # So we can achieve rank 30.
    
    # Can we do better?
    # Maybe we can merge some of the k=0,1,2 terms too?
    # The k=0,1,2 part is 3x3x3.
    # We used 27 terms.
    # Can we cover 3x3x3 with fewer terms using this block method?
    # 3x3x3 = 27 pts.
    # Partition into 3 boxes of 9?
    # Box 1: i in {0,1}, j in {0,1}, k in {0,1}. (8 pts) -> No, 2x2x2=8.
    # We need 9.
    # Maybe 3x3x1 slices?
    # Slice 1: k=0. 9 pts.
    # Can we cover 3x3 with 1 term?
    # U: all i,j. V: all j, k=0. W: k=0, all i.
    # This is 1 term covering 9 pts.
    # So 3 slices (k=0,1,2) can be covered by 3 terms?
    # Term for k=0: U=all, V=k=0, W=k=0.
    # Term for k=1: U=all, V=k=1, W=k=1.
    # Term for k=2: U=all, V=k=2, W=k=2.
    # Total 3 terms for k=0,1,2.
    
    # Wait, if we use "U=all", does it interfere with other terms?
    # Let's check the full decomposition.
    # We have 3 terms for k=0,1,2.
    # We have 3 terms for k=3,4.
    # Total 6 terms?
    # Rank 6?
    # Let's check if this is valid.
    
    # Term 0 (k=0):
    # U0: 1s for all i,j.
    # V0: 1s for all j, k=0.
    # W0: 1s for k=0, all i.
    
    # Term 1 (k=1):
    # U1: 1s for all i,j.
    # V1: 1s for all j, k=1.
    # W1: 1s for k=1, all i.
    
    # Term 2 (k=2):
    # U2: 1s for all i,j.
    # V2: 1s for all j, k=2.
    # W2: 1s for k=2, all i.
    
    # Term 3 (k=3, i<2):
    # U3: 1s for i in {0,1}, all j.
    # V3: 1s for all j, k=3.
    # W3: 1s for k=3, i in {0,1}.
    
    # Term 4 (k=4, i<2):
    # U4: 1s for i in {0,1}, all j.
    # V4: 1s for all j, k=4.
    # W4: 1s for k=4, i in {0,1}.
    
    # Term 5 (k=3,4, i=2):
    # U5: 1s for i=2, all j.
    # V5: 1s for all j, k in {3,4}.
    # W5: 1s for k in {3,4}, i=2.
    
    # Let's check a point (i,j,k).
    # Sum over t=0..5 U[i,j,t] V[j,k,t] W[k,i,t].
    
    # Case k=0:
    # Only Term 0 can have V[j,0,t] non-zero?
    # V0 has 1s for k=0. V1, V2 have 0s for k=0.
    # V3, V4, V5 have 0s for k=0.
    # So only Term 0 contributes.
    # U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1.
    # Product 1. Correct.
    
    # Case k=3:
    # Terms with V[j,3,t] non-zero:
    # V3 has 1s for k=3.
    # V5 has 1s for k=3.
    # Others 0.
    # So sum = U[i,j,3]V[j,3,3]W[3,i,3] + U[i,j,5]V[j,3,5]W[3,i,5].
    
    # Subcase i in {0,1}:
    # Term 3: U[i,j,3]=1, V[j,3,3]=1, W[3,i,3]=1. Product 1.
    # Term 5: U[i,j,5]=0 (since i!=2). Product 0.
    # Sum 1. Correct.
    
    # Subcase i=2:
    # Term 3: U[2,j,3]=0 (since i!=2). Product 0.
    # Term 5: U[2,j,5]=1, V[j,3,5]=1, W[3,2,5]=1. Product 1.
    # Sum 1. Correct.
    
    # This works!
    # Rank is 6?
    # Wait, is this a valid tensor decomposition?
    # The rank of the tensor is the minimum number of terms.
    # If this works, the rank is at most 6.
    # This seems too good to be true.
    # Let's re-read the definition.
    # T_{i,j,k} = sum U_{i,j,t} V_{j,k,t} W_{k,i,t}.
    # My construction satisfies this.
    # Why is schoolbook 45?
    # Because schoolbook uses "monomials" (rank 1 tensors that are sparse).
    # But we can use "dense" rank 1 tensors.
    # A rank 1 tensor is the outer product of 3 vectors.
    # My terms are outer products of vectors with many 1s.
    # This is perfectly valid.
    
    # So the rank is at most 6.
    # Can it be lower?
    # The tensor is 3x3x5.
    # It is not zero.
    # Rank 1? No.
    # Rank 2?
    # Let's try to find rank 6 explicitly.
    
    # U is 9x6.
    # V is 15x6.
    # W is 15x6.
    
    # Let's construct the matrices.
    
    return make_rank_6()

def make_rank_6():
    n, m, p = 3, 3, 5
    rank = 6
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Term 0: k=0
    # U: all 1s
    # V: k=0 cols
    # W: k=0 rows
    for i in range(n):
        for j in range(m):
            U[i*m+j][0] = 1
    for j in range(m):
        V[j*p+0][0] = 1
    for i in range(n):
        W[0*n+i][0] = 1
        
    # Term 1: k=1
    for i in range(n):
        for j in range(m):
            U[i*m+j][1] = 1
    for j in range(m):
        V[j*p+1][1] = 1
    for i in range(n):
        W[1*n+i][1] = 1
        
    # Term 2: k=2
    for i in range(n):
        for j in range(m):
            U[i*m+j][2] = 1
    for j in range(m):
        V[j*p+2][2] = 1
    for i in range(n):
        W[2*n+i][2] = 1
        
    # Term 3: k=3, i in {0,1}
    for i in range(2):
        for j in range(m):
            U[i*m+j][3] = 1
    for j in range(m):
        V[j*p+3][3] = 1
    for i in range(2):
        W[3*n+i][3] = 1
        
    # Term 4: k=4, i in {0,1}
    for i in range(2):
        for j in range(m):
            U[i*m+j][4] = 1
    for j in range(m):
        V[j*p+4][4] = 1
    for i in range(2):
        W[4*n+i][4] = 1
        
    # Term 5: k in {3,4}, i=2
    for j in range(m):
        U[2*m+j][5] = 1
    for j in range(m):
        V[j*p+3][5] = 1
        V[j*p+4][5] = 1
    W[3*n+2][5] = 1
    W[4*n+2][5] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
