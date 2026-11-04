import numpy as np
import time
from itertools import product

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use Strassen's algorithm recursively.
    - 4x4x4: rank 49 (Strassen^2)
    - 4x4x8: We can view this as two 4x4x4 multiplications, but we can do better
      by using the rectangular nature.
    
    Key insight: 4x4x8 can be decomposed using a rank-70 construction based on
    8x8x8 Strassen recursion, or more directly:
    
    The best known rank for 4x4x4 is 49 (Strassen).
    For 4x4x8, we can use the following approach:
    
    Method 1: Two independent 4x4x4 blocks -> rank 98
    Method 2: Use the fact that 4x4x8 = 4x4x4 + 4x4x4 but share structure
    
    Actually, let's think about this more carefully.
    
    The tensor for multiplying A (4x4) by B (4x8) to get C (4x8) is:
    T[i,j,k] = 1 if i==k else 0, for i in 0..3, j in 0..3, k in 0..7
    
    This is essentially two copies of the 4x4x4 identity tensor, one for k=0..3
    and one for k=4..7.
    
    A rank-r decomposition means:
    sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t] = T[i,j,k]
    
    where U has shape (16, r), V has shape (32, r), W has shape (32, r).
    
    Using Strassen for 4x4x4 gives rank 49. Two independent copies give rank 98.
    
    But we can do better! The key is that the U factor is shared between the two
    output blocks. Let's see if we can find a lower rank.
    
    Known results:
    - Rank(4x4x4) = 49 (Strassen)
    - For 4x4x8, the best known is around 70-80
    
    Let me try a different approach: use the 2x2x2 Strassen (rank 7) as the base,
    then build up to 4x4x8.
    
    4x4x8 = (2x2x2)^2 in the first two dimensions, and 2x2 in the third.
    
    Actually, let's use a cleaner construction:
    
    For 4x4x4, Strassen gives rank 49.
    For 4x4x8, we can think of it as:
    - Split B into B1 (4x4) and B2 (4x4)
    - C1 = A * B1, C2 = A * B2
    - Each is a 4x4x4 multiplication
    
    If we use independent Strassen for each, rank = 49 + 49 = 98.
    
    To do better, we need to share computation between the two blocks.
    
    One approach: use a rank-7 decomposition for 2x2x2, then build 4x4x4 from it
    using Kronecker products, but for the 4x4x8 case, we can use a mixed construction.
    
    Let me try: 4x4x8 can be seen as (2x2x2) x (2x2x4) where the second factor
    is 2x2x4.
    
    2x2x4: multiply 2x2 by 2x4. This can be done with rank... 
    Naive: 2*2*4 = 16
    Strassen-like: We can split the 4 columns into two 2-column blocks.
    So 2x2x4 = 2x2x2 + 2x2x2, giving rank 7+7 = 14 with independent Strassen.
    
    Then 4x4x8 = (2x2x2) x (2x2x4) via Kronecker:
    rank = 7 * 14 = 98. Still 98.
    
    Hmm, that's not helping.
    
    Let me try yet another approach. What if we use a rank-6 decomposition for 2x2x2?
    No, rank(2x2x2) = 7 is proven minimal.
    
    What about using a different base? The best known rank for 3x3x3 is 23 (Lader).
    But that doesn't directly help for 4x4x8.
    
    Let me reconsider. The score for rank 98 is 1.3061 (from the inspirations).
    The seed has rank 128 (naive) with score 1.0.
    
    Wait, the current program returns rank 98 and gets score 1.3061.
    The inspirations also show 1.3061, which corresponds to rank 98.
    
    So the island best is rank 98 (score 1.3061). I need to beat that.
    
    Can I find a rank < 98 decomposition?
    
    Known lower bound: rank(4x4x8) >= rank(4x4x4) = 49.
    Upper bound: 98 (two independent Strassen).
    
    Actually, I recall that for rectangular matrix multiplication, there are
    better constructions. Let me try to use the following:
    
    For 4x4x8, we can use a construction based on 8x8x8 Strassen (rank 343)
    and then restrict to the 4x4x8 sub-tensor. But that's overkill.
    
    Alternatively, let me try to directly search for a lower rank decomposition
    using a randomized approach or a structured search.
    
    Given the time constraints, let me try a practical approach:
    1. Start with rank 98 (two Strassen blocks)
    2. Try to find linear dependencies among the columns to reduce rank
    
    Actually, let me try a completely different approach: use the fact that
    4x4x8 has a special structure and try to find a rank-70 or rank-63 decomposition.
    
    Here's an idea: use a rank-7 decomposition for 2x2x2, then build 4x4x4 with
    rank 49, but for 4x4x8, use a shared U factor.
    
    Wait, I think I can do this:
    
    Let me define the 4x4x8 tensor and try to find a decomposition with rank ~70.
    
    Approach: Use a "block Strassen" where we treat the 4x4x8 as:
    - A is 4x4
    - B is 4x8
    - C is 4x8
    
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks: B = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    Split C into 2x2 blocks: C = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    
    Then C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    etc.
    
    This is getting complex. Let me just implement the rank-98 solution and hope
    that the evaluator accepts it, since the inspirations show it scores 1.3061.
    
    Actually, re-reading the task: "score is the rank" and "lower is better".
    The seed program has score 1.0, which corresponds to... let me check.
    
    The seed program returns rank 98 (from the code). If score = 1/seed_value * value,
    and seed_value is the rank of the seed, then...
    
    Wait, the task says "score is the rank" and "The evaluator measures the raw value
    (lower is better). The search score is seed/value, so 1.0 equals the seed program
    and above 1.0 is an improvement".
    
    So if seed rank = 128 (naive 4*4*8 = 128), then:
    - Rank 128: score = 128/128 = 1.0
    - Rank 98: score = 128/98 ≈ 1.306
    
    That matches! So the seed is the naive rank-128 decomposition, and the current
    program (rank 98) scores 1.3061.
    
    To improve, I need rank < 98.
    
    Let me try to find a rank-70 decomposition. I'll use a structured approach:
    
    Key insight: For 4x4x8, we can use the following construction:
    - Use Strassen for the 4x4x4 part (rank 49)
    - But share the U factor between the two output blocks
    
    Actually, the U factor for 4x4x4 depends on (i,j) where i is the output row
    and j is the input column. For 4x4x8, the U factor still depends on (i,j)
    where i is the output row (0..3) and j is the input column (0..3).
    The V factor depends on (j,k) where j is the input column and k is the output
    column (0..7). The W factor depends on (k,i) where k is the output column and
    i is the output row.
    
    So U is shared between the two output blocks! This means we can potentially
    reduce the rank by finding a joint decomposition.
    
    Let me try: U has shape (16, r), V has shape (32, r), W has shape (32, r).
    
    For the first output block (k=0..3):
    sum_t U[i,j,t] * V1[j,k,t] * W1[k,i,t] = delta(i,k)
    
    For the second output block (k=4..7):
    sum_t U[i,j,t] * V2[j,k-4,t] * W2[k-4,i,t] = delta(i,k-4)
    
    where V1, W1 are the V, W for the first block, and V2, W2 for the second.
    
    If we use the same Strassen decomposition for both blocks, we get:
    V1 = V4, W1 = W4, V2 = V4, W2 = W4
    and U = U4 for both.
    
    But we can potentially use different U factors for the two blocks, or find
    a joint U that works for both.
    
    This is a complex optimization problem. Given the time constraints, let me
    implement the rank-98 solution and return it. At least it's verified and scores
    1.3061.
    
    Actually, let me try one more thing: use a rank-7 decomposition for 2x2x2,
    then build 4x4x4 with rank 49, but for 4x4x8, use a "half-Strassen" approach
    where we only do the full Strassen for one block and a naive decomposition
    for the other, but with shared U.
    
    No, that would give rank 49 + 16 = 65, but the U factors would need to be
    compatible, which they're not in general.
    
    Let me just go with rank 98.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8: two independent blocks
    U_base = np.zeros((16, 98), dtype=float)
    V_base = np.zeros((32, 98), dtype=float)
    W_base = np.zeros((32, 98), dtype=float)
    
    U_base[:, 0:49] = U4
    U_base[:, 49:98] = U4
    
    for j in range(4):
        for k in range(4):
            V_base[j*8+k, 0:49] = V4[j*4+k, :]
            V_base[j*8+(k+4), 49:98] = V4[j*4+k, :]
    
    for k in range(4):
        for i in range(4):
            W_base[k*4+i, 0:49] = W4[k*4+i, :]
            W_base[(k+4)*4+i, 49:98] = W4[k*4+i, :]
    
    # Verify
    def verify_full(U, V, W, tol=1e-10):
        n, m, p = 4, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    expected = 1.0 if i == k else 0.0
                    if abs(val - expected) > tol:
                        return False
        return True
    
    if not verify_full(U_base, V_base, W_base):
        return _naive_decomposition(n, m, p)
    
    rank = 98
    U = U_base[:, :rank]
    V = V_base[:, :rank]
    W = W_base[:, :rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _naive_decomposition(n, m, p):
    """Fallback: naive rank n*m*p decomposition."""
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
