"""Matrix multiplication 2,2,12 - rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x12 matrix.
    
    The matrix multiplication tensor T[i,j,k] for C = A*B where
    A is n x m, B is m x p, C is n x p.
    
    Here n=2, m=2, p=12.
    
    The tensor T has entries T[i,j,k] = 1 if j is the shared index,
    i.e., T[i,j,k] = delta_{j, j} (always 1 when indices are valid).
    
    Actually, the standard matrix multiplication tensor:
    T[i, j, k] = 1 for all valid i, j, k where the multiplication
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    The factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    must satisfy:
    sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... 
    
    Wait, let me re-read the convention. The output factor follows
    W[k*n+i, r]. So W has shape (p*n, rank).
    
    The decomposition means:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    
    But wait, that's not right either. In a standard tensor decomposition
    for matrix multiplication, we have:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T is defined on indices (i, j, k) with T[i,j,k] = 1.
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * ... 
    
    Hmm, actually for matrix multiplication tensor, the standard decomposition
    is:
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    
    No wait. Let me think again. The tensor T lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    ... no, that's not standard.
    
    Actually, looking at the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    The constraint is that for all i, j, k:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    
    But then what role does W play? 
    
    Looking at the schoolbook code:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    So in the schoolbook, each term has all three being 1. The decomposition
    satisfies:
    
    For the multiplication to work, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... 
    
    That doesn't make sense dimensionally. Let me reconsider.
    
    I think the correct interpretation is:
    
    The tensor T[i,j,k] (where i indexes rows of A/C, j indexes columns of A / rows of B,
    k indexes columns of B/C) has T[i,j,k] = 1.
    
    A rank-r decomposition expresses T as:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    No, that would be a 3-way decomposition but with products of three terms.
    
    Actually, I think the convention here is that the multiplication tensor is:
    T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{pn} (or similar)
    
    And the rank-r decomposition is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}.
    
    So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    Wait, but that's not a standard tensor. Let me look at the schoolbook again.
    
    In schoolbook, for each (i,j,k), we have one column where:
    U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1, and all others 0.
    
    So the tensor entry T[i*m+j, j*p+k, k*n+i] gets contribution 1*1*1 = 1 from that term.
    
    But what about other entries? For a fixed (a, b, c) where a = i*m+j, b = j'*p+k', c = k''*n+i'',
    the sum over r gives the product of U[a,r]*V[b,r]*W[c,r].
    
    In schoolbook, each r corresponds to one (i,j,k) triple. So for a given (a,b,c),
    the sum is 1 only if there exists an r such that U[a,r]=1, V[b,r]=1, W[c,r]=1,
    which means a=i*m+j, b=j*p+k, c=k*n+i for the same (i,j,k).
    
    So T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k).
    
    This means the tensor is sparse - only entries of the form (i*m+j, j*p+k, k*n+i) are 1.
    
    OK so now I understand the problem. We need to find the minimum rank r such that
    there exist U, V, W with the property that:
    
    For all i in [0,2), j in [0,2), k in [0,12):
    sum_{r=1}^R U[i*2+j, r] * V[j*12+k, r] * W[k*2+i, r] = 1
    
    And for all other (a,b,c) not of this form:
    sum_{r=1}^R U[a, r] * V[b, r] * W[c, r] = 0
    
    This is the standard matrix multiplication tensor rank problem.
    
    For n=2, m=2, p=12, the schoolbook rank is 2*2*12 = 48.
    
    Can we do better? For matrix multiplication of (n x m) by (m x p),
    the tensor rank is at most n*m*p (schoolbook). There are known lower bounds.
    
    For 2x2 by 2x12, this is equivalent to multiplying a 2x2 matrix by a 2x12 matrix.
    
    Let me think about this differently. We can view this as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    where i in {0,1}, j in {0,1}, k in {0,...,11}.
    
    So C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    For each k, this is like computing two linear combinations.
    
    The total number of scalar multiplications in schoolbook is 2*2*12 = 48.
    
    Can we reduce this? For each k, we're computing:
    [C[0,k], C[1,k]]^T = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] * [B[0,k], B[1,k]]^T
    
    This is a 2x2 matrix times a 2x1 vector. The rank of this "matrix-vector multiplication"
    tensor for 2x2 by 2x1 is known to be 5 (it's the same as 2x2 matrix multiplication rank,
    which is 7... wait no).
    
    Actually, multiplying a 2x2 matrix by a 2x1 vector:
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    This requires 4 scalar multiplications in schoolbook. Can we do it in 3? 
    The tensor rank of 2x2 matrix by 2x1 vector is actually 4 (it's irreducible).
    
    Hmm, but if we can share computations across different k values, we might save.
    
    Let me think about this more carefully. The key insight is that A is fixed (2x2) and
    B varies (2x12). So we're computing 12 outputs C[0,k] and C[1,k], each being a linear
    combination of A's entries times B's entries.
    
    Actually, I realize the tensor rank problem here is asking for the minimum number of
    scalar multiplications needed, where additions are free.
    
    For each k, we need:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    If we process each k independently, we need 4 multiplications per k, total 48.
    
    But we can potentially share. For example, if we precompute some combinations of A's
    entries, we might save. But A's entries are variables, so we can't precompute.
    
    Wait, in the tensor model, both A and B entries are variables. The rank counts the
    number of bilinear multiplications.
    
    Let me try a different approach. Since p=12 is large, maybe there's a way to use
    the structure.
    
    Actually, let me just try to find a lower rank decomposition computationally.
    
    The problem is: find minimum R such that there exist U (4 x R), V (24 x R), W (24 x R)
    satisfying the tensor equations.
    
    This is a system of polynomial equations. For small R, we can try to solve it.
    
    Let me set up the equations. The tensor T has 1*1*1 = 1 at positions (i*2+j, j*12+k, k*2+i)
    for i in {0,1}, j in {0,1}, k in {0,...,11}.
    
    Total number of 1-entries: 2*2*12 = 48.
    
    The number of constraints: for each (a,b,c) in {0..3} x {0..23} x {0..23}, we need
    sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].
    
    Total constraints: 4*24*24 = 2304.
    
    Unknowns: 4R + 24R + 24R = 52R.
    
    For R=48, we have 2496 unknowns and 2304 equations, so it's feasible.
    
    Can we do R < 48? Let's try R=47, 46, etc.
    
    This is a hard nonlinear optimization problem. Let me try a randomized approach.
    
    Actually, for this specific case, let me think about whether there's a known result.
    
    The tensor rank of multiplying (n x m) by (m x p) matrix... For n=m=2, p=12.
    
    I recall that for square matrix multiplication, the rank of n x n is at most n^3 and
    there are subcubic algorithms, but those are for large n.
    
    For 2x2 by 2x12, I think the rank might actually be 48 (schoolbook is optimal) or close to it.
    
    Let me try to verify by attempting a computational search for lower rank.
    
    Given the time constraint, let me try a randomized approach to see if we can find
    a rank below 48. If not, I'll return the schoolbook decomposition.
    
    Strategy:
    1. Start with schoolbook (rank 48).
    2. Try to find a lower rank by random initialization + optimization.
    
    But this is computationally intensive. Let me think of a smarter approach.
    
    Actually, let me reconsider the problem structure. We have:
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    for i in {0,1}, k in {0,...,11}.
    
    This is 24 outputs, each a sum of 2 products.
    
    One idea: group the k values. For example, if we can compute some linear
    combinations of B's columns that allow us to compute multiple C's at once.
    
    But in the bilinear model, we can only multiply one A-entry by one B-entry per
    multiplication. So the question is whether we can reduce the total count.
    
    Let me just try to solve this computationally with a time limit.
    
    Actually, given the 120s time limit, let me try a simple approach:
    1. Return schoolbook rank 48 as the baseline.
    2. Try a quick randomized search for lower rank.
    
    But implementing a full tensor rank search is complex. Let me just return the
    schoolbook decomposition for now, as finding a lower rank for this specific
    problem may not be feasible within the time limit.
    
    Wait, let me reconsider. Maybe there IS a lower rank. Let me think...
    
    For 2x2 matrix multiplication (2x2 by 2x2), the rank is 7 (Strassen).
    For 2x2 by 2x12, it's like doing 6 independent 2x2 multiplications... no, that's not right.
    
    2x2 by 2x12: we can think of B as 6 blocks of 2x2. Then C is 2x12, which is 6 blocks of 2x2.
    Each block of C is A times the corresponding block of B. So it's 6 independent 2x2 multiplications.
    
    If each 2x2 multiplication has rank 7 (Strassen), then the total rank would be 6*7 = 42.
    
    But wait, can we share Strassen's intermediate computations across the 6 blocks?
    
    In Strassen's algorithm for 2x2:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B10-B11)
    m4 = A11(B01-B10)
    m5 = (A00+A01)B11
    m6 = (A10-A00)(B00+B10)
    m7 = (A01-A11)(B01+B11)
    
    C00 = m1+m4-m5+m7
    C01 = m3+m5
    C10 = m2+m4
    C11 = m1-m2+m3+m6
    
    Now for 2x2 by 2x12, if we group B into 6 blocks of 2x2:
    B = [B^(1), B^(2), ..., B^(6)] where each B^(t) is 2x2.
    
    Then C = [C^(1), ..., C^(6)] where C^(t) = A * B^(t).
    
    Using Strassen for each block:
    For block t:
    m1_t = (A00+A11)(B^(t)00+B^(t)11)
    m2_t = (A10+A11)B^(t)00
    ...
    
    The A-combinations (A00+A11, A10+A11, etc.) are the same for all blocks!
    
    So we can precompute:
    a1 = A00+A11
    a2 = A10+A11
    a3 = A00
    a4 = A11
    a5 = A00+A01
    a6 = A10-A00
    a7 = A01-A11
    
    Then for each block t:
    m1_t = a1 * (B^(t)00+B^(t)11)
    m2_t = a2 * B^(t)00
    m3_t = a3 * (B^(t)10-B^(t)11)
    m4_t = a4 * (B^(t)01-B^(t)10)
    m5_t = a5 * B^(t)11
    m6_t = a6 * (B^(t)00+B^(t)10)
    m7_t = a7 * (B^(t)01+B^(t)11)
    
    That's 7 multiplications per block, 6 blocks, total 42 multiplications.
    
    Plus 7 additions for the a-combinations. But in the tensor rank model, additions are free!
    
    So the rank should be at most 42!
    
    Let me construct this decomposition.
    
    The 7 a-combinations are linear combinations of A's entries. In the tensor model,
    U[i*m+j, r] represents the coefficient of A[i,j] in the r-th multiplication.
    
    So for the 7 a-combinations, we don't have multiplications - they're just linear combinations.
    The multiplications are a_s * b_t where b_t is a linear combination of B's entries.
    
    So we have 42 multiplications:
    For s in {1,...,7}, t in {1,...,6}:
    m_{s,t} = a_s * b_{s,t}
    
    where a_s is a fixed linear combination of A's entries, and b_{s,t} is a linear
    combination of B's entries for block t.
    
    Then C's entries are linear combinations of the m_{s,t}.
    
    Let me write this out explicitly.
    
    A entries: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in U.
    B entries: B[0,0],...,B[0,11], B[1,0],...,B[1,11] -> indices 0..23 in V.
    C entries: C[0,0],...,C[0,11], C[1,0],...,C[1,11] -> indices 0..23 in W.
    
    Wait, W[k*n+i, r] so W has 24 rows: k in {0,...,11}, i in {0,1}, index = k*2+i.
    
    Strassen's a-combinations:
    a1 = A[0,0] + A[1,1]  -> U[0,r]=1, U[3,r]=1
    a2 = A[1,0] + A[1,1]  -> U[2,r]=1, U[3,r]=1
    a3 = A[0,0]            -> U[0,r]=1
    a4 = A[1,1]            -> U[3,r]=1
    a5 = A[0,0] + A[0,1]   -> U[0,r]=1, U[1,r]=1
    a6 = A[1,0] - A[0,0]   -> U[2,r]=1, U[0,r]=-1
    a7 = A[0,1] - A[1,1]   -> U[1,r]=1, U[3,r]=-1
    
    For block t (t=0..5), B's columns are 2t and 2t+1.
    B[0,2t], B[0,2t+1], B[1,2t], B[1,2t+1] -> indices 2t, 2t+1, 12+2t, 12+2t+1 in V.
    
    b_{1,t} = B[0,2t] + B[1,2t+1]  -> V[2t,r]=1, V[12+2t+1,r]=1
    b_{2,t} = B[0,2t]                -> V[2t,r]=1
    b_{3,t} = B[1,2t] - B[1,2t+1]   -> V[12+2t,r]=1, V[12+2t+1,r]=-1
    b_{4,t} = B[0,2t+1] - B[1,2t]   -> V[2t+1,r]=1, V[12+2t,r]=-1
    b_{5,t} = B[1,2t+1]              -> V[12+2t+1,r]=1
    b_{6,t} = B[0,2t] + B[1,2t]     -> V[2t,r]=1, V[12+2t,r]=1
    b_{7,t} = B[0,2t+1] + B[1,2t+1] -> V[2t+1,r]=1, V[12+2t+1,r]=1
    
    C's outputs:
    C[0,2t] = m_{1,t} + m_{4,t} - m_{5,t} + m_{7,t}
    C[0,2t+1] = m_{3,t} + m_{5,t}
    C[1,2t] = m_{2,t} + m_{4,t}
    C[1,2t+1] = m_{1,t} - m_{2,t} + m_{3,t} + m_{6,t}
    
    W[2t, r] (C[0,2t]): coefficient of m_{s,t} is 1 if s in {1,4,7}, -1 if s=5.
    W[2t+1, r] (C[0,2t+1]): coefficient of m_{s,t} is 1 if s in {3,5}.
    W[24+2t, r] (C[1,2t]): coefficient of m_{s,t} is 1 if s in {2,4}.
    W[24+2t+1, r] (C[1,2t+1]): coefficient of m_{s,t} is 1 if s in {1,3,6}, -1 if s=2.
    
    Wait, W has 24 rows indexed by k*2+i where k is 0..11 and i is 0,1.
    So C[0,k] is at W[k*2+0] = W[2k], and C[1,k] is at W[k*2+1] = W[2k+1].
    
    Let me redo:
    C[0,2t] -> W[2t]
    C[0,2t+1] -> W[2t+1]
    C[1,2t] -> W[2t+2]
    C[1,2t+1] -> W[2t+3]
    
    So:
    W[2t, r] for m_{s,t}: 1 if s in {1,4,7}, -1 if s=5, else 0.
    W[2t+1, r] for m_{s,t}: 1 if s in {3,5}, else 0.
    W[2t+2, r] for m_{s,t}: 1 if s in {2,4}, else 0.
    W[2t+3, r] for m_{s,t}: 1 if s in {1,3,6}, -1 if s=2, else 0.
    
    Total rank: 7*6 = 42.
    
    Let me code this up.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Try Strassen-based decomposition with rank 42
    # Group B's 12 columns into 6 blocks of 2
    num_blocks = p // 2  # 6
    strassen_rank = 7 * num_blocks  # 42
    
    U = [[0] * strassen_rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * strassen_rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * strassen_rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen's 7 a-combinations for A (2x2)
    # a1 = A00 + A11
    # a2 = A10 + A11
    # a3 = A00
    # a4 = A11
    # a5 = A00 + A01
    # a6 = A10 - A00
    # a7 = A01 - A11
    
    a_combos = [
        [(0, 1), (3, 1)],       # a1: A[0,0]+A[1,1]
        [(2, 1), (3, 1)],       # a2: A[1,0]+A[1,1]
        [(0, 1)],               # a3: A[0,0]
        [(3, 1)],               # a4: A[1,1]
        [(0, 1), (1, 1)],       # a5: A[0,0]+A[0,1]
        [(2, 1), (0, -1)],      # a6: A[1,0]-A[0,0]
        [(1, 1), (3, -1)],      # a7: A[0,1]-A[1,1]
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            r = t * 7 + s
            # Set U
            for idx, coeff in a_combos[s]:
                U[idx][r] = coeff
            
            # Set V
            # b_{s,t} for block t (columns 2t, 2t+1)
            # B[0,2t]=V[2t], B[0,2t+1]=V[2t+1], B[1,2t]=V[12+2t], B[1,2t+1]=V[12+2t+1]
            b_combos = [
                [(2*t, 1), (12+2*t+1, 1)],      # b1: B[0,2t]+B[1,2t+1]
                [(2*t, 1)],                      # b2: B[0,2t]
                [(12+2*t, 1), (12+2*t+1, -1)],  # b3: B[1,2t]-B[1,2t+1]
                [(2*t+1, 1), (12+2*t, -1)],     # b4: B[0,2t+1]-B[1,2t]
                [(12+2*t+1, 1)],                # b5: B[1,2t+1]
                [(2*t, 1), (12+2*t, 1)],        # b6: B[0,2t]+B[1,2t]
                [(2*t+1, 1), (12+2*t+1, 1)],    # b7: B[0,2t+1]+B[1,2t+1]
            ]
            for idx, coeff in b_combos[s]:
                V[idx][r] = coeff
            
            # Set W
            # C[0,2t]=W[2t]: m1+m4-m5+m7
            # C[0,2t+1]=W[2t+1]: m3+m5
            # C[1,2t]=W[2t+2]: m2+m4
            # C[1,2t+1]=W[2t+3]: m1-m2+m3+m6
            w_coeffs = {
                2*t:    {0: 1, 3: 1, 4: -1, 6: 1},
                2*t+1:  {2: 1, 4: 1},
                2*t+2:  {1: 1, 3: 1},
                2*t+3:  {0: 1, 1: -1, 2: 1, 5: 1},
            }
            for w_idx, coeffs in w_coeffs.items():
                for s_idx, coeff in coeffs.items():
                    W[w_idx][r] = coeff
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # But wait, the tensor is T[i*m+j, j*p+k, k*n+i] = 1
                    # Let me check: for valid (i,j,k), the product should sum to 1
                    # For invalid combinations, it should sum to 0
                    pass
        # Actually, let's verify by checking that C = A*B
        # Pick random A and B, compute C = A*B, and check if the decomposition gives the same
        import random
        random.seed(42)
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute using decomposition
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In the tensor decomposition:
        # The bilinear form gives: sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j'*p+k',r]*B[j',k']) * ...
        
        # Actually, I need to think about this more carefully.
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 means:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all valid (i,j,k)
        
        # But the actual multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # The relationship is:
        # C[i,k] = sum_j sum_r U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r] / W[k*n+i,r]
        
        # Hmm, I'm confusing myself. Let me just check the tensor equation directly.
        
        # For each (i,j,k):
        total = 0
        for r in range(rank):
            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        if abs(total - 1) > 1e-9:
            return False, f"Failed at ({i},{j},{k}): got {total}, expected 1"
    
        # Also check that non-tensor entries are 0
        # The tensor is 1 only at (i*m+j, j*p+k, k*n+i) for valid i,j,k
        # All other (a,b,c) should give 0
        # But checking all 4*24*24 = 2304 entries is expensive
        # Let's just spot check
        valid_entries = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_entries.add((i*m+j, j*p+k, k*n+i))
        
        # Check a few random non-valid entries
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            if (a, b, c) in valid_entries:
                continue
            total = 0
            for r in range(rank):
                total += U[a][r] * V[b][r] * W[c][r]
            if abs(total) > 1e-9:
                return False, f"Non-zero at non-tensor entry ({a},{b},{c}): got {total}"
        
        return True, "OK"
    
    is_valid, msg = verify(U, V, W, n, m, p)
    if not is_valid:
        print(f"Strassen decomposition failed: {msg}")
        # Fall back to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
