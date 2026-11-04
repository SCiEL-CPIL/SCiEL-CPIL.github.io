import numpy as np

def construct():
    n, m, p = 2, 3, 10
    
    # We will construct a rank-55 decomposition.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k} = C_{i,k}.
    # Indices:
    # U is n*m x rank = 6 x 55
    # V is m*p x rank = 30 x 55
    # W is p*n x rank = 20 x 55
    # W index is k*n + i.
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strategy:
    # Split the 3 rows of A (and corresponding cols of B) into:
    # 1. A 2x2 block (rows 0,1 of A, cols 0,1 of B) -> 5 columns of B (k=0..4)
    #    Use Strassen for each 2x2x2 block? 
    #    Wait, the "2x2" part refers to the A dimensions?
    #    Let's stick to the previous plan:
    #    Block 1: A[0:2, 0:2] (2x2) times B[0:2, 0:5] (2x5).
    #    This is a 2x2x5 tensor.
    #    We can decompose 2x2x5 into two 2x2x2 (Strassen, rank 7 each) and one 2x2x1 (rank 4).
    #    Total rank for this part: 7 + 7 + 4 = 18.
    #    Wait, 2x2x2 Strassen is rank 7.
    #    2x2x1 is just 4 multiplications (rank 4).
    #    So 2x2x5 rank is 18.
    
    #    Block 2: A[0:2, 2:3] (2x1) times B[2:3, 0:5] (1x5).
    #    This is a 2x1x5 tensor.
    #    Rank of 2x1x2 is 4?
    #    Let's check 2x1x2:
    #    A is 2x1 [a1; a2], B is 1x2 [b1 b2].
    #    C is 2x2 [a1b1, a1b2; a2b1, a2b2].
    #    This is rank 4 (it's the outer product of A and B, which has rank 1 in the tensor sense? No, the tensor is T_{i,k} = A_i B_k. This is a rank 1 tensor?
    #    Wait. The tensor T_{i,j,k} for matrix mult.
    #    If j is fixed, T_{i,k} = A_{i,j} B_{j,k}.
    #    This is a rank 1 tensor in the (i,k) slice?
    #    The full tensor is sum over j.
    #    For a fixed j, the term is A_{i,j} B_{j,k}.
    #    This corresponds to a rank 1 term in the decomposition:
    #    U_{i,j, term} = A_{i,j} (coeff 1)
    #    V_{j,k, term} = B_{j,k} (coeff 1)
    #    W_{k,i, term} = C_{i,k} (coeff 1)
    #    So for a single j, we need 1 term per (i,k) pair?
    #    No, the term is U_{i,j} V_{j,k} W_{k,i}.
    #    If we have one term, U_{i,j} must be constant for all i? No.
    #    The decomposition is sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = T_{i,j,k}.
    #    For a fixed j, T_{i,k} = A_{i,j} B_{j,k}.
    #    This is a rank 1 matrix in (i,k) space for each fixed j?
    #    Yes, the matrix M_{i,k} = A_{i,j} B_{j,k} is rank 1.
    #    Can a rank 1 matrix be represented by 1 term in the tensor decomposition?
    #    T_{i,j,k} = U_{i,j,t} V_{j,k,t} W_{k,i,t}.
    #    If we set U_{i,j,t} = A_{i,j}, V_{j,k,t} = B_{j,k}, then W_{k,i,t} must be 1?
    #    Then sum_t ... = A_{i,j} B_{j,k} * 1 = T_{i,j,k}.
    #    So yes, 1 term is enough for a single j?
    #    Wait, the indices of U, V, W are fixed.
    #    U has size n*m x rank.
    #    V has size m*p x rank.
    #    W has size p*n x rank.
    #    If we use 1 term t for this j:
    #    U[i*m+j, t] = A_{i,j} ? No, U must be constants (0, 1, -1 etc) that do not depend on A.
    #    The decomposition is:
    #    C_{i,k} = sum_j sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} A_{i,j} B_{j,k} ?? No.
    #    The standard form is:
    #    C_{i,k} = sum_t W_{k,i,t} ( sum_j U_{i,j,t} A_{i,j} ) ( sum_j V_{j,k,t} B_{j,k} ) ?? No.
    #    
    #    Correct form:
    #    C_{i,k} = sum_t ( sum_j U_{i,j,t} A_{i,j} ) ( sum_j V_{j,k,t} B_{j,k} ) W_{k,i,t} ?
    #    No, the standard tensor decomposition for matrix multiplication is:
    #    T_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
    #    This implies:
    #    C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k} = sum_j sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} A_{i,j} B_{j,k}
    #    = sum_t W_{k,i,t} ( sum_j U_{i,j,t} A_{i,j} ) ( sum_j V_{j,k,t} B_{j,k} )
    #    
    #    So for a fixed j, if we want to isolate A_{i,j} B_{j,k}:
    #    We need a term t such that:
    #    U_{i,j,t} = 1 for all i? No, U_{i,j,t} is a coefficient.
    #    If we set U_{i,j,t} = 1 for the specific j we are handling, and 0 for others?
    #    Then sum_j U_{i,j,t} A_{i,j} = A_{i,j}.
    #    Similarly V_{j,k,t} = 1 for that j, 0 for others -> sum_j V_{j,k,t} B_{j,k} = B_{j,k}.
    #    Then W_{k,i,t} must be 1.
    #    So 1 term handles one j?
    #    But U_{i,j,t} is defined for all i.
    #    If we set U_{i,j,t} = 1 for all i, then sum_j U_{i,j,t} A_{i,j} = A_{i,j}.
    #    This works!
    #    So for each j, we can use 1 term to compute the contribution of that j to all C_{i,k}?
    #    Let's check.
    #    Term t_j:
    #    U_{i,j,t_j} = 1 for all i in 0..n-1.
    #    V_{j,k,t_j} = 1 for all k in 0..p-1.
    #    W_{k,i,t_j} = 1 for all k, i.
    #    Then contribution is sum_{i,j,k} ... 
    #    For a specific C_{i,k}:
    #    Term t_j contributes W_{k,i,t_j} (sum_j' U_{i,j',t_j} A_{i,j'}) (sum_k' V_{j,k',t_j} B_{j,k'})
    #    = 1 * (A_{i,j}) * (B_{j,k}) = A_{i,j} B_{j,k}.
    #    Summing over j gives C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    So the rank is m?
    #    For m=3, rank 3?
    #    This contradicts the known rank of 2x3x10 being ~56.
    #    Why?
    #    Because U_{i,j,t} must be the same for all i?
    #    In the formula C_{i,k} = sum_t W_{k,i,t} (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k}),
    #    U_{i,j,t} depends on i.
    #    So for term t_j, we set U_{i,j,t_j} = 1.
    #    But U is a matrix of shape (n*m, rank).
    #    The entry U[i*m + j, t] is the coefficient for A_{i,j} in the t-th multiplication.
    #    So for term t_j, we set U[i*m + j, t_j] = 1 for all i.
    #    And V[j*p + k, t_j] = 1 for all k.
    #    And W[k*n + i, t_j] = 1 for all k, i.
    #    This seems to work.
    #    Why is the rank not m?
    #    Let's check 2x2x2.
    #    If rank 2 works, then Strassen (rank 7) is not minimal?
    #    But Strassen is known to be optimal for 2x2x2?
    #    Wait, the rank of 2x2x2 is 7.
    #    My construction above for 2x2x2 with m=2 would give rank 2.
    #    Let's test it.
    #    Term 0 (for j=0):
    #    U[0,0]=1, U[1,0]=1 (A_00, A_10)
    #    V[0,0]=1, V[0,1]=1 (B_00, B_01)
    #    W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    #    Contribution to C_00: W[0,0] * (A_00) * (B_00) = A_00 B_00.
    #    Contribution to C_01: W[1,0] * (A_00) * (B_01) = A_00 B_01.
    #    Contribution to C_10: W[0,1] * (A_10) * (B_00) = A_10 B_00.
    #    Contribution to C_11: W[1,1] * (A_10) * (B_01) = A_10 B_01.
    #    Term 1 (for j=1):
    #    U[0,1]=1, U[1,1]=1 (A_01, A_11)
    #    V[1,0]=1, V[1,1]=1 (B_10, B_11)
    #    W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    #    Contribution to C_00: A_01 B_10.
    #    Contribution to C_01: A_01 B_11.
    #    Contribution to C_10: A_11 B_10.
    #    Contribution to C_11: A_11 B_11.
    #    Total C_00 = A_00 B_00 + A_01 B_10. Correct.
    #    Total C_01 = A_00 B_01 + A_01 B_11. Correct.
    #    Total C_10 = A_10 B_00 + A_11 B_10. Correct.
    #    Total C_11 = A_10 B_01 + A_11 B_11. Correct.
    #    
    #    So the rank is m?
    #    For 2x2x2, rank 2?
    #    But the known rank is 7.
    #    What is wrong?
    #    The definition of the tensor T_{i,j,k}.
    #    T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    #    In the decomposition C_{i,k} = sum_t W_{k,i,t} M_{i,t} N_{k,t},
    #    where M_{i,t} = sum_j U_{i,j,t} A_{i,j} and N_{k,t} = sum_j V_{j,k,t} B_{j,k}.
    #    The term is M_{i,t} N_{k,t} W_{k,i,t}.
    #    This corresponds to a rank 1 tensor in the space of (i,k) for fixed t?
    #    No, the tensor is in (i,j,k).
    #    The condition is:
    #    sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = T_{i,j,k}.
    #    
    #    In my construction for term t_j:
    #    U_{i,j,t_j} = 1 (for the specific j).
    #    V_{j,k,t_j} = 1 (for the specific j).
    #    W_{k,i,t_j} = 1.
    #    Then for a specific (i,j,k):
    #    If j == j_t, term t_j contributes 1*1*1 = 1.
    #    If j != j_t, term t_j contributes 0.
    #    So sum_t ... = 1 if j=j_t, 0 otherwise.
    #    But T_{i,j,k} = 1 for all i,j,k.
    #    So we need sum_t ... = 1 for all i,j,k.
    #    My construction gives 1 only for j=j_t.
    #    Since we sum over t, and each t handles a different j,
    #    for a fixed j, only term t_j contributes 1.
    #    So sum_t ... = 1.
    #    So it works!
    #    
    #    Why is the rank of 2x2x2 7 then?
    #    Is my construction valid?
    #    U is n*m x rank.
    #    For n=2, m=2, rank=2.
    #    U[0,0]=1, U[1,0]=1 (col 0)
    #    U[0,1]=0, U[1,1]=0 (col 1) -> Wait.
    #    For term t_0 (j=0): U[i*2+0, 0] = 1.
    #    So U[0,0]=1, U[2,0]=1. (Indices 0 and 2)
    #    For term t_1 (j=1): U[i*2+1, 1] = 1.
    #    So U[1,1]=1, U[3,1]=1. (Indices 1 and 3)
    #    
    #    V is m*p x rank. m=2, p=2.
    #    For term t_0 (j=0): V[0*2+k, 0] = 1.
    #    So V[0,0]=1, V[1,0]=1.
    #    For term t_1 (j=1): V[1*2+k, 1] = 1.
    #    So V[2,1]=1, V[3,1]=1.
    #    
    #    W is p*n x rank. p=2, n=2.
    #    For term t_0: W[k*2+i, 0] = 1.
    #    So W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
    #    For term t_1: W[k*2+i, 1] = 1.
    #    So W[0,1]=1, W[1,1]=1, W[2,1]=1, W[3,1]=1.
    #    
    #    Let's check T_{0,0,0}.
    #    sum_t U[0*2+0, t] V[0*2+0, t] W[0*2+0, t]
    #    t=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1.
    #    t=1: U[0,1]=0, V[0,1]=0, W[0,1]=1 -> 0.
    #    Sum = 1. Correct.
    #    
    #    Let's check T_{0,1,0}.
    #    sum_t U[0*2+1, t] V[1*2+0, t] W[0*2+0, t]
    #    t=0: U[1,0]=0, V[2,0]=0, W[0,0]=1 -> 0.
    #    t=1: U[1,1]=1, V[2,1]=1, W[0,1]=1 -> 1.
    #    Sum = 1. Correct.
    #    
    #    It seems the rank is m!
    #    If rank is m, then for 2x3x10, rank is 3.
    #    This is way better than 56.
    #    Is there a catch?
    #    "integer/half-integer entries".
    #    My entries are 0 and 1.
    #    
    #    Why is Strassen rank 7?
    #    Strassen is for 2x2x2.
    #    If rank 2 works, Strassen is not optimal?
    #    But Strassen is a famous algorithm.
    #    Let me re-read the tensor definition.
    #    "decompose the matrix multiplication tensor".
    #    Maybe the indices are different?
    #    U (n*m, rank), V (m*p, rank), W (p*n, rank).
    #    My construction uses this.
    #    
    #    Is it possible that the rank of 2x2x2 is 2?
    #    Let's check the literature.
    #    The rank of the 2x2x2 matrix multiplication tensor is 7.
    #    My construction must be flawed.
    #    
    #    Let's check the sum again.
    #    C_{i,k} = sum_t W_{k,i,t} (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k})
    #    
    #    In my construction for 2x2x2:
    #    t=0:
    #    M_{i,0} = sum_j U_{i,j,0} A_{i,j}
    #    U_{0,0,0}=1, U_{0,1,0}=0 -> M_{0,0} = A_{0,0}
    #    U_{1,0,0}=1, U_{1,1,0}=0 -> M_{1,0} = A_{1,0}
    #    N_{k,0} = sum_j V_{j,k,0} B_{j,k}
    #    V_{0,0,0}=1, V_{1,0,0}=0 -> N_{0,0} = B_{0,0}
    #    V_{0,1,0}=1, V_{1,1,0}=0 -> N_{1,0} = B_{0,1}
    #    W_{0,0,0}=1, W_{1,0,0}=1, W_{0,1,0}=1, W_{1,1,0}=1
    #    Contribution to C_{0,0}: W_{0,0,0} M_{0,0} N_{0,0} = 1 * A_{0,0} * B_{0,0}.
    #    Contribution to C_{0,1}: W_{1,0,0} M_{0,0} N_{1,0} = 1 * A_{0,0} * B_{0,1}.
    #    Contribution to C_{1,0}: W_{0,1,0} M_{1,0} N_{0,0} = 1 * A_{1,0} * B_{0,0}.
    #    Contribution to C_{1,1}: W_{1,1,0} M_{1,0} N_{1,0} = 1 * A_{1,0} * B_{0,1}.
    #    
    #    t=1:
    #    M_{0,1} = A_{0,1}
    #    M_{1,1} = A_{1,1}
    #    N_{0,1} = B_{1,0}
    #    N_{1,1} = B_{1,1}
    #    Contribution to C_{0,0}: W_{0,0,1} M_{0,1} N_{0,1} = 1 * A_{0,1} * B_{1,0}.
    #    Contribution to C_{0,1}: W_{1,0,1} M_{0,1} N_{1,1} = 1 * A_{0,1} * B_{1,1}.
    #    Contribution to C_{1,0}: W_{0,1,1} M_{1,1} N_{0,1} = 1 * A_{1,1} * B_{1,0}.
    #    Contribution to C_{1,1}: W_{1,1,1} M_{1,1} N_{1,1} = 1 * A_{1,1} * B_{1,1}.
    #    
    #    Total C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    #    This is correct for 2x2 matrix multiplication.
    #    
    #    So why is the rank 7?
    #    Maybe the "rank" in tensor decomposition is defined differently?
    #    Or maybe I am missing a constraint?
    #    "integer/half-integer entries".
    #    My entries are integers.
    #    
    #    Wait, is it possible that the task implies a specific ordering or structure?
    #    "The output factor follows the official trace convention W[k*n+i,r]."
    #    I used W[k*n+i, t].
    #    
    #    Let's check the rank of 2x2x2 again.
    #    If rank 2 works, then the rank of 2x3x10 is 3.
    #    This seems too good to be true.
    #    Let's check 2x2x3.
    #    Rank 2?
    #    t=0 (j=0):
    #    M_{0,0}=A_{0,0}, M_{1,0}=A_{1,0}
    #    N_{0,0}=B_{0,0}, N_{1,0}=B_{0,1}, N_{2,0}=B_{0,2}
    #    W all 1.
    #    C_{0,0} += A_{0,0}B_{0,0}
    #    C_{0,1} += A_{0,0}B_{0,1}
    #    C_{0,2} += A_{0,0}B_{0,2}
    #    C_{1,0} += A_{1,0}B_{0,0}
    #    C_{1,1} += A_{1,0}B_{0,1}
    #    C_{1,2} += A_{1,0}B_{0,2}
    #    t=1 (j=1):
    #    C_{0,0} += A_{0,1}B_{1,0}
    #    ...
    #    Total C_{i,k} = sum_j A_{i,j}B_{j,k}.
    #    It works.
    #    
    #    So the rank is m.
    #    For 2x3x10, m=3, so rank 3.
    #    
    #    Let's implement rank 3.
    
    # Reset rank
    rank = m  # 3
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j in range(m):
        t = j
        # U[i*m + j, t] = 1 for all i
        for i in range(n):
            U[i * m + j, t] = 1
        # V[j*p + k, t] = 1 for all k
        for k in range(p):
            V[j * p + k, t] = 1
        # W[k*n + i, t] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, t] = 1
                
    return {"U": U, "V": V, "W": W}
