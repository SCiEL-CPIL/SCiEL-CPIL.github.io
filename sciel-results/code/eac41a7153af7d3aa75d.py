"""
Matrix multiplication 2x2 * 2x13: rank-minimal exact tensor decomposition.

Strategy:
  1. Start with schoolbook rank = 2*2*13 = 52.
  2. Apply known rank-reduction techniques:
     - Strassen-style reductions on the 2x2 block structure for each column
     - Exploit the cyclic structure of the 13 columns
  3. Use integer linear algebra to find a minimal-rank decomposition.
  4. Validate that the tensor identity holds exactly.

The key insight: A (2x2) times B (2x13) produces C (2x13).
C[i,k] = sum_j A[i,j] * B[j,k]

For each column k of C, we need to compute a 2x2 matrix product of A and the k-th column of B.
Using Strassen's algorithm, each 2x2 product can be done with 7 multiplications instead of 4.

However, the 13 columns share the same A, so we can potentially do better by
treating B as a 2x13 matrix and finding a decomposition that exploits the
structure across all 13 columns simultaneously.

Known results:
  - The tensor rank of matrix multiplication <2,2,2> is 7 (Strassen).
  - For <2,2,p>, the rank is bounded. For p=13, we aim to find the minimum.

We'll construct a decomposition using:
  - 7 multiplications for the "core" 2x2 structure per group of columns
  - Careful bookkeeping of the 13 columns

Actually, let me think more carefully. The multiplication tensor for (n,m,p) = (2,2,13)
has 2*2*13 = 52 trivially. But we can do better.

The matrix multiplication tensor T = sum_{i,j,k} x_i y_j z_k * E_{i,j,k}
where i in [0,1], j in [0,1], k in [0,12].

A rank-r decomposition is:
  T = sum_{t=0}^{r-1} u_t * v_t * w_t
where u_t is in R^{nm}, v_t in R^{mp}, w_t in R^{pn}.

U[i*m + j, t] = u_t(i,j)
V[j*p + k, t] = v_t(j,k)
W[k*n + i, t] = w_t(k,i)

The identity: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 if (i,j,k) is a valid index
Wait, that's not right. The tensor for matrix multiplication (A*B) where A is n×m, B is m×p:
C[i,k] = sum_j A[i,j] * B[j,k]

So the tensor is: sum_{i,j,k} A_i_j * B_j_k * C_i_k

In our factorization:
C_i_k = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]

Wait, that's not the standard form. Let me re-read the task.

"U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
"The output factor follows the official trace convention W[k*n+i,r]"

So the decomposition is:
C[i,k] = sum_{t} sum_{j} U[i*m+j, t] * W[k*n+i, t] * V[j*p+k, t] * A[i,j] * B[j,k]

Hmm, that doesn't look right either. Let me think about the standard matrix multiplication tensor.

The standard decomposition for matrix multiplication <n,m,p> is:
C = A * B where A is n×m, B is m×p.
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is: T = sum_{i,j,k} E_{i,j}^{(A)} * E_{j,k}^{(B)} * E_{i,k}^{(C)}

A rank-1 decomposition would be: u * v * w where:
u is in the A-space (n*m entries), v in B-space (m*p entries), w in C-space (p*n entries).

The condition for a valid decomposition of rank r is:
For all i, j, k:
  sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
  if the (i,j,k) triple contributes to C[i,k], which it does for all i,j,k where j is the summation index.

Wait, I need to be more careful. The matrix multiplication tensor for A (n×m) × B (m×p) = C (n×p) is:

T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} a_{i,j} * b_{j,k} * c_{i,k}

as a multilinear form. The rank is the minimum number of rank-1 tensors to sum to T.

A rank-1 tensor is: u * v * w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{pn} (with w indexed as w[k*n+i]).

The condition is:
sum_{t} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1  for all i,j,k

This is because a_{i,j} * b_{j,k} * c_{i,k} is the monomial, and we need the coefficient to be 1.

So the schoolbook decomposition sets for each (i,j,k):
U[i*m+j, (i,j,k)] = 1
V[j*p+k, (i,j,k)] = 1
W[k*n+i, (i,j,k)] = 1
and zeros elsewhere. This gives rank = n*m*p = 52.

To reduce rank, we need to find a set of r triples (u_t, v_t, w_t) such that:
sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i ∈ [0,1], j ∈ [0,1], k ∈ [0,12]

This is a system of 52 equations (one for each (i,j,k) triple) in r*3 unknowns (entries of U, V, W).

For r < 52, this is an underdetermined system, and we need to find integer or half-integer solutions.

Let me try a different approach. I'll set up the problem as finding a rank-r decomposition by solving a system of polynomial equations, or by using known constructions.

Known result: The tensor rank of <2,2,p> is 2p + 3 for p ≥ 2? No, that's not right either.

Actually, for <2,2,p>, we can think of it as p copies of <2,2,1> sharing the same A.
<2,2,1> has rank 4 (schoolbook) but can be done in... well, it's just 4 scalar multiplications, and I don't think there's a known reduction for a single column.

Wait, <2,2,1> is just computing C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] for i=0,1. That's 4 scalar multiplications, and the tensor rank is 4.

For <2,2,p>, we can use Strassen's algorithm if we group the columns. But Strassen's algorithm applies to square matrices.

Let me try a direct approach: set up the equations and solve them.

Actually, let me just try to find a decomposition of rank 7 + 13 - something. 

Hmm, let me try a different approach. I'll use the fact that the 13 columns of B can be treated as elements of a polynomial ring, and use the structure of that ring to find a lower-rank decomposition.

But for now, let me just implement the schoolbook decomposition and verify it works, then try to improve it.

Given the time constraints, let me implement a solution that:
1. Starts with schoolbook rank 52.
2. Tries to reduce the rank by finding linear dependencies.
3. Returns the best decomposition found.

Actually, let me just verify the schoolbook decomposition is correct and return it. The task says "rank-minimal exact tensor decomposition", but if I can't find a better one in the time limit, I'll return the schoolbook one.

Let me implement a more clever approach: I'll use the fact that for each pair (i,j), the term A[i,j]*B[j,k] appears in C[i,k] for all k. So I can factor out the dependence on k.

For fixed i, j: the contribution to C[i,k] for all k is A[i,j] * B[j,k]. This is a rank-1 matrix (as a function of k) for fixed i,j.

So the total C = sum_{i,j} A[i,j] * (B[j,:] * e_i^T) where e_i is the i-th standard basis vector.

This suggests a decomposition where:
- For each (i,j), we have one "factor" that multiplies A[i,j] by the j-th row of B and puts it in the i-th row of C.

This is exactly the schoolbook decomposition, giving rank = n*m*p = 52.

To do better, we need to find linear combinations that share multiplications. This is essentially what Strassen's algorithm does for the 2x2 case.

Let me try to implement a Strassen-like decomposition for the 2x2 x 2x13 case.

I'll group the 13 columns into groups and apply Strassen's algorithm within each group, but since the groups overlap (Strassen's algorithm uses sums/differences), this gets complicated.

For now, let me just return the schoolbook decomposition and ensure it's correct.
"""
import time
import itertools

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def validate(decomp):
        """Check that the decomposition is valid."""
        U = decomp["U"]
        V = decomp["V"]
        W = decomp["W"]
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True
    
    best = build_schoolbook()
    if not validate(best):
        # Should not happen, but just in case
        raise RuntimeError("Schoolbook decomposition failed validation")
    
    best_rank = len(best["U"][0])
    
    # Try to find a lower-rank decomposition by searching over smaller ranks
    # This is a heuristic search; we try ranks from best_rank-1 down
    # For each rank, we attempt to solve the system of equations
    
    # The system: for each (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # This is a system of n*m*p = 52 equations.
    
    # We can fix W to be the identity-like matrix and solve for U, V, but that's not straightforward.
    
    # Alternative: try known constructions
    # For <2,2,p>, one known construction uses rank = 2*p + 1 = 27? Let me check.
    
    # Actually, I recall that for matrix multiplication <n,m,p>, the rank is at most n*m*p,
    # and Strassen showed <2,2,2> has rank 7 < 8.
    
    # For <2,2,p>, we can tile p/2 blocks of <2,2,2> and use 7 per block, giving ~3.5*p.
    # For p=13: 6 full blocks of 2 + 1 leftover = 6*7 + 4 = 46? Not great.
    
    # Let me try a different approach: use the polynomial representation.
    # Treat B as a 2x13 matrix, which can be seen as two polynomials of degree 12.
    # A[i,j] * B[j,k] for k=0..12.
    
    # The product C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a linear combination of the two rows of B.
    
    # I'll try to construct a decomposition with rank 7*p or similar.
    
    # For now, return the schoolbook decomposition as the best found so far.
    # If time permits, try to improve.
    
    if time.time() > deadline - 30:
        return best
    
    # Try a Strassen-based decomposition for the 2x2 part, replicated per column group
    # Strassen for 2x2: 7 multiplications
    # If we apply this to each column independently, we still need 4 per column, so no gain.
    # The gain comes from sharing multiplications across columns.
    
    # Let's try: treat B as a 2x13 matrix. Partition into ceil(13/2) = 7 groups of size <= 2.
    # For each group of 2 columns, use Strassen (7 mults). For the last group of 1 column, use 4 mults.
    # Total: 6*7 + 4 = 46. Better than 52!
    
    # Strassen's 7 multiplications for computing C = A * B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our case, A is 2x2 (shared), B is 2x13.
    # We can group the 13 columns of B into pairs: (B[:,0],B[:,1]), (B[:,2],B[:,3]), ..., (B[:,12])
    # That's 6 pairs + 1 single = 7 groups.
    
    # For each pair (B[:,2g], B[:,2g+1]) for g=0..5:
    #   Use Strassen with A fixed, B_g = [B[:,2g] | B[:,2g+1]]
    #   7 multiplications: M1_g, ..., M7_g
    #   C[:,2g] = [M1_g + M4_g - M5_g + M7_g, M2_g + M4_g]
    #   C[:,2g+1] = [M3_g + M5_g, M1_g - M2_g + M3_g + M6_g]
    
    # For the single column B[:,12]:
    #   C[:,12] = A * B[:,12] = [A00*B0,12 + A01*B1,12, A10*B0,12 + A11*B1,12]
    #   4 multiplications.
    
    # Total rank: 6*7 + 4 = 46.
    
    # Let me implement this.
    
    def build_strassen_grouped():
        rank = 6 * 7 + 4  # 46
        U = [[0] * rank for _ in range(n * m)]  # 4 x 46
        V = [[0] * rank for _ in range(m * p)]  # 26 x 46
        W = [[0] * rank for _ in range(p * n)]  # 26 x 46
        
        col = 0
        # Process 6 pairs of columns
        for g in range(6):
            k0 = 2 * g
            k1 = 2 * g + 1
            # M1 = (A00+A11)(B0,k0+B0,k1) -> U has (A00+A11) in row 0 (i=0,j=0) and row 3 (i=1,j=1)
            # Actually, U[i*m+j, t] is the coefficient for A[i,j]
            # So for M1, we need U[0, t] += 1 (for A[0,0]), U[3, t] += 1 (for A[1,1])
            # V[j*p+k, t] is the coefficient for B[j,k]
            # For M1, V[0*p+k0, t] += 1, V[0*p+k1, t] += 1 (for B[0,k0]+B[0,k1])
            # W[k*n+i, t] is the coefficient for C[i,k]
            # M1 contributes to C00 (k=k0, i=0) and C11 (k=k1, i=1)
            # W[k0*n+0, t] += 1, W[k1*n+1, t] += 1
            
            # M1
            U[0][col] += 1   # A[0,0]
            U[3][col] += 1   # A[1,1]
            V[0 * p + k0][col] += 1  # B[0,k0]
            V[0 * p + k1][col] += 1  # B[0,k1]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1
            
            # M2 = (A10+A11)B0,k0
            U[1][col] += 1   # A[1,0]
            U[3][col] += 1   # A[1,1]
            V[0 * p + k0][col] += 1  # B[0,k0]
            W[k0 * n + 1][col] += 1  # C[1,k0]
            col += 1
            
            # M3 = A00(B1,k0 - B1,k1)
            U[0][col] += 1   # A[0,0]
            V[1 * p + k0][col] += 1  # B[1,k0]
            V[1 * p + k1][col] -= 1  # B[1,k1]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            W[k1 * n + 0][col] += 1  # C[0,k1]
            col += 1
            
            # M4 = A11(B0,k1 - B0,k0)
            U[3][col] += 1   # A[1,1]
            V[0 * p + k1][col] += 1  # B[0,k1]
            V[0 * p + k0][col] -= 1  # B[0,k0]
            W[k0 * n + 0][col] += 1  # C[0,k0]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1
            
            # M5 = (A00+A01)B1,k1
            U[0][col] += 1   # A[0,0]
            U[1][col] += 1   # A[0,1]
            V[1 * p + k1][col] += 1  # B[1,k1]
            W[k1 * n + 0][col] += 1  # C[0,k1]
            col += 1
            
            # M6 = (A10-A00)(B0,k0+B1,k0)
            U[1][col] += 1   # A[1,0]
            U[0][col] -= 1   # A[0,0]
            V[0 * p + k0][col] += 1  # B[0,k0]
            V[1 * p + k0][col] += 1  # B[1,k0]
            W[k0 * n + 1][col] += 1  # C[1,k0]
            col += 1
            
            # M7 = (A01-A11)(B0,k1+B1,k1)
            U[1][col] += 1   # A[0,1]
            U[3][col] -= 1   # A[1,1]
            V[0 * p + k1][col] += 1  # B[0,k1]
            V[1 * p + k1][col] += 1  # B[1,k1]
            W[k1 * n + 1][col] += 1  # C[1,k1]
            col += 1
        
        # Process the last single column (k=12)
        k = 12
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        U[0][col] += 1   # A[0,0]
        V[0 * p + k][col] += 1  # B[0,k]
        W[k * n + 0][col] += 1  # C[0,k]
        col += 1
        
        U[1][col] += 1   # A[0,1]
        V[1 * p + k][col] += 1  # B[1,k]
        W[k * n + 0][col] += 1  # C[0,k]
        col += 1
        
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        U[2][col] += 1   # A[1,0]
        V[0 * p + k][col] += 1  # B[0,k]
        W[k * n + 1][col] += 1  # C[1,k]
        col += 1
        
        U[3][col] += 1   # A[1,1]
        V[1 * p + k][col] += 1  # B[1,k]
        W[k * n + 1][col] += 1  # C[1,k]
        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    try:
        strassen_decomp = build_strassen_grouped()
        if validate(strassen_decomp):
            best = strassen_decomp
            best_rank = len(best["U"][0])
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return best
    
    # Try to find even lower rank by removing linearly dependent columns
    # Check if any column in U, V, W is zero (shouldn't be, but just in case)
    # More importantly, check if we can merge columns
    
    # For now, return the best decomposition found
    return best
# EVOLVE-BLOCK-END
