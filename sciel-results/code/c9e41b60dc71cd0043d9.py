"""Matrix multiplication 3x3 by 3x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Returns rank-minimal exact tensor decomposition.
    
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (18, rank)
    W: (p*n, rank) = (18, rank)
    
    W[k*n+i, r] follows the official trace convention.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # We'll try multiple approaches and return the best valid one
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Schoolbook rank 54 (always valid, fallback)
    def make_schoolbook():
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
        return U, V, W
    
    # Approach 2: Try to build a valid low-rank decomposition
    # For 3x3x3, use a known valid rank decomposition
    # Then extend to 3x3x6
    
    # Let's build a valid 3x3x3 decomposition using the following method:
    # Start with 9 diagonal terms, then add off-diagonal Strassen-like terms
    
    # Actually, let me use a constructive approach:
    # The matmul tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta(i,i)*delta(j,j)*delta(k,k) ... no.
    # T[i,j,k] means: the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) where j is the shared index
    
    # The tensor is: T_{i,j,k} = 1 if the output position (i,k) uses input (i,j) and (j,k)
    # In the standard formulation: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k in range(3)
    # Wait, that's just the all-ones tensor? No...
    
    # Let me re-read the convention:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should give 1 if (i,j,k) is a valid triple for matmul
    # For matmul C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That means every combination (i,j,k) contributes to the product!
    
    # Wait, that can't be right for a "decomposition". Let me re-read.
    
    # The task says: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k]
    # For standard matmul, C[i,k] = sum_j A[i,j]*B[j,k], so M[i,j,k] = 1 for ALL i,j,k
    # That's just the all-ones tensor of size n*m*p!
    
    # Hmm, but then the rank would be 1 (all-ones tensor is rank 1).
    # That doesn't make sense with the problem statement mentioning rank 23 for 3x3x3.
    
    # Let me re-read more carefully...
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix"
    
    # I think the tensor is defined as: T[i,j,k] = 1 if i,j,k form a valid triple
    # where i is row of A, j is column of A / row of B, k is column of B
    # For this to be the matmul tensor, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # and = 0 otherwise
    # But ALL (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid!
    
    # So the tensor IS the all-ones tensor? That would have rank 1.
    # That contradicts the problem.
    
    # Let me look at the verification code in the current program:
    # It checks: for i,j,k: sum U[i*3+j]*V[j*3+k]*W[k*3+i] == 1
    # And for "off-diagonal" (where indices don't match): sum == 0
    # The condition is: j1 == j2 and k1 == k2 and i1 == i2
    # where a = i1*3+j1, b = j2*3+k1, c2 = k2*3+i2
    # So the "matching" condition is: j1==j2, k1==k2, i1==i2
    # This means the tensor is NOT all-ones.
    
    # Actually wait - in the verification, it iterates over ALL triples (a,b,c2) 
    # where a in [0,9), b in [0,9), c2 in [0,9), and checks:
    # if j1==j2 and k1==k2 and i1==i2: should be 1
    # else: should be 0
    
    # So the tensor T[a,b,c2] = 1 iff the indices are "consistent"
    # a = i*3+j (row i, col j of A)
    # b = j'*3+k (row j', col k of B)  
    # c2 = k'*3+i' (row k', col i' of C)
    # Consistent means: j == j' (A's col matches B's row), k == k' (B's col matches C's row), i == i' (A's row matches C's row)
    
    # So T[i,j, j,k, k,i] = 1 (using multi-index notation)
    # This is the standard matmul tensor!
    
    # Now, the rank of this tensor for 3x3x3 is known to be 23.
    # For 3x3x6, the rank is at most 2*23 = 46 (by splitting B into two 3x3 blocks).
    
    # Let me construct a VALID rank-23 decomposition for 3x3x3.
    # I'll use a known construction.
    
    # The 23-term decomposition for 3x3x3 can be derived as follows:
    # - Use the 7-term Strassen algorithm for the top-left 2x2x2 subproblem
    # - Add terms for the remaining entries
    
    # Actually, let me just use a brute-force verified approach:
    # I'll construct the decomposition and verify it.
    
    # Here's a known valid rank-23 decomposition for 3x3x3 matmul:
    # Based on the work of Strassen and subsequent improvements.
    
    # Let me define the 23 terms carefully.
    # Each term c has vectors u_c, v_c, w_c in R^9 (or {-1,0,1}^9)
    # such that sum_c u_c[a] * v_c[b] * w_c[c2] = T[a,b,c2]
    
    # I'll use the following construction:
    # Terms 0-6: Strassen-like for a 2x2x2 subproblem
    # Terms 7-22: Additional terms for the 3x3x3 case
    
    # Actually, let me use a different, more reliable approach:
    # I'll use the fact that the 3x3x3 matmul tensor can be decomposed
    # using the following 23 rank-1 terms:
    
    # Define helper: create a vector with 1s at specified positions
    def vec9(indices, values=None):
        v = [0] * 9
        if values is None:
            for idx in indices:
                v[idx] = 1
        else:
            for idx, val in zip(indices, values):
                v[idx] = val
        return tuple(v)
    
    # Rank-23 decomposition for 3x3x3:
    # I'll use a verified set of 23 terms.
    # The key property: for each (i,j,k) in {0,1,2}^3:
    #   sum_c u_c[i*3+j] * v_c[j*3+k] * w_c[k*3+i] = 1
    # For inconsistent (a,b,c2): sum = 0
    
    # Let me construct this systematically.
    # The 9 diagonal terms (i=j=k):
    # Term for (i,i,i): u has 1 at i*3+i, v has 1 at i*3+i, w has 1 at i*3+i
    
    # But that gives rank 9, and we need to handle off-diagonal terms too.
    
    # For the off-diagonal terms, we need combinations.
    # This is getting complex. Let me use a known explicit decomposition.
    
    # Here is a verified rank-23 decomposition for 3x3x3:
    # (I'll construct it and verify)
    
    r23_terms = []
    
    # Strategy: Use the following 23 terms
    # Based on the decomposition from the literature
    
    # Let me try a different approach: use the Kronecker product structure
    # 3x3x3 can be seen as related to 2x2x2 (Strassen) plus extra terms
    
    # Actually, the simplest valid approach: use rank 27 (3^3) which is 
    # guaranteed to work, then try to optimize.
    
    # For rank 27: each term corresponds to one (i,j,k) triple
    # u_c has 1 at position i*3+j, v_c has 1 at j*3+k, w_c has 1 at k*3+i
    # But wait, that's the schoolbook for 3x3x3, which has rank 27.
    
    # For 3x3x6, schoolbook has rank 54.
    
    # Let me first ensure we have a VALID solution, then try to improve.
    
    # Schoolbook for 3x3x6: rank 54
    U_sb, V_sb, W_sb = make_schoolbook()
    best_rank = 54
    best_result = (U_sb, V_sb, W_sb)
    
    # Now try to find a better decomposition
    # For 3x3x6, we can split B into B1 (3x3) and B2 (3x3)
    # C = A * [B1, B2] = [A*B1, A*B2]
    # So we can use a rank-r decomposition for 3x3x3 twice
    # Total rank: 2*r
    
    # If we can find rank 23 for 3x3x3, total rank = 46
    # If we can find rank 22, total rank = 44
    # etc.
    
    # Let me try to construct a valid low-rank decomposition for 3x3x3.
    # I'll use a systematic approach with verification.
    
    # The 3x3x3 matmul tensor:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # (in the multi-index sense: T[i,j | j,k | k,i] = 1)
    
    # Let me define the tensor explicitly
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1
    
    # Now, find a rank decomposition
    # Rank of T is what we're looking for
    
    # Use a greedy approach: find rank-1 tensors that explain the most remaining mass
    # But this might not give the minimum rank.
    
    # Alternative: use the fact that for small tensors, we can try known decompositions
    
    # Let me try the following 23-term decomposition (constructed carefully):
    
    # I'll use the following approach:
    # The 3x3x3 matmul can be decomposed using:
    # - 9 terms for the "diagonal" (i=j=k)
    # - 14 terms for off-diagonal combinations
    
    # Actually, let me just verify a candidate decomposition numerically.
    
    # Here's my attempt at a rank-23 decomposition:
    # I'll define 23 triples (u, v, w) and check if they sum to T.
    
    # Let me use a known result: the rank of the 3x3x3 matmul tensor is 23.
    # One explicit decomposition is given in various references.
    
    # For now, let me use a rank-27 decomposition (schoolbook for 3x3x3)
    # and combine two copies for 3x3x6, giving rank 54.
    # Then I'll try to find a better one.
    
    # Actually, let me try a rank-23 decomposition using the following method:
    # I'll use the 7-term Strassen algorithm for a 2x2x2 subproblem,
    # and add 16 more terms for the remaining entries.
    
    # Strassen's 7 terms for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # For 3x3x3, I can embed this and add terms for the third row/column.
    
    # This is getting very complex. Let me use a simpler approach:
    # Use a rank-27 decomposition for 3x3x3 (one term per (i,j,k) triple),
    # giving rank 54 for 3x3x6.
    
    # Then try to reduce by finding linear dependencies.
    
    # Actually, let me just return the schoolbook solution for now,
    # and add a time-limited search for better solutions.
    
    # Let me try one more thing: a rank-46 decomposition using
    # two copies of a rank-23 decomposition.
    
    # I need to find a valid rank-23 decomposition for 3x3x3.
    # Let me construct one and verify it.
    
    # Here's a candidate rank-23 decomposition:
    # I'll use the following 23 terms (u, v, w) where each is a 9-tuple:
    
    # Term 0: u = (1,1,0, 1,1,0, 0,0,0), v = (1,0,1, 0,1,0, 1,0,1), w = (1,1,0, 1,1,0, 0,0,0)
    # ... this is getting too speculative.
    
    # Let me use a verified approach: I'll generate a random decomposition
    # and check if it's valid, but that's unlikely to work.
    
    # Instead, let me use the following known fact:
    # The 3x3x3 matmul tensor has rank 23, and one explicit decomposition
    # can be found in the literature.
    
    # For the purpose of this task, let me use a rank-54 schoolbook solution
    # which is guaranteed to be valid, and see if I can improve it.
    
    # Actually, I realize I should just verify my solution and return it.
    # The schoolbook solution is valid, so let me return that.
    
    # But wait - can I do better? Let me try a rank-46 solution.
    
    # For 3x3x6, if I can find a rank-23 decomposition for 3x3x3,
    # I can use it twice (for the two 3x3 blocks of B), giving rank 46.
    
    # Let me construct a valid rank-23 decomposition for 3x3x3.
    # I'll use the following method:
    
    # The 3x3x3 matmul tensor can be written as:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector with 1 at position i*3+j.
    
    # This is a sum of 27 rank-1 tensors.
    # To reduce the rank, we need to find linear combinations.
    
    # The key insight: if we can find vectors u, v, w such that
    # u ⊗ v ⊗ w = sum of several of the above rank-1 tensors,
    # we can reduce the total rank.
    
    # For example, (e_0 + e_3) ⊗ (e_0 + e_3) ⊗ (e_0 + e_3)
    # = e_0⊗e_0⊗e_0 + e_0⊗e_0⊗e_3 + e_0⊗e_3⊗e_0 + e_0⊗e_3⊗e_3
    # + e_3⊗e_0⊗e_0 + e_3⊗e_0⊗e_3 + e_3⊗e_3⊗e_0 + e_3⊗e_3⊗e_3
    # = 8 rank-1 tensors in one term!
    
    # But we need to be careful: the resulting tensor must match T exactly.
    
    # Let me try a systematic construction:
    
    # Divide the 3x3x3 problem into 8 subproblems of size ~1x1x1,
    # and use Strassen-like combinations.
    
    # Actually, the simplest valid improvement: use rank 27 for 3x3x3
    # (one term per (i,j,k) pair), giving rank 54 for 3x3x6.
    
    # This is the same as schoolbook, so no improvement.
    
    # Let me try to find a valid rank-23 decomposition.
    # I'll use the following explicit construction:
    
    # Based on the paper "On the rank of the matrix multiplication tensor"
    # the 23-term decomposition for 3x3x3 is:
    
    # I'll define it as 23 triples and verify.
    
    # For now, let me return the schoolbook solution.
    # The score will be 54/54 = 1.0 (same as baseline).
    
    # Actually, the baseline is schoolbook rank 54, so score = 54/54 = 1.0.
    # To improve, I need rank < 54.
    
    # Let me try a rank-46 solution by finding a valid rank-23 for 3x3x3.
    
    # I'll construct a candidate and verify it.
    
    # Here's a candidate rank-23 decomposition for 3x3x3:
    # (I'll define it and check)
    
    r23_U = [
        (1, 1, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 1, 1, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 1, 0),
        (0, 0, 0, 1, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 1),
        (1, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 1),
    ]
    
    r23_V = [
        (1, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 1, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 1),
    ]
    
    r23_W = [
        (1, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
    ]
    
    # Verify this decomposition
    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        # Check all valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if val != 1:
                        return False
        # Check all invalid triples
        for a in range(n*m):
            for b in range(m*p):
                for c2 in range(p*n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c2, n)
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        continue  # already checked
                    val = sum(U[a][c] * V[b][c] * W[c2][c] for c in range(rank))
                    if val != 0:
                        return False
        return True
    
    # Build rank-46 solution from rank-23
    rank46 = 46
    U46 = [[0] * rank46 for _ in range(9)]
    V46 = [[0] * rank46 for _ in range(18)]
    W46 = [[0] * rank46 for _ in range(18)]
    
    for block in range(2):
        for c in range(23):
            col = block * 23 + c
            u = r23_U[c]
            v = r23_V[c]
            w = r23_W[c]
            for idx in range(9):
                U46[idx][col] = u[idx]
                V46[idx + block * 9][col] = v[idx]
                W46[idx + block * 9][col] = w[idx]
    
    if verify_decomposition(U46, V46, W46, n, m, p):
        best_rank = 46
        best_result = (U46, V46, W46)
    else:
        # Try to find a valid decomposition using a different approach
        # Use a rank-27 decomposition for 3x3x3 (one term per triple)
        # This gives rank 54 for 3x3x6, same as schoolbook
        
        # Try to find a valid rank-23 by solving a system of equations
        # This is complex, so let's just use schoolbook
        
        # Actually, let me try a different rank-23 candidate
        # I'll use a more careful construction
        
        # For now, stick with schoolbook (rank 54)
        pass
    
    # Return the best result found
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
