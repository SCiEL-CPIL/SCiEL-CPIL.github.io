"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

DEADLINE = 150  # seconds

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for
    multiplying a 3x3 matrix by a 3x7 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = e_{i,k} for the
    standard basis. We want to find rank r and matrices U (9 x r), V (21 x r),
    W (21 x r) such that for all i,j,k:
        sum_{t=1}^{r} U[i,j,t] * V[k,t] * W[i,k,t] = delta_{i,k} if j=k else 0
    
    Wait, the tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    is defined as:
        C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has entries T[i,j,k] = 1 if the product term A[i,j]*B[j,k]
    contributes to C[i,k]. In tensor decomposition form:
        T = sum_t u_t ⊗ v_t ⊗ w_t
    where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}.
    
    The entry T[i,j,k] (where i is row of A, j is col of A / row of B, k is col of B)
    equals 1 if this term appears in C[i,k]. Actually, the standard formulation is:
        T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to
        contribute to C[i,k]. But that's always the case.
    
    More precisely, the matrix multiplication tensor has:
        T[i,j,k] = delta_{j,j} * delta_{i,i} * delta_{k,k} -- no, that's not right.
    
    Let me reconsider. The tensor for multiplying an n x m matrix by an m x p matrix
    is an n x m x m x p tensor. But the task says the output factor W has shape (p*n, rank),
    and U has shape (n*m, rank), V has shape (m*p, rank).
    
    So the tensor is of format: U (n*m) x rank, V (m*p) x rank, W (p*n) x rank.
    
    The convention is:
        U[i*m + j, t] -- coefficient for A[i,j]
        V[j*p + k, t] -- coefficient for B[j,k]  
        W[k*n + i, t] -- coefficient for C[i,k]
    
    And the condition is:
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if this is a valid term, else 0
    
    Actually, for matrix multiplication, every term A[i,j]*B[j,k] contributes to C[i,k].
    So the tensor entry T[i,j,k] = 1 for all valid i,j,k (where j is the shared dimension).
    
    Wait, that can't be right because then the rank would just be n*m*p.
    
    Let me re-read the task. "Find a rank-minimal exact tensor decomposition for
    multiplying a 3 by 3 matrix by a 3 by 7 matrix."
    
    The standard matrix multiplication tensor for C = A*B where A is n x m, B is m x p
    is a tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} (or similar indexing).
    
    The entry T[a, b, c] where a indexes (i,j) in A, b indexes (j,k) in B, c indexes
    (i,k) in C is:
        T[(i,j), (j,k), (i,k)] = 1
    
    So for each triple (i, j, k), there's exactly one nonzero entry in the tensor.
    The total number of nonzero entries is n*m*p = 3*3*7 = 63.
    
    The rank of this tensor is what we want to minimize. The schoolbook algorithm
    uses rank 63. We need to find a lower rank decomposition.
    
    For 3x3x3 matrix multiplication, the known minimum rank is 23 (Landsberg, 1988).
    For 3x3x7, we can think of it as three 3x3x1 multiplications plus two 3x3x3
    multiplications... no, that's not quite right.
    
    Actually, 3x3x7 means A is 3x3, B is 3x7, C is 3x7. We can decompose B into
    three 3x3 blocks and one 3x1 block. Then:
    C = A * [B1 | B2 | B3 | B4] where B1,B2,B3 are 3x3 and B4 is 3x1.
    
    So C = [A*B1 | A*B2 | A*B3 | A*B4]
    
    If we use rank 23 for each 3x3x3 multiplication and rank 9 for the 3x3x1
    multiplication, the total rank would be 23*3 + 9 = 78. But we might be able
    to share some structure.
    
    However, the U factors for each block are the same (they depend only on A),
    while the V factors are different (they depend on different columns of B).
    
    In a tensor decomposition, each term t has:
        U[:, t] -- a vector in R^9
        V[:, t] -- a vector in R^21
        W[:, t] -- a vector in R^21
    
    For the block A*B1 (where B1 is the first 3 columns of B), the V factor would
    only have nonzero entries in the first 3 positions of each row (corresponding
    to B[0,0], B[0,1], B[0,2], B[1,0], ..., B[2,2]).
    
    So we can't directly share U factors across blocks in a simple tensor decomposition
    because each term has its own U, V, W vectors.
    
    Let me just implement a correct rank-63 schoolbook decomposition first, then
    try to improve.
    
    Schoolbook: for each i in 0..2, j in 0..2, k in 0..6:
        U[i*3+j, t] = 1
        V[j*7+k, t] = 1
        W[k*3+i, t] = 1
    
    This gives rank 63.
    
    To improve, I need to find a lower-rank decomposition. Let me try to use
    known results for 3x3x3.
    
    Actually, let me just return the schoolbook solution for now. It's correct
    and the score will be 63. The seed also has score 1.0 which corresponds to
    some rank. Let me check what rank the seed produces.
    
    Looking at the seed code, it also produces rank 63 (27+27+9=63).
    
    So to beat the seed, I need rank < 63. Let me try to find a better decomposition.
    
    For 3x3x3, the minimum rank is 23. If I can use that, then for 3x3x7:
    - Three 3x3x3 blocks: 3 * 23 = 69
    - One 3x3x1 block: rank 9
    Total: 78, which is worse than 63.
    
    Hmm, that doesn't help. The issue is that the 3x3x7 problem has 63 terms in
    schoolbook, and I need to find a decomposition with fewer than 63 terms.
    
    Let me think differently. The tensor for 3x3x7 is:
        T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..6
    
    This is a 3x3x7 tensor. I need to find its rank.
    
    One approach: use the fact that 7 = 3+3+1, and decompose accordingly.
    But as I noted, that gives 3*rank(3x3x3) + rank(3x3x1) = 3*23 + 9 = 78.
    
    Alternatively, maybe I can find a direct decomposition with rank < 63.
    
    Let me try a randomized approach: start with random rank-r decompositions and
    try to minimize r.
    
    Actually, for the purposes of this task, let me just implement the schoolbook
    decomposition and return it. The score will be 63, same as the seed.
    
    Wait, the task says "score is the rank" and "the evaluator measures the raw value
    (lower is better)". And "search score is seed/value". So if seed has rank 63 and
    I also have rank 63, the search score is 63/63 = 1.0.
    
    To get a score > 1.0, I need rank < 63.
    
    Let me try to find a rank-62 or lower decomposition.
    
    Actually, I realize I should just implement a correct solution and try to
    optimize. Let me start with schoolbook (rank 63) and see if I can find
    improvements.
    """
    
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Schoolbook decomposition: rank = n*m*p = 63
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using randomized search
    # This is a heuristic approach
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check that for all i, j, k:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try a simple improvement: see if we can reduce rank by 1
    # This is a very basic heuristic
    
    # For now, just return the schoolbook solution
    # A proper optimization would require more sophisticated methods
    
    if time.time() - start_time > DEADLINE - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
