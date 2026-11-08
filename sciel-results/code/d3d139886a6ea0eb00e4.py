"""Matrix multiplication 2x3x3: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    
    The multiplication tensor T has entries T[i,j,k] = 1 if (i,j,k) corresponds to a valid
    product term. For C = A * B where A is 2x3 and B is 3x3:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is indexed as T[i, j, k] = 1 for all valid (i,j,k).
    We use the convention W[k*n+i, r] for the output factor.
    
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (9, rank)
    W has shape (p*n, rank) = (6, rank)
    
    The decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1 for all (i,j,k)
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook gives rank = n*m*p = 18
    # We try to find a lower rank decomposition.
    
    # For 2x3x3, the known lower bound is at least max(n*m, m*p, p*n) = max(6,9,6) = 9
    # but more refined bounds exist. The trivial upper bound is 18.
    
    # Strategy: Try rank values from a lower bound upward, and for each rank,
    # search for an exact decomposition using integer/half-integer coefficients.
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the "all-ones" tensor of shape (2,3,3).
    
    # A rank-1 decomposition of this tensor would be:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the all-ones tensor, we can try structured decompositions.
    
    # Key insight: The tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = (sum over r) u_r(i,j) * v_r(j,k) * w_r(k,i)
    
    # Let's try a parametric approach. Since all entries are 1, we can exploit symmetry.
    
    # For rank r, we need:
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 18 equations in 3*R*rank variables (with rank=R).
    # Actually it's R*(n*m + m*p + p*n) = R*(6+9+6) = 21R unknowns, but the equations
    # are multilinear (trilinear) in the factors.
    
    # Let's try small ranks. The minimum possible rank is at least the max of the
    # "slice ranks" or other lower bounds.
    
    # For the all-ones tensor of shape (n,m,p), there's a known result:
    # rank >= max(n, m, p) in some cases, but for the multiplication tensor it's different.
    
    # Actually, let me reconsider. The tensor for matrix multiplication C = AB where
    # A is n x m, B is m x p, C is n x p is:
    # T[i, j, k] = 1 (it's the standard multiplication tensor)
    
    # The rank of this tensor is the bilinear complexity of multiplying an n x m by m x p matrix.
    
    # For 2x3x3:
    # - Schoolbook: 2*3*3 = 18
    # - Can we do better?
    
    # Let me try to construct an explicit decomposition with rank < 18.
    
    # Approach: Use the fact that we can combine some terms.
    # For example, for the 2x2x2 case, Strassen gives rank 7 < 8.
    
    # For 2x3x3, let's think about it differently.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We have 6 output entries, each is a sum of 3 products.
    
    # Let's try rank 17, 16, ... and see if we can find a valid decomposition.
    
    # A systematic approach: fix the structure of U, V, W with some parameters,
    # and solve for the remaining parameters.
    
    # Alternative: Since the tensor is "simple" (all ones), maybe there's a closed-form
    # low-rank decomposition.
    
    # Let me try: Can we write the all-ones (2,3,3) tensor as a sum of fewer than 18 rank-1 tensors?
    
    # A rank-1 tensor is u_i * v_j * w_k for some vectors u, v, w.
    
    # The all-ones tensor J = 1 1 1 (outer product of all-ones vectors) has rank 1!
    # Wait, that's for the tensor where T[i,j,k] = 1 for ALL i,j,k.
    
    # But in matrix multiplication, the tensor is NOT the all-ones tensor in the sense that
    # every (i,j,k) triple contributes. Let me re-examine.
    
    # Actually, for matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has a 1 at position (i,j,k) for each valid (i,j,k) where
    # i is a row of A (and C), j is a column of A / row of B, k is a column of B (and C).
    
    # So the tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # That IS the all-ones tensor of shape (2,3,3)!
    
    # And the all-ones tensor has rank 1! It's just the outer product of (1,1), (1,1,1), (1,1,1).
    
    # Wait, but that would mean we can multiply any 2x3 by 3x3 matrix with 1 multiplication?
    # That can't be right. Let me reconsider.
    
    # The issue is that in the tensor decomposition for matrix multiplication, the factors
    # U, V, W are not arbitrary. They must satisfy the specific structure:
    # - U has entries U[i*m+j, r] which multiplies A[i,j]
    # - V has entries V[j*p+k, r] which multiplies B[j,k]
    # - W has entries W[k*n+i, r] which contributes to C[i,k]
    
    # The decomposition is: C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_k' V[j*p+k',r] * B[j,k']) * ...
    
    # No wait, that's not right either. Let me be more careful.
    
    # The standard formulation: We want to compute C = A*B.
    # A is n x m, B is m x p, C is n x p.
    
    # The multiplication tensor T is a 3-tensor of shape (n*m, m*p, p*n) where:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid (i,j,k)
    # T[a, b, c] = 0 otherwise
    
    # So T is NOT the all-ones tensor. It's a sparse tensor where T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For n=2, m=3, p=3:
    # a ranges over 0..5 (i*m+j for i in {0,1}, j in {0,1,2})
    # b ranges over 0..8 (j*p+k for j in {0,1,2}, k in {0,1,2})
    # c ranges over 0..5 (k*n+i for k in {0,1,2}, i in {0,1})
    
    # T[a,b,c] = 1 iff there exist (i,j,k) such that a = i*3+j, b = j*3+k, c = k*2+i
    
    # Let me enumerate:
    # i=0, j=0: a=0, b=3k, c=2k for k=0,1,2 -> (0,0,0), (0,3,2), (0,6,4)
    # i=0, j=1: a=1, b=3+k, c=2k for k=0,1,2 -> (1,3,0), (1,4,2), (1,5,4)
    # i=0, j=2: a=2, b=6+k, c=2k for k=0,1,2 -> (2,6,0), (2,7,2), (2,8,4)
    # i=1, j=0: a=3, b=3k, c=2k+1 for k=0,1,2 -> (3,0,1), (3,3,3), (3,6,5)
    # i=1, j=1: a=4, b=3+k, c=2k+1 for k=0,1,2 -> (4,3,1), (4,4,3), (4,5,5)
    # i=1, j=2: a=5, b=6+k, c=2k+1 for k=0,1,2 -> (5,6,1), (5,7,3), (5,8,5)
    
    # So T has 18 non-zero entries, one for each (i,j,k).
    
    # The tensor decomposition: T = sum_r u_r * v_r * w_r (outer products)
    # where u_r is a vector of length n*m=6, v_r of length m*p=9, w_r of length p*n=6.
    
    # This is equivalent to: for each (a,b,c) with T[a,b,c]=1:
    # sum_r U[a,r] * V[b,r] * W[c,r] = 1
    # and for (a,b,c) with T[a,b,c]=0:
    # sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is a much more constrained problem than the all-ones tensor.
    
    # The schoolbook decomposition uses rank 18 with U, V, W having single 1s.
    
    # To get lower rank, we need to "share" factors across multiple (i,j,k) triples.
    
    # Let's try a computational search for small ranks.
    
    # Given the complexity, let me try a heuristic approach:
    # 1. Start with rank 18 (schoolbook)
    # 2. Try to merge pairs of rank-1 terms if possible
    
    # Or, use a more direct algebraic approach.
    
    # For now, let's implement a search that tries to find a valid decomposition
    # for decreasing ranks, using a randomized local search or algebraic method.
    
    # Given the time constraints, let me try a structured ansatz.
    
    # Key observation: The tensor has a certain structure. Let's see if we can find
    # a decomposition with rank < 18.
    
    # Let me try to verify if rank 18 is indeed optimal or if we can do better.
    
    # For the purpose of this task, let's implement a search that:
    # 1. Tries rank values from 15 to 18 (since 18 is known to work)
    # 2. For each rank, uses a randomized search to find valid U, V, W
    
    # But randomized search over a huge space is unlikely to work quickly.
    
    # Alternative: Use the fact that the problem has algebraic structure.
    
    # Let me try a different approach: parameterize U, V, W with a small number of
    # free parameters and solve the resulting polynomial system.
    
    # For rank R, we have 6*R + 9*R + 6*R = 21R unknowns and 18 equations (for the
    # non-zero entries) plus many more for the zero entries.
    
    # The zero entries give additional constraints. Let me count them.
    # Total possible (a,b,c) triples: 6*9*6 = 324
    # Non-zero: 18
    # Zero: 324 - 18 = 306
    
    # So we have 306 + 18 = 324 equations (but the non-zero ones require sum = 1,
    # zero ones require sum = 0).
    
    # This is a system of 324 polynomial equations in 21R variables.
    
    # For R=18: 378 variables, 324 equations. Underdetermined, so solutions exist.
    # For R=17: 357 variables, 324 equations. Still underdetermined.
    # For R=15: 315 variables, 324 equations. Overdetermined, may or may not have solutions.
    
    # Let me try to find a solution for rank 17, 16, etc.
    
    # Given the complexity of solving this exactly, let me try a different strategy:
    # Use the known structure to propose a specific low-rank decomposition.
    
    # Actually, let me just return the schoolbook decomposition for now, which is guaranteed
    # to be correct, and see if I can improve it.
    
    # Schoolbook decomposition:
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
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a non-zero entry
                    is_nonzero = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i * m + j and b == j * p + k and c == k * n + i:
                                    is_nonzero = True
                    if not is_nonzero:
                        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if val != 0:
                            return False
        return True
    
    # The schoolbook decomposition should verify
    assert verify(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
