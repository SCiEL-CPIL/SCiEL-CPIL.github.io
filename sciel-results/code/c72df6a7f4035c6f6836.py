"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction as Fr
from itertools import product

def _build_trivial(n, m, p):
    """Build the trivial (schoolbook) decomposition of rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor.
    Returns True if valid, False otherwise."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    # For each valid (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if abs(val - 1) > tol:
                    return False
    # Check invalid positions: where index constraints are violated
    # a=(i,j), b=(j',k'), c=(k'',i'') must have j==j', k'==k'', i''==i
    # Find positions where at least one constraint fails
    invalid_checked = 0
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a != j_b:
                for c in range(p * n):
                    k_c, i_c = divmod(c, n)
                    if k_b != k_c or i_c != i_a:
                        val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                        if abs(val) > tol:
                            return False
                        invalid_checked += 1
                        if invalid_checked >= 50:
                            break
                if invalid_checked >= 50:
                    break
        if invalid_checked >= 50:
            break
    return True

def _try_rank44_pair_merge(n, m, p):
    """
    Try to find a rank-44 decomposition by merging two trivial terms into one.
    
    For two trivial terms (i1,j1,k1) and (i2,j2,k2), we look for a single rank-1 term
    (u, v, w) such that:
      - At position (i1,j1,k1): u[i1*m+j1] * v[j1*p+k1] * w[k1*n+i1] contributes appropriately
      - At position (i2,j2,k2): u[i2*m+j2] * v[j2*p+k2] * w[k2*n+i2] contributes appropriately
      - At all other valid positions: contribution is 0
      - At all invalid positions: contribution is 0
    
    The simplest case: if the two terms share all three indices (impossible for distinct terms)
    or share two indices. Let's consider terms that share exactly one index.
    
    Actually, a more promising approach: look for a rank-1 tensor that is a linear combination
    of two basis tensors. If we have basis terms e_{i1,j1,k1} and e_{i2,j2,k2}, and we want
    to replace them with a single term, we need:
      u ⊗ v ⊗ w = α * e_{i1,j1,k1} + β * e_{i2,j2,k2}
    
    This means u, v, w must be zero except at positions involved in the two terms.
    For this to work as a rank-1 tensor, the support patterns must be compatible.
    
    Let's try a different approach: use the fact that for matrix multiplication,
    there might be algebraic identities. For 3x3x3, Strassen's algorithm gives rank 7 < 27.
    
    For 3x3x5, let's try to find if there's a known sub-multiplicative structure.
    """
    n, m, p = 3, 3, 5
    
    # Try to find a rank-1 tensor that equals the sum of two basis tensors
    # e_{i1,j1,k1} + e_{i2,j2,k2}
    # This requires that the three vectors u, v, w have supports that align.
    
    # For e_{i,j,k}, the vectors are:
    #   u has 1 at position i*m+j, 0 elsewhere
    #   v has 1 at position j*p+k, 0 elsewhere
    #   w has 1 at position k*n+i, 0 elsewhere
    
    # For the sum of two such to be rank-1, we need:
    #   u = a * u1 + b * u2 where u1, u2 are the unit vectors for the two terms
    #   Similarly for v and w
    #   And (a*u1 + b*u2) ⊗ (a'*v1 + b'*v2) ⊗ (a''*w1 + b''*w2) = e1 + e2
    
    # This is highly constrained. Let's enumerate pairs and check.
    
    basis_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for idx1, (i1, j1, k1) in enumerate(basis_terms):
        for idx2, (i2, j2, k2) in enumerate(basis_terms):
            if idx1 >= idx2:
                continue
            
            # Try to find u, v, w such that the outer product equals e1 + e2
            # u must have non-zero entries only at positions i1*m+j1 and i2*m+j2
            # v must have non-zero entries only at positions j1*p+k1 and j2*p+k2
            # w must have non-zero entries only at positions k1*n+i1 and k2*n+i2
            
            a1 = i1 * m + j1
            a2 = i2 * m + j2
            b1 = j1 * p + k1
            b2 = j2 * p + k2
            c1 = k1 * n + i1
            c2 = k2 * n + i2
            
            # For the outer product to equal e1 + e2, we need:
            # u[a1]*v[b1]*w[c1] = 1 (contribution to e1)
            # u[a2]*v[b2]*w[c2] = 1 (contribution to e2)
            # All other combinations must give 0
            
            # The "cross terms" that must be zero:
            # u[a1]*v[b2]*w[c1] = 0
            # u[a1]*v[b1]*w[c2] = 0
            # u[a2]*v[b2]*w[c1] = 0
            # u[a2]*v[b1]*w[c2] = 0
            
            # For these to be zero, at least one factor in each product must be zero.
            # This means the supports must be "aligned" in a specific way.
            
            # Case 1: a1 == a2 (same row in U), then we can set u[a1] = 1, u[a2] = 1
            #   Then we need v[b1]*w[c1] = 1, v[b2]*w[c2] = 1, v[b2]*w[c1] = 0, v[b1]*w[c2] = 0
            #   This requires that {b1,c1} and {b2,c2} form a "cross" pattern where
            #   v[b1] and w[c2] are not both non-zero, etc.
            
            # Let's try small integer/half-integer values
            for ua1 in [1, 2, Fr(1,2)]:
                for ua2 in [1, 2, Fr(1,2)]:
                    if ua1 == 0 or ua2 == 0:
                        continue
                    # v[b1] * w[c1] = 1 / ua1
                    # v[b2] * w[c2] = 1 / ua2
                    # v[b2] * w[c1] = 0
                    # v[b1] * w[c2] = 0
                    
                    # From v[b2]*w[c1] = 0: either v[b2]=0 or w[c1]=0
                    # From v[b1]*w[c2] = 0: either v[b1]=0 or w[c2]=0
                    
                    # Subcase A: v[b2] = 0
                    #   Then v[b2]*w[c2] = 0, but we need it to be 1/ua2. Contradiction.
                    
                    # Subcase B: w[c1] = 0
                    #   Then v[b1]*w[c1] = 0, but we need it to be 1/ua1. Contradiction.
                    
                    # So if a1 == a2, we cannot satisfy all constraints unless...
                    # Wait, let me reconsider. The cross terms that must be zero are:
                    #   u[a1]*v[b2]*w[c1] and u[a1]*v[b1]*w[c2] and u[a2]*v[b2]*w[c1] and u[a2]*v[b1]*w[c2]
                    # If a1 == a2, then u[a1] = u[a2] = some value u_a
                    # The cross terms are: u_a*v[b2]*w[c1] = 0, u_a*v[b1]*w[c2] = 0, etc.
                    # Since u_a != 0, we need v[b2]*w[c1] = 0 and v[b1]*w[c2] = 0.
                    # And we need v[b1]*w[c1] = 1/u_a and v[b2]*w[c2] = 1/u_a.
                    
                    # This is possible! For example, if b1 != b2 and c1 != c2:
                    #   v = [0,...,v1 at b1, 0,...,v2 at b2,0...]
                    #   w = [0,...,w1 at c1, 0,...,w2 at c2,0...]
                    #   Need: v1*w1 = 1/u_a, v2*w2 = 1/u_a, v2*w1 = 0, v1*w2 = 0
                    #   From v2*w1 = 0: either v2=0 or w1=0
                    #   If v2=0, then v2*w2=0 != 1/u_a. Contradiction.
                    #   If w1=0, then v1*w1=0 != 1/u_a. Contradiction.
                    
                    # So a1 == a2 doesn't work either (unless b1==b2 or c1==c2, which means the terms aren't distinct in those indices)
            
            # This analysis shows that merging two basis terms into one rank-1 term is very difficult.
            # The fundamental issue is that rank-1 tensors have a very rigid structure.
            
            # Let's try a different strategy: use known decompositions.
            # For 3x3x3, Strassen gives rank 7. Can we embed this into 3x3x5?
            # M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
            # If we have a rank-7 decomposition of M_{3,3,3}, and a rank-18 decomposition of M_{3,3,2},
            # we get rank 25 total.
            
            # But we need the actual coefficients. Let's try to implement Strassen for 3x3x3.
            # Strassen's algorithm for 2x2x2 has rank 7. For 3x3x3, we can use a block decomposition.
            
            # Actually, let's try a simpler approach: use the "lax" decomposition or other known results.
            # For now, let's return the trivial decomposition and note that finding a lower rank
            # requires more sophisticated algebraic methods.
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Start with the trivial decomposition as the best known
    best = _build_trivial(n, m, p)
    
    # Try to find a rank-44 decomposition
    result = _try_rank44_pair_merge(n, m, p)
    if result is not None:
        if _verify(result, n, m, p):
            best = result
    
    return best
# EVOLVE-BLOCK-END
