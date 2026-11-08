import numpy as np
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    # Multiplication tensor: for each (i, j, k) with i<n, j<m, k<p:
    #   A[i,j] * B[j,k] contributes to C[i,k]
    # Index conventions:
    #   U has n*m rows (index i*m + j)
    #   V has m*p rows (index j*p + k)
    #   W has p*n rows (index k*n + i)
    #   T[(i*m+j), (j*p+k)] = 1 iff (i,j,k) is a valid term, and contributes to W at (k*n+i)
    #
    # The tensor is a 3-way tensor T with entries T[u,v] where u = i*m+j, v = j*p+k
    # and the output is W[k*n + i].
    #
    # We want to find rank-1 decompositions: sum_r U[:,r] * V[:,r] * W[:,r]
    # such that for all (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for invalid combinations the product is 0.
    #
    # This is a tensor rank problem. We'll use a greedy sparse approach:
    # Build the multiplication tensor explicitly, then try to cover it with
    # rank-1 terms found via pattern matching.

    # The multiplication tensor has the structure:
    # For each j in range(m):
    #   For each i in range(n), k in range(p):
    #     U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] * 1 (since T=1)
    #
    # This is essentially: for each j, we're multiplying an (n x 1) vector by (1 x p) vector,
    # giving an (n x p) matrix. The j-th block is outer product of column i from A-row and B-row.
    #
    # Actually, the standard matrix multiplication tensor for (n,m,p) has:
    # T[i, j, k] = 1 for all valid (i,j,k), and the contraction gives C[i,k] = sum_j A[i,j] B[j,k]
    #
    # In the U,V,W factorization: C[i,k] = sum_r U[i*m+? ...
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    #
    # So we need: for all valid (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all invalid (i,j,k) (i.e., where the product A[i,j]*B[j,k] doesn't contribute),
    # the sum should be 0. But actually in matrix multiplication, ALL (i,j,k) with i<n, j<m, k<p
    # are valid, and each contributes exactly once to C[i,k].
    #
    # So the constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a bilinear/trilinear system. Let's think of it differently.
    #
    # For fixed j, define A_j as the n-vector (U[i*m+j, r] for i in range(n)) and
    # B_j as the p-vector (V[j*p+k, r] for k in range(p)). Then for each r,
    # A_j[r] * B_j[r] is an n x p matrix (outer product), and we need:
    #   sum_r (A_j[r] * B_j[r])_{i,k} * W[k*n+i, r] = 1 for all i,k
    #
    # This is getting complex. Let me try a different approach: use the known
    # Strassen-like constructions or try to find a low-rank decomposition via
    # integer programming / greedy methods.
    #
    # Given the 120s time limit and the complexity, let's try a structured approach:
    # For m=3, we can think of this as 3 matrix multiplications (for j=0,1,2)
    # each of size (2 x 7), combined.
    #
    # Actually, let me just implement a proper search using the tensor structure.
    # I'll use the fact that the multiplication tensor is bilinear and try to
    # find rank-1 components that cover the tensor.

    # Let's use a direct approach: enumerate possible rank-1 terms and use
    # a greedy set cover / matching approach.
    #
    # A rank-1 term is defined by:
    #   u_vector: n*m entries (nonzero at some positions)
    #   v_vector: m*p entries
    #   w_vector: p*n entries
    # such that for all (i,j,k): u[i*m+j] * v[j*p+k] * w[k*n+i] = T[i,j,k]
    # where T[i,j,k] = 1 if (i,j,k) is a valid multiplication term.
    #
    # For the term to be valid, we need:
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all valid (i,j,k) that the term covers
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] = 0 for all invalid (i,j,k)
    #
    # Since all entries are integers/half-integers, and the product must be 1,
    # the nonzero entries must multiply to 1, meaning each nonzero entry in
    # u, v, w for a given term must be ±1 (or combinations that multiply to 1).
    #
    # Actually, for a rank-1 term to contribute exactly 1 to T[i,j,k], we need
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = 1. If u, v, w are integers, this means
    # each must be ±1 and the product must be +1.
    #
    # Let me try a completely different approach: use the known optimal or
    # near-optimal constructions from the literature for small matrix multiplications.
    #
    # For 2x3 by 3x7, the trivial rank is n*m*p = 2*3*7 = 42.
    # The current best is 39. Let's see if we can do better.
    #
    # I'll implement a greedy algorithm that builds up the decomposition term by term,
    # choosing terms that cover the most remaining tensor entries.

    # Build the tensor: T[(i,j,k)] = 1 for all i<n, j<m, k<p
    # Total entries: n*m*p = 42

    # For a rank-1 term to be valid, it must be of the form:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for some subset of (i,j,k)
    # and 0 for all others.
    #
    # Since u[i*m+j] * v[j*p+k] * w[k*n+i] must be 1 or 0, and entries are integers,
    # the nonzero pattern of u, v, w must be consistent.
    #
    # Key insight: for a given term, if u[i*m+j] != 0 and v[j*p+k] != 0, then
    # w[k*n+i] must also be nonzero, and the product must be 1.
    #
    # This means the support of the term forms a "box" in the (i,j,k) space where
    # for each j, the set of (i,k) pairs with u[i*m+j]!=0 and v[j*p+k]!=0 must have
    # w[k*n+i]!=0, and the product condition must hold.
    #
    # Given the complexity of enumerating all valid rank-1 terms, let me try a
    # structured approach based on partitioning the j dimension.

    # For each j, we have an n x p multiplication (2 x 7).
    # The rank of a 2 x 7 matrix multiplication is at most 2*7 = 14 (trivial).
    # But we can do better with Strassen-like methods.
    #
    # Actually, for a single j, the "multiplication" is just the outer product
    # of a 2-vector and a 7-vector, giving a 2x7 matrix. The rank-1 decomposition
    # of the outer product tensor A_i * B_k = C_{i,k} for i<2, k<7 requires
    # at least max(2,7) = 7 rank-1 terms if we're decomposing the bilinear map
    # from R^2 x R^7 to R^{14}.
    #
    # Wait, I'm confusing myself. Let me reconsider.
    #
    # The full problem is: given A (2x3) and B (3x7), compute C = A*B (2x7).
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # The tensor T has entries indexed by (i, j, k) -> (i, k), with T[i,j,k] = 1.
    # We want to decompose this as sum_r u_r * v_r * w_r where:
    # u_r is a vector in R^{n*m} = R^6
    # v_r is a vector in R^{m*p} = R^{21}
    # w_r is a vector in R^{p*n} = R^{14}
    # such that for all (i,j,k): u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # Summing over r: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # This is the standard tensor rank problem for matrix multiplication.
    #
    # Let me try to find a low-rank decomposition using a simple heuristic:
    # start with the trivial 42-rank decomposition and try to merge terms.

    # Trivial decomposition: one term for each (i,j,k)
    # u_r[i*m+j] = 1, v_r[j*p+k] = 1, w_r[k*n+i] = 1
    # But this gives rank 42.
    #
    # To reduce rank, we need to find terms where a single rank-1 component
    # covers multiple (i,j,k) entries.
    #
    # A rank-1 term can cover multiple (i,j,k) if there exist sets I, J, K such that
    # for all i in I, j in J, k in K: u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # and for all other (i,j,k): the product is 0.
    #
    # This requires that the support of u is contained in {(i,j): i in I, j in J},
    # the support of v is contained in {(j,k): j in J, k in K},
    # the support of w is contained in {(k,i): k in K, i in I},
    # and for all (i,j,k) in I x J x K: u[i*m+j] * v[j*p+k] * w[k*n+i] = 1.
    #
    # For the product to be 1 for all (i,j,k) in I x J x K, we need:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # This means that for fixed j, the vectors u[:,j] (over i) and v[j,:] (over k)
    # must be such that u[i*m+j] * v[j*p+k] is constant (independent of i,k)
    # for i in I, k in K, and that constant times w[k*n+i] = 1.
    #
    # The simplest case: |I|=1, |J|=1, |K|=1 -> single term, rank contribution 1.
    # Next: |I|=1, |J|=1, |K|=2 -> u has one nonzero, v has one nonzero, w has two nonzeros.
    #   u[i0*m+j0] = 1, v[j0*p+k0] = 1, v[j0*p+k1] = 1, w[k0*n+i0] = 1, w[k1*n+i0] = 1
    #   This covers (i0,j0,k0) and (i0,j0,k1).
    #
    # Let me try to build a decomposition by grouping terms.

    # Given the time constraint, let me try a specific construction.
    # For n=2, m=3, p=7:
    #
    # Group by j:
    # j=0: terms (i,0,k) for i in {0,1}, k in {0,...,6} -> 14 terms
    # j=1: terms (i,1,k) for i in {0,1}, k in {0,...,6} -> 14 terms
    # j=2: terms (i,2,k) for i in {0,1}, k in {0,...,6} -> 14 terms
    # Total: 42 terms
    #
    # Can we merge terms across j values?
    # If we have a term that covers (i,j1,k) and (i,j2,k) for j1 != j2, then
    # u[i*m+j1] * v[j1*p+k] * w[k*n+i] = 1 and u[i*m+j2] * v[j2*p+k] * w[k*n+i] = 1
    # This requires u[i*m+j1] * v[j1*p+k] = u[i*m+j2] * v[j2*p+k] (since w is the same).
    #
    # Let's try to find a construction with rank < 42.

    # I'll implement a greedy approach:
    # 1. Start with all 42 tensor entries uncovered.
    # 2. Repeatedly find the best rank-1 term that covers the most uncovered entries.
    # 3. Add it to the decomposition and mark those entries as covered.
    # 4. Repeat until all entries are covered.

    # To find the best rank-1 term, we need to enumerate possible supports.
    # This is combinatorially expensive, but for small n, m, p it might be feasible.

    # Let's define a function to check if a given (u_support, v_support, w_support)
    # forms a valid rank-1 term, and what entries it covers.

    def get_entries():
        """Return all (i,j,k) entries of the multiplication tensor."""
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entries.append((i, j, k))
        return entries

    def term_covers(i_set, j_set, k_set):
        """
        For a rank-1 term with u nonzero at {(i,j) for i in i_set, j in j_set},
        v nonzero at {(j,k) for j in j_set, k in k_set},
        w nonzero at {(k,i) for k in k_set, i in i_set},
        the term covers entries (i,j,k) where i in i_set, j in j_set, k in k_set.
        But we need to verify that the product condition can be satisfied.
        """
        # For the product u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all (i,j,k) in i_set x j_set x k_set,
        # we need: for each j in j_set, the value u[i*m+j] * v[j*p+k] must be
        # independent of i (for i in i_set) and k (for k in k_set), and equal to
        # 1/w[k*n+i] which must be independent of k (for fixed i) and consistent.
        #
        # The simplest valid term: |i_set|=1, |j_set|=1, |k_set| arbitrary.
        # Then u has one nonzero, v has |k_set| nonzeros, w has |k_set| nonzeros.
        # u[i0*m+j0] = 1, v[j0*p+k] = 1 for k in k_set, w[k*n+i0] = 1 for k in k_set.
        # This works.
        #
        # Another: |i_set|=1, |j_set| arbitrary, |k_set|=1.
        # u[i0*m+j] = 1 for j in j_set, v[j*p+k0] = 1 for j in j_set, w[k0*n+i0] = 1.
        # This works.
        #
        # |i_set| arbitrary, |j_set|=1, |k_set|=1 also works.
        #
        # For larger sets, it's more complex. Let's start with the simple cases.
        pass

    # Given the complexity, let me just try to find a good decomposition by
    # using known techniques. I'll use the following strategy:
    #
    # For each j, the j-th "slice" is a 2x7 multiplication. We can decompose
    # each slice independently and then combine.
    #
    # For a 2x7 multiplication (A is 2x1, B is 1x7, C is 2x7):
    # C[i,k] = A[i] * B[k]
    # This is a rank-1 bilinear map, which can be decomposed into 7 rank-1 terms
    # (one for each k), or potentially fewer if we use Strassen-like methods.
    #
    # Actually, the tensor for C[i,k] = A[i] * B[k] has entries T[i,k] = 1 for
    # all i,k. The rank of this tensor is min(2, 7) = 2? No, the tensor rank
    # of the outer product is 1 if we're just doing A*B as a scalar, but here
    # we're producing a 2x7 matrix.
    #
    # I think I'm overcomplicating this. Let me just implement the known 39-rank
    # construction and try to improve it slightly.

    # Actually, let me try a completely different approach: use numpy to solve
    # for a low-rank decomposition directly.
    #
    # The constraint is: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of 42 equations in rank * 41 unknowns (6 + 21 + 14 = 41 per rank).
    # If we fix rank = R, we have 41*R unknowns and 42 equations.
    #
    # We can try to find a solution by iteratively setting R = 1, 2, ... and
    # checking if a solution exists. This is a nonlinear problem (trilinear),
    # so it's hard to solve directly.
    #
    # Given the time constraint, let me just return the known 39-rank construction.

    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(k_start):
        nonlocal col
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        U[0][col] += 1; U[4][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        U[3][col] += 1; U[4][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += -1
        col += 1

        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] += -1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        U[4][col] += 1
        V[b10][col] += 1; V[b00][col] += -1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1

        U[3][col] += 1; U[0][col] += -1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        U[1][col] += 1; U[4][col] += -1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)

    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1

    U[0][col] += 1; V[b06][col] += 1; W[c06][col] += 1; col += 1
    U[1][col] += 1; V[b16][col] += 1; W[c06][col] += 1; col += 1
    U[3][col] += 1; V[b06][col] += 1; W[c16][col] += 1; col += 1
    U[4][col] += 1; V[b16][col] += 1; W[c16][col] += 1; col += 1

    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1

        U[2][col] += 1; V[b2k][col] += 1; W[c0k][col] += 1; col += 1
        U[5][col] += 1; V[b2k][col] += 1; W[c1k][col] += 1; col += 1

    return {"U": U, "V": V, "W": W}
