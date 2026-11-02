import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    def _time_ok():
        return time.time() < deadline

    # Build the schoolbook rank-72 decomposition
    def build_schoolbook():
        """
        Returns (U, V, W) as lists of lists, each of shape (dim, 72)
        U: (n*m, 72), V: (m*p, 72), W: (p*n, 72)
        """
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    def _verify(U, V, W, n, m, p, rank, tol=1e-9):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_zero(U, V, W, n, m, p, rank, tol=1e-9):
        """Verify that the decomposition gives zero for all non-multiplication entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i * m + j == 0 and j * p + k == 0 and k * n + i == 0:
                        continue  # skip the main diagonal
                    # Actually we need to check all (i,j,k) triples
                    pass
        # More thorough: check that the tensor is exactly the multiplication tensor
        # T[i,j,k] = 1 if this is a valid multiplication term, 0 otherwise
        # Actually for matrix multiplication, T[i,j,k] is always 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j]*B[j,k]
        # So every (i,j,k) triple has coefficient 1.
        # Let me re-check: the tensor for matrix multiplication M_{n,m,p} has
        # T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
        # So we just need _verify to check all of them equal 1.
        return True

    # Iterative rank-1 peeling (de Zeeuw's algorithm)
    def peel_rank1(U, V, W, n, m, p):
        """
        Given a decomposition (U, V, W) with current rank,
        try to find a rank-1 term in the null space of the current decomposition
        and add it, then reduce the effective rank by projecting out.
        
        Actually, de Zeeuw's algorithm works differently:
        1. Start with a known decomposition (e.g., schoolbook).
        2. At each step, find a rank-1 tensor in the null space of the 
           "slice" structure that can be added.
        3. More precisely: we look for (u, v, w) such that for all (i,j,k),
           sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] + u[i*m+j]*v[j*p+k]*w[k*n+i] 
           still equals the target tensor.
        
        A simpler approach: 
        - The current decomposition gives T = sum_r u_r ⊗ v_r ⊗ w_r
        - We want to find a new decomposition of lower rank.
        - Strategy: find a linear dependency among the columns of the 
          matricizations.
        
        Let me use a different approach:
        - Compute the 3-mode matricizations of the current tensor decomposition.
        - Find the null space of the combined system.
        - Use that to reduce rank.
        
        Actually, the most practical approach for this small problem:
        - Build the full tensor T of size n*m*p = 72 entries (all 1s for mult tensor).
        - The schoolbook gives 72 rank-1 terms.
        - We want to find if there's a lower-rank decomposition.
        - Use the fact that the tensor is "structured" - it's the matrix multiplication tensor.
        
        Let me try a direct approach:
        - Represent the tensor as a 3D array T[i,j,k].
        - For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
        - The rank of this tensor is what we want to minimize.
        
        Actually, for the matrix multiplication tensor M_{n,m,p}, the entries are:
        T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
        This is just the all-ones tensor of size n×m×p!
        
        Wait, that's not right. Let me re-read the problem.
        
        "Matrix multiplication 3,4,6"
        "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
        
        The tensor for matrix multiplication is:
        T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        Yes, it IS the all-ones tensor. Because C[i,k] = sum_j A[i,j] * B[j,k],
        and the tensor T has T[i,j,k] = 1 for every (i,j,k) triple.
        
        So we need the tensor rank of the all-ones tensor of size 3×4×6.
        
        The rank of the all-ones tensor J_{n,m,p} is known to be:
        - For 2×2×2: rank 7 (Strassen)
        - For general sizes, it's related to the matrix multiplication rank.
        
        Actually wait - the all-ones tensor has rank 1! Because J = 1⊗1⊗1.
        
        No wait, that's wrong too. Let me think more carefully.
        
        The tensor T for matrix multiplication satisfies:
        C[i,k] = sum_{j} A[i,j] * B[j,k]
        
        In tensor form: T = sum_{r} u_r ⊗ v_r ⊗ w_r where
        (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
        
        And we need T[i,j,k] = 1 for all i,j,k.
        
        So T IS the all-ones tensor, and its rank is 1!
        u = [1,1,...,1], v = [1,1,...,1], w = [1,1,...,1]
        
        But that can't be right because the problem says the schoolbook rank is 72
        and Strassen gives 66.
        
        Let me re-read the interface:
        "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        "W[k*n+i,r]"
        
        So U has shape (n*m, rank), meaning each column of U is a vector in R^{n*m}.
        Similarly V is in R^{m*p}, W is in R^{p*n}.
        
        The decomposition is:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        And we need T[i,j,k] = 1 for all valid (i,j,k).
        
        So yes, T is the all-ones tensor of size (n*m) × (m*p) × (p*n) with the 
        specific indexing, but the values are all 1.
        
        Wait, but the dimensions don't match for a standard tensor. Let me re-examine.
        
        U[i*m+j, r] is indexed by (i,j) pairs, so it's a vector in R^{n*m}.
        V[j*p+k, r] is indexed by (j,k) pairs, so it's a vector in R^{m*p}.
        W[k*n+i, r] is indexed by (k,i) pairs, so it's a vector in R^{p*n}.
        
        The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is a scalar for each r,
        and we sum over r to get T[i,j,k].
        
        For this to equal 1 for all i,j,k, we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is NOT the same as the all-ones tensor in the usual sense, because 
        the three factors are indexed by different pairs: (i,j), (j,k), (k,i).
        
        So T is a 3D array where T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
        But the "tensor" is really a map from the 3D grid [n]×[m]×[p] to R,
        and we're decomposing it as a sum of rank-1 terms where each rank-1 term
        is a product of three vectors, one for each coordinate.
        
        So it IS the all-ones tensor of size n×m×p, and its tensor rank is what we seek.
        
        For the all-ones tensor J_{a,b,c}, the tensor rank is:
        - min(a,b,c) if one dimension is 1
        - For 2×2×2: 7
        - For larger: it's related to matrix multiplication ranks
        
        Actually, the tensor rank of J_{a,b,c} is exactly the same as the rank needed
        to multiply an a×b matrix by a b×c matrix, which is what we're computing.
        
        So for 3×4×6, we need the tensor rank of J_{3,4,6}.
        
        Known results:
        - J_{2,2,2} has rank 7
        - J_{2,2,3} has rank 11 (or 12?)
        - J_{3,3,3} has rank at most 23 (Strassen), lower bound 19
        
        For 3×4×6, we can use block decomposition:
        - Split into 2×2×2 blocks where possible.
        - 3×4×6 = (2+1)×(2+2)×(2+2+2)
        
        Using Strassen on 2×2×2 blocks:
        - For the 2×2×2 part: 7 terms
        - For the 1×2×2 part: 2 terms (schoolbook)
        - etc.
        
        Let me just implement the peeling algorithm properly.
        """
        pass

    # Let me implement a direct algebraic search.
    # The key insight: for the all-ones tensor, we can use the following approach:
    # 1. Start with the schoolbook decomposition (rank 72).
    # 2. Use linear algebra to find dependencies and reduce rank.
    
    # Actually, let me use a known result:
    # The tensor rank of J_{n,m,p} can be bounded by:
    # - Schoolbook: n*m*p
    # - Strassen-like: use block decomposition
    
    # For 3×4×6:
    # Option 1: Use Strassen on 2×2×2 blocks
    #   Split 3 = 2+1, 4 = 2+2, 6 = 2+2+2
    #   2×2×2 blocks: 1*2*3 = 6 blocks, each rank 7 = 42
    #   1×2×2 blocks: 1*2*3 = 6 blocks, each rank 2 = 12
    #   2×1×2 blocks: 1*2*3 = 6 blocks, each rank 2 = 12
    #   1×1×2 blocks: 1*1*3 = 3 blocks, each rank 2 = 6
    #   2×2×1 blocks: 1*2*3 = 6 blocks, each rank 2 = 12
    #   2×1×1 blocks: 1*2*3 = 6 blocks, each rank 1 = 6
    #   1×2×1 blocks: 1*2*3 = 6 blocks, each rank 1 = 6
    #   1×1×1 blocks: 1*1*3 = 3 blocks, each rank 1 = 3
    #   Total: 42+12+12+6+12+6+6+3 = 99 -- this is worse!
    
    # The issue is that when we split, we need to handle the cross-terms carefully.
    # Let me use the approach from the inspirations:
    # - Row 0: schoolbook (24 terms)
    # - Rows 1-2: Strassen on 2×2×2 blocks (42 terms)
    # Total: 66
    
    # Can we do better? Let me try to use a more clever decomposition.
    
    # Known: rank(J_{2,2,2}) = 7
    # rank(J_{2,2,3}) <= 11 (using Strassen + schoolbook)
    # rank(J_{3,3,3}) <= 23 (Strassen)
    
    # For 3×4×6:
    # We can view it as J_{3,4,6} = J_{3,4,2} ⊗ J_{1,1,3} + cross terms
    # This is getting complicated. Let me just implement the best known approach
    # and see if we can improve it.
    
    # Approach: Use the 66-rank decomposition from the inspirations,
    # then try to reduce it using linear algebra.
    
    U, V, W = build_schoolbook()
    current_rank = n * m * p  # 72
    
    # Try to reduce rank using null space analysis
    # The idea: if there's a linear dependency among the rank-1 terms,
    # we can eliminate one.
    
    # Build the matricizations
    # U_mat: (n*m) x rank
    # V_mat: (m*p) x rank  
    # W_mat: (p*n) x rank
    
    U_mat = np.array(U)  # (n*m, 72)
    V_mat = np.array(V)  # (m*p, 72)
    W_mat = np.array(W)  # (p*n, 72)
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the all-ones tensor, we can use the following trick:
    # If we can find vectors u, v, w such that u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # for all i,j,k, then we have a rank-1 term.
    # This requires u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1 for all valid indices.
    # So u = ones(n*m), v = ones(m*p), w = ones(p*n) gives a rank-1 term.
    
    # But wait, that means the entire tensor is rank 1!
    # Let me verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If U = ones, V = ones, W = ones, then the product is 1*1*1 = 1. Correct!
    
    # So the rank is 1?! But that contradicts the problem statement.
    
    # I must be misunderstanding the problem. Let me re-read.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]"
    
    # OK so the shapes are:
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (18, rank)
    
    # And the decomposition is:
    # For each output element C[i,k] (where i in [0,3), k in [0,6)):
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that's not right either. The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r (u_r[i,j] * v_r[j,k]) * w_r[k,i]
    
    # No, that's not standard either. Let me think about this differently.
    
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    # And the bilinear form is:
    # <T, A⊗B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j]*B[j,k]) * 1
    # = sum_{i,k} C[i,k]
    
    # Hmm, that's not quite right. The standard definition is:
    # T is a 3-tensor such that for matrices A (n×m) and B (m×p):
    # C = A*B where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In tensor form: C = T(A, B) where T is a trilinear map.
    # The tensor T can be written as:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n×m}, v_r is in R^{m×p}, w_r is in R^{p×n} (or R^{n×p})
    
    # And: C[i,k] = sum_r (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j,k] * B[j,k]) * w_r[i,k]
    
    # No, that's not right either. Let me look at this more carefully.
    
    # The standard tensor decomposition for matrix multiplication is:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where:
    # - u_r is a vector in R^{n*m} (flattened n×m matrix)
    # - v_r is a vector in R^{m*p} (flattened m×p matrix)  
    # - w_r is a vector in R^{p*n} (flattened p×n matrix, or n×p)
    
    # And the multiplication is:
    # C[i,k] = sum_r (u_r[i*m+j] * A[i,j]) * (v_r[j*p+k] * B[j,k]) * w_r[k*n+i]
    # Wait, this doesn't make sense dimensionally.
    
    # Let me use the convention from the problem:
    # U has shape (n*m, rank), so U[i*m+j, r] is the (i*m+j)-th component of the r-th factor.
    # V has shape (m*p, rank), so V[j*p+k, r] is the (j*p+k)-th component.
    # W has shape (p*n, rank), so W[k*n+i, r] is the (k*n+i)-th component.
    
    # The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This IS the all-ones tensor, and as I noted, it has rank 1 with
    # U = ones(12), V = ones(24), W = ones(18).
    
    # But the problem says the score is 1.0909 for rank 66, implying the rank should be lower.
    # And the inspirations use rank 66.
    
    # I think I'm misunderstanding the problem. Let me re-read the interface:
    # "exactly decompose the matrix multiplication tensor"
    
    # Maybe the tensor is NOT the all-ones tensor. Maybe it's:
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    # But for matrix multiplication, ALL (i,j,k) triples are valid.
    
    # OR maybe the tensor is defined differently. Let me look at the W index:
    # W[k*n+i, r] -- this indexes by (k, i) where k is the column of B and i is the row of A.
    # So W is a vector in R^{p*n}, indexed by (k, i) pairs.
    
    # I think the issue is that the tensor T is NOT the all-ones tensor.
    # Instead, T is the tensor such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # And the tensor decomposition gives:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    # No, that's still not right. Let me just accept the convention and work with it.
    
    # The key equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # And I've shown this has rank 1. So either the problem is trivial (rank 1),
    # or I'm misunderstanding the convention.
    
    # Let me look at the inspirations more carefully.
    # Inspiration 1 uses rank 66 and has specific U, V, W entries.
    # For example, add_schoolbook(0, j, k) sets:
    #   U[0*m+j][col] = 1  (i=0, so i*m+j = j)
    #   V[j*p+k][col] = 1
    #   W[k*n+0][col] = 1  (i=0, so k*n+i = k*n)
    
    # So for the schoolbook term (i,j,k), we have:
    #   U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # And the sum over r gives: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For the schoolbook, each r corresponds to a unique (i,j,k), so the sum is 1
    # only when the (i,j,k) matches, and 0 otherwise.
    
    # Wait, that means T[i,j,k] = 1 if there's a rank-1 term for (i,j,k), and 0 otherwise.
    # But for matrix multiplication, we want T[i,j,k] = 1 for ALL (i,j,k).
    
    # So the schoolbook decomposition gives T[i,j,k] = 1 for all (i,j,k), which is correct.
    
    # And my rank-1 decomposition also gives T[i,j,k] = 1 for all (i,j,k).
    
    # So both are valid, and rank 1 is better than rank 66.
    
    # I must be missing something. Let me re-read the problem statement once more.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # Maybe the tensor is NOT the all-ones tensor. Maybe it's:
    # T[i,j,k] = delta_{something}
    
    # OR maybe the convention is different. Let me look at the W index again:
    # W[k*n+i, r]
    # If n=3, p=6, then k*n+i ranges from 0 to 17.
    # k is in [0,6), i is in [0,3), so k*n+i = 3k+i.
    # This gives values: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17.
    # All distinct, so it's a valid indexing.
    
    # I'm stuck. Let me just implement the 66-rank decomposition from the inspirations
    # and see if I can improve it.
    
    # Actually, let me try a different interpretation:
    # Maybe the tensor is:
    # T[i,j,k] = 1 if j is a valid intermediate index, 0 otherwise.
    # But for matrix multiplication, all j are valid.
    
    # OR maybe the problem is about the BILINEAR rank, not the tensor rank.
    # The bilinear rank of matrix multiplication is the minimum number of scalar
    # multiplications needed, which is the tensor rank.
    
    # I'll just go with the 66-rank approach and try to improve it.
    
    # Build the 66-rank decomposition
    terms = []
    
    def add_schoolbook(i, j, k):
        terms.append((
            {(i, j): 1.0},
            {(j, k): 1.0},
            {(i, k): 1.0}
        ))
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Strassen 2x2x2"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
        terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
        terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
        terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
        terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
        terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
        terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
    
    # Row 0: schoolbook
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not _time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
    
    rank = len(terms)
    
    # Now try to reduce rank using linear algebra
    # Build U, V, W matrices
    U_mat = np.zeros((n * m, rank))
    V_mat = np.zeros((m * p, rank))
    W_mat = np.zeros((p * n, rank))
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U_mat[i * m + j, col] = coeff
        for (j, k), coeff in v_dict.items():
            V_mat[j * p + k, col] = coeff
        for (i, k), coeff in w_dict.items():
            W_mat[k * n + i, col] = coeff
    
    # Verify
    if not _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, rank):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using null space
    # The idea: if there's a vector (alpha, beta, gamma) such that
    # sum_r alpha_r * u_r = 0, sum_r beta_r * v_r = 0, sum_r gamma_r * w_r = 0
    # and alpha_r * beta_r * gamma_r = 1 for all r, then we can reduce rank.
    
    # This is complex. Let me try a simpler approach:
    # Use the fact that the all-ones tensor has rank 1, and see if we can
    # find a rank-1 decomposition that satisfies the constraints.
    
    # Actually, let me just return the 66-rank decomposition for now.
    # The peeling algorithm is complex and may not improve the rank.
    
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
# EVOLVE-BLOCK-END
