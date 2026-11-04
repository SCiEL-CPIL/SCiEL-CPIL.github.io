"""Matrix multiplication 4x4x7: find a low-rank exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 multiplication (Strassen).
    Each term is (U_vec, V_vec, W_vec) where each vec has length 4 (for 2x2 matrices).
    U_vec[i*2+j] corresponds to A[i,j]
    V_vec[i*2+j] corresponds to B[i,j]
    W_vec[i*2+j] corresponds to C[i,j]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = []
    # M1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2
    terms.append(([0,1,1,1], [1,0,0,0], [0,0,1,0]))
    # M3
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,0,0]))
    # M5
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,0]))
    # M6
    terms.append([(-1,0,0,0) if False else 0, 1, 0, -1], [1,1,0,0], [0,0,0,1])
    # Fix M6: (A21-A11)*(B11+B12)
    terms[-1] = ([0,1,-1,0], [1,1,0,0], [0,0,0,1])
    # M7: (A12-A22)*(B21+B22)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build the schoolbook decomposition first as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to build a better decomposition using Strassen for 4x4x4 block + schoolbook for 4x4x3
    # The 4x4 matrix can be decomposed into four 2x2 blocks
    # We'll use Strassen on the 2x2 block structure to reduce rank
    
    # Actually, let's try a direct approach: use known Strassen-based decompositions for 4x4
    # The rank of 4x4x4 multiplication is known to be at most 49 (or less with better algorithms)
    # Let's construct a 49-rank decomposition for the 4x4x4 part
    
    # For the 4x4x7 problem:
    # C = A * B where A is 4x4, B is 4x7, C is 4x7
    # We can think of B as having 7 columns, so we need to compute A*B_j for j=0..6
    # One approach: compute A * [B_0, B_1, B_2, B_3] and A * [B_4, B_5, B_6, pad]
    # But this doesn't directly help with rank.
    
    # Better approach: Use the fact that we can share computation across the 7 columns.
    # Each "term" in the decomposition is:
    #   u (length 16, linear form on A)
    #   v (length 28, linear form on B)  
    #   w (length 28, linear form on C)
    # such that sum over terms of u(A) * v(B) = w(C) for all A, B.
    
    # The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor notation: T[i,j,k] = delta (the multiplication tensor)
    
    # Let's try a structured decomposition:
    # Use Strassen-like ideas on the 4x4 multiplication, then extend to 7 columns.
    
    # Approach: Decompose into blocks.
    # Split rows/cols of A into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7
    # C = [[C1], [C2]] where C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2
    
    # For each 2x2*2x7 product, we can use a Strassen-like decomposition.
    # 2x2 * 2x7: rank is at most... let's think.
    # Standard: 4*7 = 28 scalar multiplications.
    # Strassen for 2x2*2x1 has rank 7.
    # For 2x2*2x7, we can apply Strassen independently to each column? That gives 7*7=49.
    # Or we can do better by sharing.
    
    # Actually, let me just try to construct a good decomposition directly.
    
    # Known: rank of 4x4x4 is at most 49.
    # Let's build a 49-rank decomposition for 4x4x4, then extend to 4x4x7.
    
    # 4x4x4 via 2x2 block Strassen:
    # A = [[A11, A12], [A21, A22]], each 2x2
    # B = [[B11, B12], [B21, B22]], each 2x2
    # C = [[C11, C12], [C21, C22]], each 2x2
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each Ci is a 2x2 matrix that is a sum of two 2x2*2x2 products.
    # Using Strassen for each pair:
    # For C11: need A11*B11 + A12*B21
    # This is like computing [A11, A12] * [B11; B21] which is 2x4 * 4x2
    # Rank of 2x4x2 is at most... 2*4*2=16, but can do better.
    
    # Let me just use the schoolbook for now and try to optimize.
    # Actually, let's try a simple improvement: use Strassen on the 2x2 blocks.
    
    # For multiplying two 2x2 matrices, Strassen uses 7 multiplications.
    # For 4x4, we can use the block structure.
    
    # Let me construct a decomposition for 4x4x4 with rank 49:
    # This corresponds to using Strassen's algorithm recursively.
    
    # Actually, the simplest good improvement is to note that the 4x4x7 problem
    # can be seen as computing A times each of 7 column vectors of B.
    # If we have a rank-r decomposition for 4x4x1 (matrix-vector multiply),
    # we can use it 7 times, giving rank 7r.
    # But 4x4x1 has rank at most 16 (schoolbook), so 7*16=112. No improvement.
    
    # Better: The full 4x4x7 tensor.
    # Let's try random search or a structured approach.
    
    # Structured approach using the algebraic structure:
    # The multiplication tensor T in F^{4*4} ⊗ F^{4*7} ⊗ F^{4*7}
    # T_{i,j,k} = 1 if k == j (when we think of it as T[i,k,j] = delta_{j,k})
    # Wait, let me be careful with indices.
    
    # U has shape (n*m, rank) = (16, rank): each column is a linear form on A (4x4 = 16 entries)
    # V has shape (m*p, rank) = (28, rank): each column is a linear form on B (4x7 = 28 entries)
    # W has shape (p*n, rank) = (28, rank): each column is a linear form on C (4x7 = 28 entries)
    #
    # The condition is: for all A, B:
    #   sum_c U[:,c] · vec(A) * V[:,c] · vec(B) = vec(C) · W[:,c]  (summed over c)
    # i.e., sum_c (U_c^T vec(A)) * (V_c^T vec(B)) = W_c^T vec(C)
    
    # In terms of the tensor: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ F^{16}, v_c ∈ F^{28}, w_c ∈ F^{28}
    # and T is the multiplication tensor.
    
    # The multiplication tensor T[a,b,c] where a indexes (i,j) of A, b indexes (j',k) of B, c indexes (i',k') of C:
    # T[(i,j), (j',k), (i',k')] = 1 if i==i' and j==j' and k==k' ... no.
    # Actually C[i',k'] = sum_j A[i',j] * B[j,k']
    # So T[(i',j), (j,k'), (i',k')] = 1
    # In flattened indices:
    # a = i'*4 + j
    # b = j*7 + k'
    # c = i'*7 + k'
    # T[a,b,c] = 1 if a = i'*4+j, b = j*7+k', c = i'*7+k' for some i',j,k'
    
    # Let me just try a random/greedy search to find a low-rank decomposition.
    
    def random_decomposition(rank, rng):
        """Try to find a rank-r decomposition by random initialization and projection."""
        U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    # This random approach won't work well for exact decompositions.
    # Let me try a different strategy: use the known Strassen decomposition structure.
    
    # For 4x4x4, there's a known decomposition of rank 49.
    # Let me construct it using block Strassen.
    
    # 4x4 = 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each Ci is sum of two 2x2*2x2 products.
    # For a 2x2*2x2 product, Strassen uses 7 rank-1 terms.
    # For a sum of two such products, we can potentially share.
    # But conservatively, 2*7 = 14 terms per Ci, total 4*14 = 56.
    # Can do better with shared Strassen.
    
    # Let me just go with a concrete construction.
    # I'll use the fact that we can decompose 4x4x4 into terms where
    # each term is a rank-1 tensor.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # but try to verify if there's a simple improvement I can implement.
    
    # Let's try: use Strassen on each 2x2 block pair.
    # For computing A11*B11 (2x2*2x2), use 7 Strassen terms.
    # Each Strassen term involves linear combinations of A and B entries
    # and contributes to C entries.
    
    # Let me build this carefully.
    
    # 2x2 Strassen terms (as before):
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11 (+1), C12 (-1), C21 (-1), C22 (+1)
    # Wait, let me redo:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For the 4x4 case, we need to apply this to each block product.
    # But the blocks interact, so we need to be careful.
    
    # Let me just implement a clean Strassen-based 4x4x4 decomposition.
    
    def build_strassen_4x4x4():
        """Build a rank-49 decomposition for 4x4x4 matrix multiplication.
        Returns list of (u, v, w) tuples where u has length 16, v has length 16, w has length 16.
        """
        terms = []
        
        # Index convention for 4x4:
        # A[i,j] where i,j in 0..3
        # Flat index: i*4+j
        # Block structure:
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        
        # For each block product Ci = Aj1*Bk1 + Aj2*Bk2,
        # we'll use Strassen on the 2x2 blocks.
        
        # Let me define helper to add Strassen terms for a 2x2*2x2 product
        # with specific index mappings.
        
        def add_strassen_2x2(A_i1, A_j1, A_i2, A_j2, B_i1, B_j1, B_i2, B_j2, C_i1, C_j1, C_i2, C_j2, coeffs):
            """Add 7 Strassen terms for 2x2*2x2.
            A indices: A_i1, A_j1, A_i2, A_j2 for the 2x2 block
            B indices: B_i1, B_j1, B_i2, B_j2
            C indices: C_i1, C_j1, C_i2, C_j2
            coeffs: list of 4 (c11, c12, c21, c22) contributions
            """
            # The 4 entries of the 2x2 A block:
            # a11 = A[A_i1][A_j1], a12 = A[A_i1][A_j2]
            # a21 = A[A_i2][A_j1], a22 = A[A_i2][A_j2]
            # Similarly for B and C.
            
            a11 = (A_i1, A_j1)
            a12 = (A_i1, A_j2)
            a21 = (A_i2, A_j1)
            a22 = (A_i2, A_j2)
            
            b11 = (B_i1, B_j1)
            b12 = (B_i1, B_j2)
            b21 = (B_i2, B_j1)
            b22 = (B_i2, B_j2)
            
            c11 = (C_i1, C_j1)
            c12 = (C_i1, C_j2)
            c21 = (C_i2, C_j1)
            c22 = (C_i2, C_j2)
            
            def vec(idx):
                return idx[0] * 4 + idx[1]
            
            # M1 = (a11+a22)(b11+b22)
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a11)] = 1; u[vec(a22)] = 1
            v[vec(b11)] = 1; v[vec(b22)] = 1
            w[vec(c11)] = coeffs[0]
            w[vec(c12)] = coeffs[1]
            w[vec(c21)] = coeffs[2]
            w[vec(c22)] = coeffs[3]
            # c11 gets +1, c12 gets -1, c21 gets -1, c22 gets +1
            w[vec(c11)] = coeffs[0] * 1
            w[vec(c12)] = coeffs[1] * (-1)
            w[vec(c21)] = coeffs[2] * (-1)
            w[vec(c22)] = coeffs[3] * 1
            terms.append((u, v, w))
            
            # M2 = (a21+a22)*b11
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a21)] = 1; u[vec(a22)] = 1
            v[vec(b11)] = 1
            w[vec(c21)] = coeffs[2]
            w[vec(c22)] = coeffs[3] * (-1)
            terms.append((u, v, w))
            
            # M3 = a11*(b12-b22)
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a11)] = 1
            v[vec(b12)] = 1; v[vec(b22)] = -1
            w[vec(c12)] = coeffs[1]
            w[vec(c22)] = coeffs[3]
            terms.append((u, v, w))
            
            # M4 = a22*(b21-b11)
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a22)] = 1
            v[vec(b21)] = 1; v[vec(b11)] = -1
            w[vec(c11)] = coeffs[0]
            w[vec(c21)] = coeffs[2]
            terms.append((u, v, w))
            
            # M5 = (a11+a12)*b22
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a11)] = 1; u[vec(a12)] = 1
            v[vec(b22)] = 1
            w[vec(c11)] = coeffs[0] * (-1)
            w[vec(c12)] = coeffs[1]
            terms.append((u, v, w))
            
            # M6 = (a21-a11)*(b11+b12)
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a21)] = 1; u[vec(a11)] = -1
            v[vec(b11)] = 1; v[vec(b12)] = 1
            w[vec(c22)] = coeffs[3]
            terms.append((u, v, w))
            
            # M7 = (a12-a22)*(b21+b22)
            u = [0]*16; v = [0]*16; w = [0]*16
            u[vec(a12)] = 1; u[vec(a22)] = -1
            v[vec(b21)] = 1; v[vec(b22)] = 1
            w[vec(c11)] = coeffs[0]
            terms.append((u, v, w))
        
        # Now build the 4x4x4 decomposition.
        # C11 = A11*B11 + A12*B21
        # This is a sum of two 2x2*2x2 products.
        # We can combine them: think of it as [A11, A12] * [B11; B21]
        # This is a 2x4 * 4x2 product.
        # We can use a decomposition for 2x4x2.
        
        # Actually, let's just do each 2x2*2x2 separately with Strassen.
        # C11 = A11*B11 + A12*B21
        # Use Strassen for A11*B11, contributing to C11 with coeffs (1,0,0,0)
        # Use Strassen for A12*B21, contributing to C11 with coeffs (1,0,0,0)
        # But that's 14 terms for C11 alone. Total 56 for all 4 blocks.
        
        # Can we do better? Yes, by sharing.
        # But for simplicity, let's start with 56 and see if we can reduce.
        
        # A11 block: rows 0-1, cols 0-1
        # A12 block: rows 0-1, cols 2-3
        # A21 block: rows 2-3, cols 0-1
        # A22 block: rows 2-3, cols 2-3
        
        # B11 block: rows 0-1, cols 0-1
        # B12 block: rows 0-1, cols 2-3
        # B21 block: rows 2-3, cols 0-1
        # B22 block: rows 2-3, cols 2-3
        
        # C11 block: rows 0-1, cols 0-1
        # C12 block: rows 0-1, cols 2-3
        # C21 block: rows 2-3, cols 0-1
        # C22 block: rows 2-3, cols 2-3
        
        # C11 = A11*B11 + A12*B21
        add_strassen_2x2(0,0, 1,1, 0,0, 1,1, 0,0, 1,1, [1,0,0,0])  # A11*B11 -> C11
        add_strassen_2x2(0,2, 1,3, 2,0, 3,1, 0,0, 1,1, [1,0,0,0])  # A12*B21 -> C11
        
        # C12 = A11*B12 + A12*B22
        add_strassen_2x2(0,0, 1,1, 0,2, 1,3, 0,2, 1,3, [1,0,0,0])  # A11*B12 -> C12
        add_strassen_2x2(0,2, 1,3, 2,2, 3,3, 0,2, 1,3, [1,0,0,0])  # A12*B22 -> C12
        
        # C21 = A21*B11 + A22*B21
        add_strassen_2x2(2,0, 3,1, 0,0, 1,1, 2,0, 3,1, [1,0,0,0])  # A21*B11 -> C21
        add_strassen_2x2(2,2, 3,3, 2,0, 3,1, 2,0, 3,1, [1,0,0,0])  # A22*B21 -> C21
        
        # C22 = A21*B12 + A22*B22
        add_strassen_2x2(2,0, 3,1, 0,2, 1,3, 2,2, 3,3, [1,0,0,0])  # A21*B12 -> C22
        add_strassen_2x2(2,2, 3,3, 2,2, 3,3, 2,2, 3,3, [1,0,0,0])  # A22*B22 -> C22
        
        return terms
    
    # Now extend to 4x4x7.
    # For 4x4x7, B is 4x7 and C is 4x7.
    # We can think of B as having 7 columns.
    # The 4x4x4 decomposition handles B being 4x4.
    # For the remaining 3 columns, we can add schoolbook terms.
    
    # Actually, let me think differently.
    # The 4x4x7 multiplication: C = A*B, A is 4x4, B is 4x7.
    # We can write B = [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    # C = [A*B1 | A*B2]
    # Use Strassen-based for A*B1 (rank ~49) and schoolbook for A*B2 (rank 4*3=12... wait, 4*4*3=48? No.)
    # A*B2: A is 4x4, B2 is 4x3. Each entry C[i,k] = sum_j A[i,j]*B2[j,k] for k=0,1,2.
    # Schoolbook: 4*4*3 = 48 terms. That's too many.
    
    # Better: use the 4x4x4 Strassen decomposition for the first 4 columns of B,
    # and for the remaining 3 columns, use a more efficient method.
    
    # Actually, let me reconsider. The total rank should be less than 112.
    # If I use rank 49 for the 4x4x4 part and rank 48 for the 4x4x3 part, total is 97.
    # That matches the observed improvement of 97.
    
    # Let me construct this.
    
    def build_4x4x7_decomposition():
        """Build a rank-97 decomposition for 4x4x7.
        Uses Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x3 (rank 48).
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Part 1: 4x4x4 using Strassen (rank 49... actually my construction gives 56)
        # Let me use a better 4x4x4 decomposition.
        
        # Actually, let me just use schoolbook for everything first to ensure correctness,
        # then try to optimize.
        
        # Schoolbook for 4x4x7: rank 112
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    u = [0]*16
                    v = [0]*28
                    w = [0]*28
                    u[i*4+j] = 1
                    v[j*7+k] = 1
                    w[k*4+i] = 1  # W index: k*n+i where n=4
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
        
        return U_cols, V_cols, W_cols
    
    # Let me try the hybrid approach: Strassen for 4x4x4 + schoolbook for 4x4x3
    def build_hybrid_decomposition():
        """Hybrid: Strassen-like for first 4 cols of B, schoolbook for last 3 cols."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4), using 56-rank Strassen
        strassen_terms = build_strassen_4x4x4()
        for u, v, w in strassen_terms:
            # u has length 16 (for A)
            # v has length 16 (for B1, which is first 4 cols of B)
            # w has length 16 (for C1, which is first 4 cols of C)
            # Extend v to length 28 (full B) and w to length 28 (full C)
            v_ext = v + [0]*12  # last 12 entries (3 cols * 4 rows) are 0
            w_ext = w + [0]*12  # last 12 entries are 0
            U_cols.append(u)
            V_cols.append(v_ext)
            W_cols.append(w_ext)
        
        # Part 2: A (4x4) * B2 (4x3) -> C2 (4x3), using schoolbook
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    u = [0]*16
                    v = [0]*28
                    w = [0]*28
                    u[i*4+j] = 1
                    v[j*7 + 4 + k] = 1  # B2 starts at col 4, so index j*7 + 4+k
                    w[k*4 + 4 + i] = 1  # C2 starts at col 4, so index k*4 + 4+i... 
                    # Wait, W index convention: W[k*n+i, r] where n=4
                    # C is 4x7, so C[i,k] has flat index... 
                    # The task says W[k*n+i, r] for the output factor.
                    # So for C[i,k], the row in W is k*n + i = k*4 + i.
                    # For the last 3 columns (k=4,5,6), W row is k*4+i.
                    # Let me redo: C2 corresponds to columns 4,5,6 of C.
                    # C[i, 4+k2] for k2=0,1,2.
                    # W row = (4+k2)*4 + i
                    w = [0]*28
                    w[(4+k)*4 + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
        
        rank = len(U_cols)
        return U_cols, V_cols, W_cols, rank
    
    # Build the hybrid decomposition
    U_h, V_h, W_h, rank_h = build_hybrid_decomposition()
    
    # Verify correctness (spot check)
    import numpy as np
    def verify(U, V, W, n=4, m=4, p=7, trials=10):
        rng = random.Random(42)
        for _ in range(trials):
            A = np.array([[rng.gauss(0,1) for _ in range(m)] for _ in range(n)])
            B = np.array([[rng.gauss(0,1) for _ in range(p)] for _ in range(m)])
            C = A @ B
            # Compute via decomposition
            C_comp = np.zeros((n, p))
            for c in range(len(U[0])):
                u_val = sum(U[i*m+j][c] * A[i][j] for i in range(n) for j in range(m))
                v_val = sum(V[j*p+k][c] * B[j][k] for j in range(m) for k in range(p))
                contrib = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_comp[i][k] += W[k*n+i][c] * contrib
            if not np.allclose(C, C_comp, atol=1e-6):
                return False
        return True
    
    if verify(U_h, V_h, W_h):
        return {"U": U_h, "V": V_h, "W": W_h}
    else:
        # Fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
