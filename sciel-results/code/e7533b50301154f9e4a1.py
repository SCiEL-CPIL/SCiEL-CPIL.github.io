import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
    n, m, p = DIMENSIONS  # 3, 3, 7

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

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # Attempt to find a valid rank-22 or rank-23 factorization for 3x3x3
    # using known algebraic constructions.
    
    # The key insight: for 3x3x3 matrix multiplication, the tensor is:
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    # So we need: sum_r u_r[i,l] * v_r[l,k] * w_r[i,k] = 1 for all (i,k) where the output is C[i,k]
    
    # Wait, let me re-read the convention:
    # W[k*n+i, r] - so W is indexed by (output_row_of_B, output_col_of_A) = (k, i)
    # U[i*m+j, r] - U indexed by (row_of_A, col_of_A) = (i, j)  
    # V[j*p+k, r] - V indexed by (row_of_B, col_of_B) = (j, k)
    
    # The product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for each (i, j, k) term in the matrix multiplication A(3x3) * B(3x7)
    
    # Actually for the full 3x3x7: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # For 3x3x3: sum_r u[i,l]*v[l,k]*w[i,k] = 1 for all (i,k) output positions... 
    # No wait. Let me think again.
    
    # Matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has one free (i,j) index for A, one (j,k) for B, and one (k,i) for C.
    # The decomposition requires: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = delta for the tensor
    # But actually the multiplication tensor M_3x3x3[i,j,k] = 1 if we think of it as
    # the bilinear form that computes all entries.
    
    # More carefully: the matrix multiplication of A(3x3)*B(3x3) gives C(3x3) where
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    
    # The multiplication tensor is a trilinear form: <M, A, B, C> = sum_{i,j,k} M[i,j,k] * A[i,j]*B[j,k]*C[k,i]
    # where M[i,j,k] = 1 for all i,j,k.
    
    # Wait no. Let me be precise. The multiplication of A(i,j) and B(j,k) produces C(i,k).
    # So C(i,k) = sum_j A(i,j) B(j,k).
    
    # As a tensor: M[i,j,k] such that C(i,k) = sum_j M[i,j,k] A[i,j] B[j,k]... 
    # But that's not right either because the output index is (i,k) not (j,k).
    
    # The standard view: the matrix multiplication tensor T_{m} has indices (i, j, k) where:
    # - i is the row of A and row of C  
    # - j is the col of A and row of B
    # - k is the col of B and col of C
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W[k,i,r] corresponds to C[i,k] (note the index swap to match k*n+i convention)
    
    # For 3x3x3: need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1,2}
    
    # Laderman's rank 23: I'll try to construct one using the known formula.
    # The standard Laderman factors use entries in {0, 1, -1}.
    
    # Let me try the known 23-term Laderman decomposition:
    # Using the formulation from Strassen's work and Laderman's improvement.
    
    # A known valid rank-22 decomposition was found by Landsberg et al. 
    # Let me try a different approach: use the explicit formula.
    
    # Actually, let me try to find the factors computationally.
    # For rank r, we need U (9xr), V (9xr), W (9xr) such that:
    # sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1 for all i,j,k
    
    # This is a system of 27 equations with r * 27 unknowns (for rank r).
    # It's nonlinear, but we can try r=22 or r=23 with a search.
    
    # For r=23, let me use a known parametrization.
    # The Laderman factors can be described as follows (from literature):
    
    # Let me use a verified approach: try rank 23 with specific structure.
    # Known: for 3x3x3, the minimum rank is between 22 and 23 (over complex numbers).
    # Over rationals/integers, it might be higher.
    
    # Since we need integer/half-integer entries, let me try to construct explicitly.
    # 
    # Strategy: Use the following known identity for 3x3 matrix multiplication.
    # Define 23 "intermediate" quantities that are bilinear forms in A and B,
    # and express C as products of these intermediates.
    
    # I'll use the explicit Laderman-23 formula. Here's one valid set:
    # The factors are 9-vectors (representing 3x3 matrices in row-major order).
    
    # Let me generate them systematically using the known construction:
    # Split the 3x3 matrices into overlapping 2x2 blocks and use Strassen.
    
    # Actually, the cleanest approach: I know that 2x2x2 needs rank 7 (Strassen).
    # For 3x3x3, Laderman showed rank 23 is achievable.
    
    # Let me hardcode a VALID Laderman 23 set. I'll use the one from:
    # "Lower Bound of Matrix Multiplication Tensors" - verified formulation.
    
    # Here is a valid rank-23 decomposition for 3x3x3:
    # u_r, v_r, w_r are 9-vectors for r=0,...,22
    
    L23_u = [
        [1,0,0, 0,0,0, 0,0,0],
        [1,0,0, 0,0,0, 0,1,0],
        [0,1,0, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,1,1],
        [0,1,0, 0,0,0, 0,1,1],
        [1,0,0, 0,0,0, 0,0,1],  # duplicate, need to fix
        [1,0,0, 0,0,0, 1,0,0],
        [1,0,0, 0,0,0, 1,1,0],
        [0,1,0, 0,0,0, 1,1,0],
        [1,0,0, 0,0,0, 1,0,1],
        [1,0,0, 0,0,0, 1,1,1],
        [0,1,0, 0,0,0, 1,1,1],
        [1,0,0, 0,0,0, 0,0,1],
        [0,0,0, 1,0,0, 0,0,1],
        [0,0,0, 1,0,0, 0,1,1],
        [0,0,0, 0,1,0, 0,1,1],
        [0,0,0, 1,0,0, 1,0,0],
        [0,0,0, 1,0,0, 1,1,0],
        [0,0,0, 0,1,0, 1,1,0],
        [0,0,0, 1,0,0, 1,0,1],
        [0,0,0, 1,0,0, 1,1,1],
        [0,0,0, 0,1,0, 1,1,1],
    ]
    
    # This is getting too complicated to do by hand. Let me take a different approach.
    # I'll use a constructive method: the 3x3x3 multiplication can be done by
    # 3 Strassen steps of 2x2 + correction terms, but that's not rank-optimal.
    
    # Better approach: I know the answer must be between 22 and 63.
    # Let me just try to find ANY valid sub-63 factorization.
    
    # Simple improvement: Instead of pure schoolbook (63), split into
    # two 3x3x3 blocks (each can use rank <= 23 if we find valid factors)
    # and one 3x3x1 block (rank 9).
    
    # For the 3x3x1 block (3x3 times 3x1): C[i] = sum_j A[i,j]*B[j]
    # This needs rank 9 (each output C[i] needs all 3 products), and 9 is minimal.
    
    # For 3x3x3: Let me try to find a valid rank-23 factorization.
    # I'll search computationally.
    
    # Actually, let me use a different known result: 
    # The rank of 3x3x3 over Q is at most 23 (Laderman) and at least 22 (Landsberg).
    # For our purposes, even rank 24 would give total 24+24+9=57 < 63.
    
    # Let me try a different strategy: use block-Strassen.
    # Split 3x3 into 2x2 blocks + 1 extra row/col.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Similarly for B and C.
    
    # Strassen on the 2x2 block: 7 products
    # Cross terms: A12*B22, A21*B12, A22*B11, and corrections...
    # This gets complicated and doesn't obviously give rank < 23.
    
    # Let me try yet another approach: use the fact that we can add slack.
    # If I can't find a valid rank-23 for 3x3x3, I'll use schoolbook for the 3x3x3 parts
    # but maybe I can find something better for the whole 3x3x7.
    
    # Total schoolbook = 63. Can I do better with a smarter partition?
    
    # Alternative: treat the whole 3x3x7 as one problem.
    # C(3x7) = A(3x3) * B(3x7)
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,...,6
    
    # We need 21 output entries. The naive way uses 63 products.
    # Can we share intermediate computations?
    
    # One idea: compute D[j,k] = B[j,k] and use Strassen-like tricks on A.
    # But A is only 3x3, so there's limited room.
    
    # Let me try: for each pair of output rows (i,i+1) of C, can we compute
    # C[i,k] and C[i+1,k] with fewer than 6*p = 42 products total?
    # C[i,k] + C[i+1,k] = sum_j (A[i,j]+A[i+1,j]) * B[j,k] -> 3*p = 21 products for sums
    # C[i,k] - C[i+1,k] = sum_j (A[i,j]-A[i+1,j]) * B[j,k] -> 3*p = 21 products for diffs
    # Then C[i,k] = (sum_k + diff_k)/2, C[i+1,k] = (sum_k - diff_k)/2
    # Total: 21 + 21 = 42 products for 2*p = 14 outputs. Same as before!
    
    # What about C[i,k] + C[i+1,k] + C[i+2,k]?
    # That doesn't help directly.
    
    # For 3 rows, we can compute:
    # S1 = A[0,:] + A[1,:] + A[2,:]  (3 values)
    # S2 = A[0,:] - A[1,:]           (3 values)  
    # S3 = A[0,:] - A[2,:]           (3 values)
    # Then A[0,:] = (S1+S2+S3)/3... no this isn't right.
    
    # Actually: 
    # A[0,j] = (S1 + S2 + S3)/3
    # A[1,j] = (S1 - S2)/3
    # A[2,j] = (S1 - S3)/3
    # Check: (S1+S2+S3)/3 + (S1-S2)/3 + (S1-S3)/3 = (3S1)/3 = S1 = A0+A1+A2. Correct? 
    # Wait: A0 = (S1+S2+S3)/3, A1 = (S1-S2)/3, A2 = (S1-S3)/3
    # A0+A1+A2 = (S1+S2+S3+S1-S2+S1-S3)/3 = 3S1/3 = S1. Yes!
    # But we need S1, S2, S3 which are 3 each = 9 values to reconstruct A.
    # Then C[:,k] = A @ B[:,k]. Using the S forms:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,k] = (1/3) sum_j (S1_j + S2_j + S3_j) B[j,k]
    # C[1,k] = (1/3) sum_j (S1_j - S2_j) B[j,k]
    # C[2,k] = (1/3) sum_j (S1_j - S3_j) B[j,k]
    
    # Let T1[k] = sum_j S1_j * B[j,k]  (7 values, 3*7=21 products)
    # T2[k] = sum_j S2_j * B[j,k]  (7 values, 3*7=21 products)
    # T3[k] = sum_j S3_j * B[j,k]  (7 values, 3*7=21 products)
    
    # Then C[0,k] = (T1[k]+T2[k]+T3[k])/3
    # C[1,k] = (T1[k]-T2[k])/3
    # C[2,k] = (T1[k]-T3[k])/3
    
    # Total products: 21*3 = 63. Same as schoolbook! No improvement.
    
    # The issue is that forming S1, S2, S3 from A is "free" (just additions of A entries),
    # and the actual multiplications are still 3*3*7 = 63.
    
    # So this particular approach doesn't help. We need a true low-rank factorization.
    
    # Let me go back to finding a valid 3x3x3 factorization.
    # I'll use a more careful construction.
    
    # Known valid Laderman rank-23 factors (verified from literature):
    # Using the notation where each factor is a 3x3 matrix (flattened to 9-vector):
    
    def make_23():
        """
        Return 23 (u, v, w) triples where u, v, w are 9-element lists.
        These implement 3x3x3 matrix multiplication:
        sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] = 1 for all i,j,k in {0,1,2}
        """
        # I'll use the explicit Laderman factors.
        # Reference: Laderman, "Combinatoric Problems in Algebra" (1976)
        
        # The 23 bilinear forms for 3x3 matrix multiplication:
        # They are built from the 7 Strassen-like terms for each 2x2 submatrix
        # plus 16 correction terms.
        
        # Let me use a known valid set. After careful thought, here are the factors:
        # Format: (u[9], v[9], w[9])
        
        factors = []
        
        # I'll construct these using the known algebraic identity.
        # For 3x3: C[i,k] = sum_j A[i,j] B[j,k]
        
        # Use the following 23-term formula:
        # Let me define helper to create sparse 9-vectors.
        def vec9(d):
            """d is a dict {(i,j): val} mapping to a 9-element list"""
            v = [0]*9
            for (a,b), val in d.items():
                v[a*3+b] = val
            return v
        
        # The 23 Laderman terms (verified):
        # Term structure uses overlapping 2x2 Strassen patterns
        
        # Let me use a different known-valid approach: 
        # Explicitly list 23 triples that I can verify.
        
        # After much deliberation, let me use this known-valid set:
        # (These are from the rank-23 Laderman decomposition, verified algebraically)
        
        L23 = [
            # (u, v, w) as 9-element lists, row-major 3x3
            # u represents coefficient for A matrix entry
            # v represents coefficient for B matrix entry  
            # w represents coefficient for C matrix entry
            
            # The constraint is: sum_r u[r][i,j] * v[r][j,k] * w[r][k,i] = delta(i,j,k contributes to C[i,k])
            # Actually more precisely: for each output position (i,k), we need
            # sum_j A[i,j] B[j,k] = C[i,k]
            # So: sum_r (sum_j u_r[i,j] A[i,j]) (sum_k' ... ) 
            # No, it's: C[i,k] = sum_r (sum_j u_r[i,j] A[i,j]) * (sum_j v_r[j,k] B[j,k]) * w_r[k,i]... 
            # That's not right either.
            
            # The correct factorization: 
            # Define intermediate M_r = sum_{i,j} u_r[i,j] A[i,j]  (scalar)
            # And N_r = sum_{j,k} v_r[j,k] B[j,k]  (scalar)
            # Then C[i,k] = sum_r M_r * N_r * w_r[i,k]... no.
            
            # Actually the tensor decomposition is:
            # M_333[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
            # where M_333[i,j,k] = 1 for all (i,j,k).
            # This means: for each (i,j,k), sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
            
            # Equivalently, if we think of the multiplication:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Then we want to express each product A[i,j]*B[j,k] as a sum over r of
            # (something involving U_r) * (something involving V_r) * W_r[k,i]
            
            # The factorization says: 
            # A[i,j] B[j,k] = sum_r (U_r[i,j] * a_r) * (V_r[j,k] * b_r) * W_r[k,i]
            # where a_r, b_r are the "intermediates" = U_r·A and V_r·B
            
            # So the decomposition is valid if:
            # sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k
        ]
        
        # Let me just try a computational search for small rank factorizations.
        # For rank 22 or 23, I need to solve the nonlinear system.
        # This is hard in general, but let me try a specific parameterization.
        
        # Alternative: I'll use the fact that rank 24 is known to be achievable,
        # and construct it explicitly.
        
        # Actually, let me use a simpler but valid approach:
        # Use 2x2 Strassen (rank 7) for the top-left 2x2x2 block,
        # and handle the remaining entries with schoolbook.
        
        # Split 3x3 into:
        # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
        # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Top-left 2x2: C[0,0], C[0,1], C[1,0], C[1,1] involve all j=0,1,2
        # So even the "top-left 2x2" of C depends on all of A and B.
        # This makes block decomposition tricky.
        
        # Let me just go with schoolbook for safety, but try one more thing:
        # Can I find a valid rank < 63 for the full 3x3x7?
        
        # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
        # Upper bound (schoolbook): n*m*p = 63
        
        # For 3x3x3: rank is 23 (Laderman, over C) or possibly 22.
        # For 3x3x7: at most 3*23 = 69 (if we use 3 copies of 3x3x3... no that's wrong)
        # Actually: 3x3x7 can be seen as 3 independent 3x3x1 problems... no.
        # It's 7 independent 3x3 vector outputs: C[:,k] = A * B[:,k]
        # Each C[:,k] needs 3*3 = 9 products naively.
        # Total: 7*9 = 63.
        
        # Can we share work across the 7 columns?
        # If we compute intermediate quantities that are linear in B's columns,
        # we might save some products.
        
        # For example, let S_j = sum_{k} alpha_{j,k} B[j,k] for some coefficients.
        # But this doesn't directly help since C[i,k] depends on specific B[j,k].
        
        # I think the best safe bet is schoolbook at 63, unless I can find
        # a verified low-rank factorization.
        
        # Let me make one final attempt with a known-valid construction.
        
        # The following is a valid rank-23 Laderman factorization for 3x3x3:
        # I'll define it using the explicit formula from the literature.
        
        # Based on careful study, here are 23 valid factors:
        # u_r, v_r, w_r are 3x3 matrices (as 9-vectors in row-major order)
        
        U23 = []
        V23 = []
        W23 = []
        
        # Using the construction: for each of the 9 (i,j) positions of A and 
        # 9 (j,k) positions of B, we have products that contribute to C.
        # Laderman's trick combines some of these.
        
        # Let me use a parameterized family that I know works:
        # The key terms are:
        
        def add_term(u, v, w):
            U23.append(u)
            V23.append(v)
            W23.append(w)
        
        # Helper to create a 9-vector with specific entries
        def mk(a00=0,a01=0,a02=0, a10=0,a11=0,a12=0, a20=0,a21=0,a22=0):
            return [a00,a01,a02, a10,a11,a12, a20,a21,a22]
        
        # The 23 Laderman terms (this is a verified set):
        # I'll use the formulation where we have:
        # 9 terms of form: u=e_ij, v=e_jk, w=e_ki (diagonal/schoolbook-like)
        # plus 14 correction terms
        
        # Actually, I realize I should just verify whatever I claim to produce.
        # Let me take a completely different approach:
        
        # I'll use rank-24 factors which are easier to construct:
        # Use Strassen-7 for 2x2x2 (applied 4 times with shifts) + corrections
        # But this is also complex.
        
        # Given the difficulty, let me just return schoolbook (rank 63) to ensure correctness.
        # The score will be 63, same as the parent's 63, so score ratio = 1.0.
        # This is the safe choice.
        
        return None  # Signal that we should use schoolbook
    
    # Let me try a more creative approach to get below 63.
    # 
    # Observation: The problem is 3x3x7. What if I use a rank-r factorization
    # for a "wider" matrix multiplication and restrict it?
    #
    # For 3xNxM matrix multiplication, the rank is at most N*M*3 (schoolbook).
    # Can we do better?
    #
    # Known results:
    # - 2xNxM: rank 2NM - 2(N+M) + 2 + omega where... this is getting complex.
    # - For 3x3x3: rank 23
    # - For 3x3x7: unknown exact value, but between 21 and 63.
    #
    # Let me try to use the 3x3x3 rank-23 result.
    # 3x3x7 = first 3 columns of 3x3x7 give a 3x3x3 (rank 23)
    # Next 3 columns give another 3x3x3 (rank 23)  
    # Last column gives a 3x3x1 (rank 9)
    # Total: 23 + 23 + 9 = 55
    #
    # This would give score 63/55 = 1.145!
    #
    # BUT I need valid rank-23 factors for 3x3x3.
    #
    # Let me try to find them using a systematic search.
    # I'll set up the problem as finding U,V,W in R^{9x23} such that
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in {0,1,2}
    
    if time_left() > 30:
        # Try to find valid 3x3x3 factors via search
        # Use the fact that we can scale: if we find rational solutions,
        # we can multiply by a common denominator to get integers.
        
        # Parametrization: use entries from {0, 1, -1} first.
        # Search over possible 23-term factorizations with small integer entries.
        
        # This is a 9*23 = 207 variable problem per matrix (3 matrices = 621 vars)
        # with 27 nonlinear equations. Too large for exhaustive search.
        
        # Let me try a specific known construction instead.
        
        # From the literature, one valid rank-23 Laderman factorization uses
        # the following 23 terms (I'll verify each):
        
        # The construction is based on the following identity:
        # For 3x3 matrices A, B, C = AB:
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Define 23 bilinear forms M_r(A) and N_r(B) such that
        # C[i,k] = sum_r M_r(A) N_r(B) * coeff_r(i,k)
        
        # Here's a valid set (using {0,1,-1} entries):
        
        # After extensive reference checking, here are 23 valid Laderman factors:
        # Each is (u_9, v_9, w_9) where the condition holds.
        
        # Let me define them explicitly and VERIFY:
        
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        # u = e_{0,0}, v = e_{0,0}, w = e_{0,0}
        # Check: sum_r ... for (i,j,k)=(0,0,0): u[0,0]*v[0,0]*w[0,0] = 1*1*1 = 1 ✓
        # For (i,j,k)=(0,1,0): u[0,0]*v[0,1]*w[0,0] = 1*0*1 = 0 ✓ (v[0,1]=0)
        # etc.
        
        # This is the schoolbook term for (0,0,0). There are 27 such terms.
        # Laderman replaces 27 schoolbook terms with 23 combined terms.
        
        # The key insight: some products can be grouped.
        # For example, A[0,0]*B[0,0] + A[0,0]*B[0,1] can be computed as
        # A[0,0]*(B[0,0]+B[0,1]) which is 1 bilinear form in A times 1 in B.
        
        # Let me try to construct 23 terms:
        
        # Group the 27 products A[i,j]*B[j,k] into 23 groups.
        # Each group g has:
        # - A coefficient vector u_g (9-dim, sparse)
        # - B coefficient vector v_g (9-dim, sparse)  
        # - C coefficient vector w_g (9-dim, sparse)
        # Such that for the products in group g:
        # sum_{(i,j,k) in g} u_g[i,j] v_g[j,k] w_g[k,i] = 1 for each (i,j,k) in g
        # And = 0 for (i,j,k) not in g.
        
        # This is still complex. Let me just use schoolbook.
        
        pass  # No valid 3x3x3 factors found; use schoolbook
    
    # Final answer: schoolbook
    result = schoolbook()
    
    # Verify to be safe
    if not verify(result['U'], result['V'], result['W']):
        # This should never happen
        raise RuntimeError("Schoolbook verification failed!")
    
    return result
