"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def strassen_2x2x2():
        """
        Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns (U, V, W) each of shape (4, 7) for the 2x2x2 tensor.
        
        Indexing: U indexed by (i*m+j) where i,j in 0..1, m=2
                  V indexed by (j*p+k) where j,k in 0..1, p=2
                  W indexed by (k*n+i) where k,i in 0..1, n=2
        """
        # Strassen's 7 multiplications for 2x2 matrix multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For our tensor: T[i,j,k] = 1 for all i,j,k in 0..1
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        # Each Strassen term r contributes to specific (i,j,k) triples
        # We need: sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1 for all i,j,k
        
        # Let's define the 7 terms:
        # Term 0: M1 = (a+d)(e+h)
        #   a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
        #   This contributes to C[0,0] with +1, C[0,1] with 0, C[1,0] with 0, C[1,1] with +1
        #   Wait, I need to be more careful.
        
        # Let me think of this differently. The Strassen algorithm computes:
        # C = AB where A is 2x2, B is 2x2
        # The 7 intermediate products are linear combinations of A and B entries
        # Each product M_r = (linear combo of A entries) * (linear combo of B entries)
        # Then C entries are linear combos of M_r
        
        # For the tensor decomposition, each rank-1 term corresponds to one "multiplication"
        # In Strassen, we have 7 multiplications, so 7 rank-1 terms
        
        # The key insight: in the standard matrix multiplication tensor,
        # T[i,j,k] = 1 means the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
        # 
        # For a rank-1 decomposition term r, we have:
        # U[i*m+j, r] = coefficient for A[i,j] in the "left factor" of term r
        # V[j*p+k, r] = coefficient for B[j,k] in the "right factor" of term r  
        # W[k*n+i, r] = coefficient for C[i,k] in the "output" of term r
        # 
        # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        #
        # This means: the (i,j,k) coefficient in the tensor = sum over r of the product
        #
        # For Strassen, we can think of it as:
        # Each of the 7 "virtual multiplications" M_r = L_r(A) * R_r(B)
        # where L_r(A) is a linear combination of A entries and R_r(B) is a linear combination of B entries
        # Then C = sum_r coeff_r * (outer product structure)
        
        # Actually, let me just directly construct the U, V, W matrices
        # by working through the Strassen formulas.
        
        # A entries: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B entries: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C entries: C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
        
        # Strassen:
        # M1 = (a+d)(e+h) = ae+ah+de+dh
        # M2 = (c+d)e = ce+de
        # M3 = a(g-h) = ag-ah
        # M4 = d(f-e) = df-de
        # M5 = (a+b)h = ah+bh
        # M6 = (c-a)(e+f) = ce+cf-ae-af
        # M7 = (b-d)(g+h) = bg+bh-dg-dh
        
        # C[0,0] = ae+bg = M1 - M2 + M3 + M4 + M7 - M5  ... let me verify
        # M1 - M2 + M3 + M4 + M7 - M5
        # = (ae+ah+de+dh) - (ce+de) + (ag-ah) + (df-de) + (bg+bh-dg-dh) - (ah+bh)
        # = ae+ah+de+dh - ce-de + ag-ah + df-de + bg+bh-dg-dh - ah-bh
        # = ae + (ah-ah-ah) + (de-de-de) + (dh-dh) + ag + df + bg + (bh-bh) - dg - ce
        # = ae - ah - de + ag + df + bg - dg - ce
        # That's not right. Let me look up the correct Strassen reconstruction.
        
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e  
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Verify C11 = ae+bg:
        # M1 + M4 - M5 + M7
        # = (ae+ah+de+dh) + (df-de) - (ah+bh) + (bg+bh-dg-dh)
        # = ae+ah+de+dh + df-de - ah-bh + bg+bh-dg-dh
        # = ae + (ah-ah) + (de-de) + (dh-dh) + df - bh + bh + bg - dg
        # = ae + df + bg - dg
        # Hmm, that gives ae + bg + df - dg, not ae+bg. Something's wrong.
        
        # Let me use the standard Strassen formulas correctly:
        # p1 = (a + d)(e + h)
        # p2 = (c + d)(e)
        # p3 = (a)(g - h)
        # p4 = (d)(f - e)
        # p5 = (a + b)(h)
        # p6 = (c - a)(e + f)
        # p7 = (b - d)(g + h)
        
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # Verify c11 = ae + bg:
        # p1 + p4 - p5 + p7
        # = (a+d)(e+h) + d(f-e) - (a+b)h + (b-d)(g+h)
        # = (ae+ah+de+dh) + (df-de) - (ah+bh) + (bg+bh-dg-dh)
        # = ae+ah+de+dh+df-de-ah-bh+bg+bh-dg-dh
        # = ae + (ah-ah) + (de-de) + (dh-dh) + df - bh + bh + bg - dg
        # = ae + df + bg - dg
        
        # That's ae + bg + df - dg, which is not ae+bg. There must be an error in my Strassen formulas.
        
        # Let me look this up properly. The correct Strassen algorithm:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M5 + M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 + M3 - M2 - M6
        
        # With A11=a, A12=b, A21=c, A22=d, B11=e, B12=f, B21=g, B22=h:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 - M5 + M4 + M7
        # = (a+d)(e+h) - (a+b)h + d(f-e) + (b-d)(g+h)
        # = ae+ah+de+dh - ah-bh + df-de + bg+bh-dg-dh
        # = ae + (ah-ah) + (de-de) + (dh-dh) - bh + bh + df + bg - dg
        # = ae + df + bg - dg
        
        # Still getting ae + bg + df - dg. This can't be right for C11 = ae + bg.
        
        # I think I'm confusing myself. Let me just directly construct the rank-1 decomposition
        # for the 2x2x2 tensor using the known fact that it has rank 7.
        
        # The 2x2x2 matrix multiplication tensor has a 7-term decomposition.
        # Let me construct it explicitly.
        
        # Actually, for this problem, I don't need to get the Strassen decomposition perfectly 
        # for the base case. Let me use a different approach.
        
        # For the 4x4x9 problem, I'll use the following strategy:
        # Split into 4x4x4 (rank 49 via recursive Strassen) + 4x4x4 (rank 49) + 4x4x1 (rank 16)
        # Total: 114
        
        # But first, let me just implement the schoolbook for now and try to improve.
        # The key is to get a valid decomposition with rank < 144.
        
        # Let me try a simpler approach: use the fact that we can share some structure.
        
        # For 4x4 x 4x9, think of it as 9 separate 4x4 x 4x1 problems.
        # Each 4x4 x 4x1 has rank 16 (schoolbook).
        # But we can potentially share the A factors.
        
        # Actually, the simplest improvement: for each column k, we need
        # sum_j A[i,j] * B[j,k] = C[i,k]
        # This is 4x4 = 16 products per column, 9 columns = 144 total.
        
        # To reduce, we need to find algebraic identities that let us compute
        # multiple C[i,k] values from shared intermediate products.
        
        # For now, let me just return the schoolbook and see if we can improve.
        return None
    
    # Construct Strassen-based decomposition for 4x4x4, then extend to 9 columns
    # 
    # Strategy: 
    # Columns 0-3: 4x4x4 subproblem, use 49-rank Strassen
    # Columns 4-7: 4x4x4 subproblem, use 49-rank Strassen  
    # Column 8: 4x4x1 subproblem, use 16-rank schoolbook
    # Total rank: 49 + 49 + 16 = 114
    
    def build_4x4x4_strassen():
        """
        Build a 49-rank decomposition for 4x4 * 4x4 -> 4x4.
        Uses recursive Strassen.
        Returns U, V, W each of shape (16, 49).
        Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        # Recursive Strassen for 4x4x4
        # Split 4x4 into 2x2 blocks of 2x2 matrices
        # A = [[A00, A01], [A10, A11]] where each Ai_j is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bi_j is 2x2
        # C = [[C00, C01], [C10, C11]] where each Ci_j is 2x2
        #
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        #
        # Using Strassen at the block level (7 block multiplications of 2x2x2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M3 - M2 - M6
        #
        # Each M_r is a 2x2 * 2x2 -> 2x2 multiplication, which has rank 7 (Strassen)
        # So total rank = 7 * 7 = 49
        
        # For each of the 7 block-level Strassen terms, we need a 2x2x2 decomposition
        # The 2x2x2 Strassen decomposition has 7 terms
        
        # Let me define the 2x2x2 Strassen decomposition first
        # For 2x2 * 2x2 -> 2x2 with n=m=p=2:
        # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r] for i,j,k,r in 0..1, r in 0..6
        
        # I'll construct the 2x2x2 Strassen decomposition explicitly
        # Based on the Strassen algorithm:
        
        # The 7 terms for 2x2x2:
        # Term 0: (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        # Term 1: (c+d)e -> contributes to C[1,0]
        # Term 2: a(g-h) -> contributes to C[0,0] and C[0,1]  
        # Term 3: d(f-e) -> contributes to C[0,0] and C[1,1]
        # Term 4: (a+b)h -> contributes to C[0,1]
        # Term 5: (c-a)(e+f) -> contributes to C[1,1]
        # Term 6: (b-d)(g+h) -> contributes to C[0,0] and C[1,1]
        
        # This is getting complex. Let me use a different approach.
        
        # I'll directly construct the 49-rank decomposition by combining
        # the block-level Strassen with the base-level Strassen.
        
        # For the base 2x2x2 Strassen, here's one valid 7-term decomposition:
        # (This is the standard Strassen algorithm expressed as a tensor decomposition)
        
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        
        # 7 products:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(g-h)
        # P4 = d(f-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # C[0,0] = P1 + P4 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P2 + P3 + P6
        
        # For the tensor decomposition, each product P_r gives a rank-1 term
        # where the "left factor" is the linear combo of A entries,
        # the "right factor" is the linear combo of B entries,
        # and the "output" is how it contributes to C.
        
        # But in the tensor decomposition framework, we have:
        # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        # This is different from the Strassen algorithm which computes C = AB.
        
        # The tensor T for 2x2 matrix multiplication has T[i,j,k] = 1 for all i,j,k in 0..1
        # because each pair (i,j) of A and (j,k) of B contributes to (i,k) of C.
        
        # Wait, that's not right either. The tensor for matrix multiplication is:
        # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute
        # to the (i,k) entry of C. For standard matrix multiplication, this is 1 for all
        # valid (i,j,k) where j is the shared index.
        
        # So for 2x2x2, T[i,j,k] = 1 for all i,j,k in 0..1 (8 entries total).
        
        # The rank-1 decomposition requires:
        # sum_{r=0}^{6} U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        # This is a system of 8 equations. With 7 terms, we have 7*3*4 = 84 unknowns
        # (but with the constraint that each term is rank-1 in a specific sense).
        
        # Actually, I think I'm overcomplicating this. Let me just use the schoolbook
        # for the 2x2x2 base case (rank 8) and see if the overall structure works.
        
        # For 2x2x2 schoolbook: rank 8
        # For 4x4x4 with recursive Strassen using schoolbook base: 7 * 8 = 56
        # For 4x4x9: 56 + 56 + 16 = 128. Still better than 144.
        
        # But if I use Strassen for the base 2x2x2 (rank 7): 7*7 = 49
        # For 4x4x9: 49 + 49 + 16 = 114
        
        # Let me try to construct the 2x2x2 Strassen decomposition explicitly.
        
        # I'll use a known valid 7-term decomposition for the 2x2x2 tensor.
        
        # After research, here's a valid 7-term decomposition for 2x2x2:
        # The entries are indexed as follows:
        # U has rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V has rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W has rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # One valid 7-term decomposition:
        U2 = np.array([
            [1, 0, 1, 0, 1, 0, 0],  # (0,0)
            [0, 0, 0, 0, 1, 1, 0],  # (0,1)
            [0, 1, 0, 0, 0, -1, 0], # (1,0)
            [1, 1, 0, 1, 0, 0, 0],  # (1,1)
        ])
        V2 = np.array([
            [1, 1, 0, 0, 0, 1, 0],  # (0,0)
            [0, 0, 0, 1, 0, 1, 0],  # (0,1)
            [0, 0, 1, 0, 0, 0, 1],  # (1,0)
            [1, 0, -1, -1, 1, 0, 1], # (1,1)
        ])
        W2 = np.array([
            [1, 0, 1, 1, -1, 0, 1],  # (0,0)
            [0, 0, 0, 0, 1, 0, 0],   # (0,1)
            [0, 1, 0, 1, 0, 0, 0],   # (1,0)
            [1, -1, 1, 0, 0, 1, 1],  # (1,1)
        ])
        
        # Verify: for all i,j,k in 0..1, sum_r U2[i*2+j,r]*V2[j*2+k,r]*W2[k*2+i,r] should be 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook for 2x2x2
            U2 = np.zeros((4, 8))
            V2 = np.zeros((4, 8))
            W2 = np.zeros((4, 8))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U2[i*2+j, col] = 1
                        V2[j*2+k, col] = 1
                        W2[k*2+i, col] = 1
                        col += 1
            base_rank = 8
        else:
            base_rank = 7
        
        # Now build the 4x4x4 decomposition using block Strassen
        # 7 block-level terms, each with base_rank sub-terms
        total_rank = 7 * base_rank
        U4 = np.zeros((16, total_rank))
        V4 = np.zeros((16, total_rank))
        W4 = np.zeros((16, total_rank))
        
        # Block-level Strassen coefficients
        # A blocks: A00=[0,0:2,0:2], A01=[0,0:2,2:4], A10=[2,4:4,0:2], A11=[2,4:4,2:4]
        # B blocks: B00=[0,0:2,0:2], B01=[0,0:2,2:4], B10=[2,4:4,0:2], B11=[2,4:4,2:4]
        # C blocks: C00=[0,0:2,0:2], C01=[0,0:2,2:4], C10=[2,4:4,0:2], C11=[2,4:4,2:4]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each block-level term, we need to express the "left factor" and "right factor"
        # as linear combinations of the 4x4 A and B entries.
        
        # Let me define the block indices:
        # A block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
        # So A[i,j] belongs to block (i//2, j//2)
        
        # For M1 = (A00+A11)(B00+B11):
        # Left factor: A00 + A11, which means for each (i,j) in 0..3:
        #   if (i//2, j//2) in {(0,0), (1,1)}: coefficient 1
        #   else: coefficient 0
        # Right factor: B00 + B11, similarly
        # Output: C00 - C11 + ... (need to track how M1 contributes to C)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M3 - M2 - M6
        
        # So M1 contributes: +1 to C00, 0 to C01, 0 to C10, +1 to C11
        # M2 contributes: 0 to C00, 0 to C01, +1 to C10, -1 to C11
        # M3 contributes: 0 to C00, +1 to C01, 0 to C10, +1 to C11
        # M4 contributes: +1 to C00, 0 to C01, +1 to C10, 0 to C11
        # M5 contributes: -1 to C00, +1 to C01, 0 to C10, 0 to C11
        # M6 contributes: 0 to C00, 0 to C01, 0 to C10, -1 to C11
        # M7 contributes: +1 to C00, 0 to C01, 0 to C10, +1 to C11
        
        # For the tensor decomposition, each (block_term, sub_term) pair gives a rank-1 term
        # The U, V, W entries are products of the block-level coefficients and sub-level coefficients
        
        # Block-level left factor coefficients for each M_r:
        # M1: A00+A11 -> blocks (0,0) and (1,1) have coeff 1
        # M2: A10+A11 -> blocks (1,0) and (1,1) have coeff 1
        # M3: A00 -> block (0,0) has coeff 1
        # M4: A11 -> block (1,1) has coeff 1
        # M5: A00+A01 -> blocks (0,0) and (0,1) have coeff 1
        # M6: A10-A00 -> block (1,0) has coeff 1, block (0,0) has coeff -1
        # M7: A01-A11 -> block (0,1) has coeff 1, block (1,1) has coeff -1
        
        # Block-level right factor coefficients for each M_r:
        # M1: B00+B11 -> blocks (0,0) and (1,1) have coeff 1
        # M2: B00 -> block (0,0) has coeff 1
        # M3: B10-B11 -> block (1,0) has coeff 1, block (1,1) has coeff -1
        # M4: B01-B00 -> block (0,1) has coeff 1, block (0,0) has coeff -1
        # M5: B11 -> block (1,1) has coeff 1
        # M6: B00+B01 -> blocks (0,0) and (0,1) have coeff 1
        # M7: B10+B11 -> blocks (1,0) and (1,1) have coeff 1
        
        # Block-level output coefficients for each M_r (which C block it contributes to, with what sign):
        # M1: C00:+1, C11:+1
        # M2: C10:+1, C11:-1
        # M3: C01:+1, C11:+1
        # M4: C00:+1, C10:+1
        # M5: C00:-1, C01:+1
        # M6: C11:-1
        # M7: C00:+1, C11:+1
        
        # Now, for each (block_term r, sub_term s), the overall rank-1 term has:
        # U[i*4+j, r*base_rank+s] = (block_left[r, block(i,j)]) * U2[(i%2)*2+(j%2), s]
        # V[j*4+k, r*base_rank+s] = (block_right[r, block(j,k)]) * V2[(j%2)*2+(k%2), s]
        # W[k*4+i, r*base_rank+s] = (block_output[r, block(i,k)]) * W2[(k%2)*2+(i%2), s]
        
        # Where block(i,j) = (i//2, j//2) for A, etc.
        
        # Let me define the block coefficient arrays
        # A blocks: (0,0), (0,1), (1,0), (1,1)
        # For a given (i,j) in 0..3, the block is (i//2, j//2)
        
        # block_left[r][bi][bj] for r in 0..6, bi,bj in 0..1
        block_left = np.zeros((7, 2, 2))
        block_left[0, 0, 0] = 1; block_left[0, 1, 1] = 1  # M1: A00+A11
        block_left[1, 1, 0] = 1; block_left[1, 1, 1] = 1  # M2: A10+A11
        block_left[2, 0, 0] = 1                              # M3: A00
        block_left[3, 1, 1] = 1                              # M4: A11
        block_left[4, 0, 0] = 1; block_left[4, 0, 1] = 1    # M5: A00+A01
        block_left[5, 1, 0] = 1; block_left[5, 0, 0] = -1   # M6: A10-A00
        block_left[6, 0, 1] = 1; block_left[6, 1, 1] = -1   # M7: A01-A11
        
        # block_right[r][bi][bj] for B blocks
        block_right = np.zeros((7, 2, 2))
        block_right[0, 0, 0] = 1; block_right[0, 1, 1] = 1  # M1: B00+B11
        block_right[1, 0, 0] = 1                              # M2: B00
        block_right[2, 1, 0] = 1; block_right[2, 1, 1] = -1  # M3: B10-B11
        block_right[3, 0, 1] = 1; block_right[3, 0, 0] = -1  # M4: B01-B00
        block_right[4, 1, 1] = 1                              # M5: B11
        block_right[5, 0, 0] = 1; block_right[5, 0, 1] = 1   # M6: B00+B01
        block_right[6, 1, 0] = 1; block_right[6, 1, 1] = 1   # M7: B10+B11
        
        # block_output[r][ci][cj] for C blocks
        block_output = np.zeros((7, 2, 2))
        block_output[0, 0, 0] = 1; block_output[0, 1, 1] = 1   # M1: C00:+1, C11:+1
        block_output[1, 1, 0] = 1; block_output[1, 1, 1] = -1  # M2: C10:+1, C11:-1
        block_output[2, 0, 1] = 1; block_output[2, 1, 1] = 1   # M3: C01:+1, C11:+1
        block_output[3, 0, 0] = 1; block_output[3, 1, 0] = 1   # M4: C00:+1, C10:+1
        block_output[4, 0, 0] = -1; block_output[4, 0, 1] = 1  # M5: C00:-1, C01:+1
        block_output[5, 1, 1] = -1                              # M6: C11:-1
        block_output[6, 0, 0] = 1; block_output[6, 1, 1] = 1   # M7: C00:+1, C11:+1
        
        # Now fill in U4, V4, W4
        for r in range(7):
            for s in range(base_rank):
                col = r * base_rank + s
                for i in range(4):
                    for j in range(4):
                        bi, bj = i // 2, j // 2
                        u_coeff = block_left[r, bi, bj]
                        if u_coeff != 0:
                            U4[i*4+j, col] = u_coeff * U2[(i%2)*2+(j%2), s]
                for j in range(4):
                    for k in range(4):
                        bi, bj = j // 2, k // 2
                        v_coeff = block_right[r, bi, bj]
                        if v_coeff != 0:
                            V4[j*4+k, col] = v_coeff * V2[(j%2)*2+(k%2), s]
                for k in range(4):
                    for i in range(4):
                        bi, bj = k // 2, i // 2
                        w_coeff = block_output[r, bi, bj]
                        if w_coeff != 0:
                            W4[k*4+i, col] = w_coeff * W2[(k%2)*2+(i%2), s]
        
        return U4, V4, W4
    
    # Build the 4x4x4 Strassen decomposition
    U4, V4, W4 = build_4x4x4_strassen()
    rank_4x4x4 = U4.shape[1]
    
    # Build the full 4x4x9 decomposition
    # Columns 0-3: first 4x4x4
    # Columns 4-7: second 4x4x4
    # Column 8: 4x4x1 schoolbook
    
    rank_4x4x1 = 16  # schoolbook for 4x4x1
    total_rank = 2 * rank_4x4x4 + rank_4x4x1
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # First 4x4x4 (columns 0-3)
    # U is the same
    U_full[:, 0:rank_4x4x4] = U4
    # V: for columns 0-3, V[j*9+k, col] = V4[j*4+k, col] for k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    # W: for columns 0-3, W[k*4+i, col] = W4[k*4+i, col] for k in 0..3
    # But W_full has shape (36, total_rank) with index k*4+i where k in 0..8, i in 0..3
    # For the first 4x4x4, k is in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]
    
    # Second 4x4x4 (columns 4-7)
    offset2 = rank_4x4x4
    U_full[:, offset2:offset2+rank_4x4x4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+(k+4), offset2:offset2+rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    for k in range(4):
        for i in range(4):
            W_full[(k+4)*4+i, offset2:offset2+rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]
    
    # 4x4x1 schoolbook (column 8)
    offset3 = 2 * rank_4x4x4
    col = offset3
    for i in range(4):
        for j in range(4):
            U_full[i*4+j, col] = 1
            V_full[j*9+8, col] = 1
            W_full[8*4+i, col] = 1
            col += 1
    
    # Validate
    if validate_fast(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        best_rank = total_rank
        best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Try to improve further if time permits
    if time.time() < deadline - 30:
        # Try random search for further reduction
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            target_rank = best_rank - 1
            if target_rank < 1:
                break
            
            # Try to find a valid decomposition with target_rank
            # This is a heuristic search
            U_rand = np.random.randn(16, target_rank)
            V_rand = np.random.randn(36, target_rank)
            W_rand = np.random.randn(36, target_rank)
            
            if validate_fast(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
    
    return best
# EVOLVE-BLOCK-END
