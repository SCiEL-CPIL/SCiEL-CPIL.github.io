"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook: rank = n*m*p = 75
    # Each term is (i,j,k) with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Strassen on 2x2x2: replace 8 terms with 7
    
    # For a 2x2x2 block with rows {i1,i2}, middle {j1,j2}, cols {k1,k2}:
    # The 8 schoolbook terms are all (i,j,k) in {i1,i2}x{j1,j2}x{k1,k2}
    # Strassen uses 7 products:
    # M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
    # M2 = (A[i1,j1]+A[i2,j1])*(B[j2,k1]+B[j2,k2])  
    # M3 = A[i1,j2]*(B[j2,k1]-B[j2,k2])
    # M4 = A[i2,j1]*(B[j1,k1]-B[j1,k2])
    # M5 = (A[i2,j1]+A[i2,j2])*(B[j1,k1]+B[j1,k2])
    # M6 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2])
    # M7 = (A[i1,j1]-A[i2,j1])*(B[j1,k1]+B[j1,k2])
    # C[i1,k1] = M1+M4-M5+M7
    # C[i1,k2] = -M3+M5+M6
    # C[i2,k1] = M1-M2+M3+M7
    # C[i2,k2] = M2-M1-M6+M7
    
    # Let's verify Strassen coefficients for 2x2x2:
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B12
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A12-A22)(B11+B12)
    # M7 = (A11-A21)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor decomposition terms, each Strassen product corresponds to a rank-1 term.
    # The U, V, W vectors for each of the 7 products:
    
    # For rows i1,i2 and cols j1,j2 (middle), k1,k2 (output):
    # Product 1: U has 1 at (i1,j1) and (i2,j2); V has 1 at (j1,k1),(j1,k2),(j2,k1),(j2,k2); W has 1 at (k1,i1),(k2,i1),(k1,i2),(k2,i2)
    # Wait, this is getting complicated. Let me think in terms of the tensor.
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition term with vectors u, v, w contributes u[i]*v[j]*w[k] to T[i,j,k].
    # For schoolbook, each (i,j,k) is its own term.
    
    # For Strassen, we need 7 rank-1 terms whose sum equals the 2x2x2 sub-tensor.
    # The 2x2x2 sub-tensor has T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1}.
    
    # Strassen rank-1 terms (u, v, w) where u is 2-dim, v is 2-dim, w is 2-dim:
    # Term 1: u=(1,1), v=(1,1), w=(1,1) -> contributes 1 to all 8 positions
    # But we need the sum to be exactly the 2x2x2 all-ones tensor.
    
    # Actually, let me look up the exact Strassen decomposition as rank-1 tensors.
    # The 2x2x2 all-ones tensor has rank 7.
    
    # One known decomposition:
    # t1: u=(1,0), v=(1,0), w=(1,1)  -> T[0,0,0]=1, T[0,0,1]=1
    # t2: u=(0,1), v=(0,1), w=(1,1)  -> T[1,1,0]=1, T[1,1,1]=1
    # ... this is getting too complex to do by hand.
    
    # Let me just use the known fact that rank(M_2,2,2)=7 and implement it.
    # I'll use a known explicit decomposition.
    
    # For the full 3x5x5 problem, let me try a different approach:
    # Use the fact that we can partition and apply Strassen where beneficial.
    
    # Simple approach: just return schoolbook rank 75 for now, 
    # but let's try to get 71 by applying Strassen to two 2x2x2 blocks.
    
    # Pick rows {0,1}, middle {0,1}, cols {0,1} for first Strassen block.
    # Pick rows {0,1}, middle {2,3}, cols {2,3} for second Strassen block.
    # These are disjoint, so we save 2 total: rank = 75 - 2 = 73.
    
    # Can we do better? Let's try more blocks.
    # Rows {0,1}, middle {0,1}, cols {0,1} - block 1
    # Rows {0,1}, middle {0,1}, cols {2,3} - block 2 (disjoint in cols)
    # Rows {0,1}, middle {2,3}, cols {0,1} - block 3 (disjoint in middle)
    # Rows {0,1}, middle {2,3}, cols {2,3} - block 4 (disjoint in middle and cols)
    # That's 4 blocks, saving 4: rank = 75 - 4 = 71.
    
    # What about using row 2? 
    # Rows {1,2}, middle {0,1}, cols {0,1} - but this overlaps with block 1 on row 1.
    # Overlapping blocks can't simply be combined.
    
    # So with rows {0,1}, we can do 4 disjoint 2x2x2 blocks: rank 71.
    # With rows {0,1,2}, can we do more? 
    # If we use rows {0,1} for 4 blocks (rank 28 for those 2x5x5 parts... wait)
    # 2x5x5 schoolbook = 50, with 4 Strassen blocks = 50-4 = 46.
    # 1x5x5 schoolbook = 25.
    # Total = 46+25 = 71.
    
    # Alternatively, interleave: use rows {0,1} for some, {1,2} for others.
    # But overlapping makes it complex.
    
    # Let me just implement rank 71 with the simple non-overlapping approach.
    
    def make_strassen_block(i1, i2, j1, j2, k1, k2):
        """Return 7 rank-1 terms (as tuples of (U_vec, V_vec, W_vec) dicts) 
        for the 2x2x2 block with given indices."""
        # Using standard Strassen decomposition of 2x2x2 all-ones tensor
        # Each term is a triple of coefficient vectors
        # u is over (i,j) space (size n*m), v over (j,k) space (size m*p), w over (k,i) space (size p*n)
        
        terms = []
        # I'll use a known 7-term decomposition.
        # Let me define local coordinates: i in {0,1}, j in {0,1}, k in {0,1}
        # Map to global: i -> [i1,i2][i_local], j -> [j1,j2][j_local], k -> [k1,k2][k_local]
        
        # Strassen products (in local coords):
        # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
        #     = (A[0,0]+A[1,1]) * (sum of all B)
        # P2 = (A[0,0]+A[1,0]) * (B[1,0]+B[1,1])
        # P3 = A[0,1] * (B[1,0]-B[1,1])
        # P4 = A[1,0] * (B[0,0]-B[0,1])
        # P5 = (A[1,0]+A[1,1]) * (B[0,0]+B[0,1])
        # P6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # P7 = (A[0,0]-A[1,0]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
        
        # Wait, I need to be more careful. Let me use the standard form.
        # C = A*B where A is 2x2, B is 2x2.
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        
        # Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(f+h)
        # p3 = a(g-h)
        # p4 = d(e-f)
        # p5 = (a+b)h
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        
        # In tensor terms, each p_i is a product of a linear form in A and a linear form in B.
        # The contribution to C is then determined by the W coefficients.
        
        # For each product p_i = (u . A) * (v . B), the rank-1 term in the tensor is:
        # U[i,j] = u[i,j], V[j,k] = v[j,k], W[k,i] = w[k,i] where w encodes how this product contributes to C.
        
        # Let me define for local coords (i in {0,1}, j in {0,1}, k in {0,1}):
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        
        # p1 = (a+d)(e+h): u=(1,0,0,1) for (a,b,c,d), v=(1,0,0,1) for (e,f,g,h)
        #   Contributes to C[0,0] with +1, C[1,1] with +1
        # p2 = (c+d)(f+h): u=(0,0,1,1), v=(0,1,0,1)
        #   Contributes to C[1,0] with +1, C[1,1] with -1
        # p3 = a(g-h): u=(1,0,0,0), v=(0,0,1,-1)
        #   Contributes to C[0,1] with +1, C[1,1] with +1
        # p4 = d(e-f): u=(0,0,0,1), v=(1,-1,0,0)
        #   Contributes to C[0,0] with +1, C[1,0] with +1
        # p5 = (a+b)h: u=(1,1,0,0), v=(0,0,0,1)
        #   Contributes to C[0,0] with -1, C[0,1] with +1
        # p6 = (c-a)(e+f): u=(-1,0,1,0), v=(1,1,0,0)
        #   Contributes to C[1,1] with +1
        # p7 = (b-d)(g+h): u=(0,1,-1,0), v=(0,0,1,1)
        #   Contributes to C[0,0] with +1, C[1,1] with +1
        
        # Let me verify: 
        # C[0,0] = p1+p4-p5+p7 = (a+d)(e+h) + d(e-f) - (a+b)h + (b-d)(g+h)
        #   = ae+ah+de+dh + de-df - ah-bh + bg+bh-dg-dh
        #   = ae + (ah-ah) + (de+de) + (dh-dh) - df - bh + bg + bh - dg
        #   = ae + 2de - df + bg - dg
        #   That's not right. I made an error.
        
        # Let me redo this more carefully using a verified Strassen decomposition.
        # I'll use the one from Wikipedia:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(f+h)  
        # p3 = (a+b)(e)  -- wait, different sources use different forms.
        
        # Let me just use a computationally verified approach.
        # I'll construct the 7 vectors numerically and verify.
        
        # Actually, for the purpose of this problem, let me just hardcode a known good decomposition.
        # The rank-7 decomposition of the 2x2x2 all-ones tensor:
        
        # Term 1: u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]  (a*e -> C[0,0])
        # This is just schoolbook. I need the actual Strassen form.
        
        # Let me use a different strategy: I'll implement the decomposition by solving for it.
        # But that's complex. Instead, let me use the fact that I can verify by construction.
        
        # For now, let me just return schoolbook rank 75 to ensure correctness,
        # and the evolution can improve it.
        
        return []
    
    # Given the complexity, let me just return the schoolbook decomposition.
    # Rank = 75
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
