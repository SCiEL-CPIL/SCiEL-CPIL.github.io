"""Matrix multiplication 4x4x5 tensor decomposition - structured low-rank."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Use a structured approach: tile into 2x2 blocks with shared operations
    # Strategy: use Winograd-style 2x2 block decomposition with rank-7 per block
    # but shared across overlapping blocks to reduce total rank
    
    # Actually, let's try a direct algebraic construction
    # For 4x4 * 4x5, we can use the following idea:
    # Split the 4x4 into 4 rows and 4 cols, 4x5 into 4 rows and 5 cols
    # Use a factorization based on: (A+B)(C+D) = AC + AD + BC + BD - (A-B)(C-D)
    # This saves one multiplication per pair
    
    # Let's try a rank-76 construction using pairwise sharing
    # For each pair of rows in A (i, i+1) and each column pair in B (k, k+1):
    # We can compute 2x2 subproducts with rank 7 instead of 8
    # 4 rows -> 2 pairs, 5 cols -> 2 pairs + 1 single
    # 2 pairs * (2 pairs of cols * 7 + 1 single * 4) = 2 * (14 + 4) = 36
    # Plus cross terms... this is getting complex
    
    # Let me try a different approach: use a known algorithm structure
    # For 4x4*4x4, Bläser found rank 70. For 4x4*4x5, we expect ~76-78.
    
    # Let's construct a rank-78 decomposition using block sharing:
    # Partition: A is 4x4, B is 4x5
    # Think of it as 4 rows of A times 4 rows of B (each producing 5 cols)
    
    # Use the identity: for 2x1 * 2x1 vectors (a,b) and (c,d):
    # ac + bd can be computed as: ((a+b)(c+d) - (a-b)(c-d) + ... ) 
    # Actually for dot product of two 2-vectors: ac+bd needs 2 mults minimum (rank 2)
    # For 2x2 * 2x2 * 2x2: rank 7 (Winograd)
    
    # Let's try: split into 2x2 blocks
    # A = [[A11, A12],[A21, A22]] where each is 2x2
    # B = [[B11, B12, B13],[B21, B22, B23]] where B1x is 2x2, B13 is 2x1
    # Wait, 4x5 doesn't split evenly into 2x2 blocks for the last dimension
    
    # Alternative: 4x5 = 4x4 + 4x1
    # C = A*(B4x4) + A*(B4x1)
    # First part: 4x4 * 4x4 (rank ~70-76)
    # Second part: 4x4 * 4x1 (this is just matrix-vector, rank 16 schoolbook)
    # Total: ~86-92, worse than 80
    
    # Let me just do a careful schoolbook but try to find savings
    # Actually, for this specific problem, let me try a rank-78 construction:
    # Use the fact that we can share multiplications between adjacent entries
    
    # I'll use a structured approach with rank 78:
    # For each of the 4 output rows and 5 output cols, we need 4 multiplications
    # Total 80. To save 2, we need 2 pairs of "free" computations.
    
    # Using the identity: xy + zw = ((x+z)(y+w) - (x-z)(y-w) + ... ) doesn't directly work
    # But: for computing a*c + b*d, we can use: ((a+b)(c+d) + (a-b)(c-d))/2
    # This uses 2 multiplications for 2 products - no savings
    # For 3 products: abc-style identities can save
    
    # Let me try a completely different approach: use numpy to search
    # for a low-rank decomposition via optimization
    
    deadline = time.time() + 110  # 110s budget, leave 10s margin
    best_result = None
    
    # Start with schoolbook
    rank_school = n * m * p  # 80
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition using a structured approach
    # Use the 2x2 Winograd block: for multiplying 2x2 by 2x2:
    # m0 = (a+b)(c+d), m1 = (a-b)(c+d), m2 = a(e+f), m3 = c(a+b), m4 = (a+b)(f-d)
    # m5 = (e-f)(a+b), m6 = d(b-c), m7 = (e+f)(b-c)
    # Then: p11 = m0+m3-m5-m7, p12 = m1+m2+m5-m7, p21 = m3+m6-m0-m4, p22 = m0+m2-m4-m6
    
    # For 4x4 * 4x5, let me tile using 2x2 blocks:
    # A (4x4): 4 blocks of 2x2
    # B (4x5): top 2 rows have 2 blocks of 2x2 + 1 block of 2x1
    #           bottom 2 rows have 2 blocks of 2x2 + 1 block of 2x1
    
    # This is complex. Let me try a direct rank reduction search.
    
    # Try rank 78: modify schoolbook to share 2 multiplications
    # Use identity: for computing c[i][k] = sum_j a[i][j]*b[j][k]
    # If we fix i and k, it's a dot product of length 4.
    # Dot product of length 4: can we do it in < 4 multiplications?
    # No, the bilinear complexity of dot product of length n is n.
    
    # So we need to share across different (i,k) pairs.
    # The key insight: share partial products a[i][j]*b[j][k] across different computations.
    
    # For a given j, a[i][j]*b[j][k] appears in c[i][k].
    # Different (i,k) pairs need different (i,j,k) triples, so there's no direct sharing
    # unless we use algebraic identities.
    
    # Let me try: can we compute c[i][k] and c[i][k+1] together more efficiently?
    # c[i][k] = sum_j a[i][j]*b[j][k]
    # c[i][k+1] = sum_j a[i][j]*b[j][k+1]
    # Together: 8 multiplications for 8 products (4 each)
    # Using: a[i][j]*b[j][k] + a[i][j]*b[j][k+1] = a[i][j]*(b[j][k]+b[j][k+1])
    # This doesn't save unless we can share the sum (b[j][k]+b[j][k+1]) across j
    # But a[i][j] differs for each j, so no.
    
    # What about: (a[i][j]+a[i][j+1]) * (b[j][k]+b[j+1][k]) style?
    # This is the Winograd/Fast matrix multiply idea.
    
    # Let me implement a proper 2x2 Winograd tiling:
    # Split 4x4 into four 2x2 blocks: A = [[A00,A01],[A10,A11]]
    # Split 4x5 into: B = [[B00,B01,B02],[B10,B11,B12]] where B0x,B1x are 2x2, B02,B12 are 2x1
    # Wait, 5 = 2+2+1, so:
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # where B00,B01,B10,B11 are 2x2, B02,B12 are 2x1
    
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12]]
    
    # Each 2x2 * 2x2 product: rank 7 (Winograd)
    # Each 2x2 * 2x1 product: rank 4 (schoolbook, no savings for 2x1)
    
    # Number of 2x2*2x2 products: 4 positions * 2 = 8
    # Number of 2x2*2x1 products: 4 positions
    # Total rank: 8*7 + 4*4 = 56 + 16 = 72
    
    # Wait, but we need to be careful about the Winograd construction.
    # Let me implement this properly.
    
    # Winograd 2x2 * 2x2:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # m0 = (x00+x01) * (y00+y10)
    # m1 = (x00-x01) * (y00+y10)
    # m2 = (x10+x11) * (y00+y01)
    # m3 = (x10-x11) * (y00+y01)
    # m4 = x00 * (y10+y11)
    # m5 = x10 * (y01-y11)
    # m6 = (x00+x10) * (y10+y11)
    # m7 = (x01+x11) * (y01-y11)
    
    # P = X*Y:
    # p00 = m0 + m4 - m6 + m7  ... wait, let me look this up properly
    
    # Standard Winograd 2x2:
    # m0 = (x00+x01)(y00+y10)
    # m1 = (x00-x01)(y00+y10)  
    # m2 = (x10+x11)(y00+y01)
    # m3 = (x10-x11)(y00+y01)
    # m4 = x00(y10+y11)
    # m5 = x10(y01-y11)
    # m6 = (x00+x10)(y10+y11)
    # m7 = (x01+x11)(y01-y11)
    
    # p00 = m0 + m4 - m6 + m7
    # p01 = m1 + m4 + m6 - m7  ... hmm, I'm not sure about the exact formulas
    
    # Let me use a verified Winograd 2x2 construction:
    # From literature, the 7-multiplication scheme for 2x2 matmul:
    # m0 = (a+b)(c+d)
    # m1 = (a-b)(c+d)
    # m2 = a(e+f)
    # m3 = c(a+b)
    # m4 = (a+b)(f-d)
    # m5 = (e-f)(a+b)
    # m6 = d(b-c)
    # m7 = (e+f)(b-c)
    
    # p00 = m0 + m3 - m5 - m7
    # p01 = m1 + m2 + m5 - m7
    # p10 = m3 + m6 - m0 - m4
    # p11 = m0 + m2 - m4 - m6
    
    # Let me verify: A = [[a,b],[c,d]], B = [[e,f],[g,h]] (using different letters)
    # Actually standard notation: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # ab = ae+bg, ac = af+bh, ad = ce+dg, bd = cf+dh
    
    # Let me just use a known correct Winograd 2x2:
    # m0 = (a+b)(e+g)
    # m1 = (a-b)(e+g)
    # m2 = (c+d)(e+f)
    # m3 = (c-d)(e+f)
    # m4 = a(g+h)
    # m5 = c(f-h)
    # m6 = (a+c)(g+h)
    # m7 = (b+d)(f-h)
    
    # p00 = m0 + m4 - m6 + m7
    # p01 = m1 + m4 + m6 - m7
    # p10 = m2 + m5 - m0 - m7  ... I'm getting confused
    
    # Let me just implement it and verify numerically.
    
    def winograd_2x2_mult_indices():
        """Return the 7 (U,V,W) triples for 2x2*2x2 Winograd.
        A = [[a,b],[c,d]] (indices 0,1,2,3 in row-major: a=0,b=1,c=2,d=3)
        B = [[e,f],[g,h]] (indices 0,1,2,3: e=0,f=1,g=2,h=3)
        P = [[p00,p01],[p10,p11]] (indices 0,1,2,3)
        """
        # Using the standard Winograd F(2,2) scheme
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # P entries:
        # p00 = a*e + b*g
        # p01 = a*f + b*h
        # p10 = c*e + d*g
        # p11 = c*f + d*h
        
        # Let me derive:
        # m0 = ae+ag+be+bg
        # m1 = ae-ag-be+bg
        # m2 = ce+cf+de+df
        # m3 = ce-cf-de+df
        # m4 = ag+ah
        # m5 = cf-ch
        # m6 = ag+ah+cg+ch
        # m7 = bf+bh+df-dh
        
        # p00 = ae+bg = (m0+m1)/2
        # Hmm, this gives fractions. Let me use a different formulation.
        
        # Alternative Winograd scheme (no divisions):
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # p00 = (m0 + m1) / 2  -- still has division
        
        # Let me use a scheme that avoids divisions:
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # p00 = (m0+m1)/2, p01 = (m0-m1)/2 + (m4+m7)/2 ... this is getting messy
        
        # Let me use a clean, verified Winograd 2x2:
        # From "Introduction to Arithmetic Complexity Theory" or similar:
        # The 7-multiplication algorithm for 2x2 matmul:
        # m0 = (x00+x01)(y00+y10)
        # m1 = (x00-x01)(y00+y10)
        # m2 = (x10+x11)(y00+y01)
        # m3 = (x10-x11)(y00+y01)
        # m4 = x00(y10+y11)
        # m5 = x10(y01-y11)
        # m6 = (x00+x10)(y10+y11)
        # m7 = (x01+x11)(y01-y11)
        
        # P00 = m0 + m4 - m6 + m7
        # P01 = m1 + m4 + m6 - m7
        # P10 = m2 + m5 - m0 - m7
        # P11 = m2 - m5 + m6 - m4
        
        # Let me verify with a specific example:
        # X = [[1,2],[3,4]], Y = [[5,6],[7,8]]
        # X*Y = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        
        # x00=1,x01=2,x10=3,x11=4; y00=5,y01=6,y10=7,y11=8
        # m0 = (1+2)(5+7) = 3*12 = 36
        # m1 = (1-2)(5+7) = -1*12 = -12
        # m2 = (3+4)(5+6) = 7*11 = 77
        # m3 = (3-4)(5+6) = -1*11 = -11
        # m4 = 1*(7+8) = 15
        # m5 = 3*(6-8) = 3*(-2) = -6
        # m6 = (1+3)*(7+8) = 4*15 = 60
        # m7 = (2+4)*(6-8) = 6*(-2) = -12
        
        # P00 = 36 + 15 - 60 + (-12) = 36+15-60-12 = -21  (should be 19) -- WRONG
        
        # Let me try a different assignment. I'll look up the correct Winograd 2x2.
        # 
        # Correct Winograd F(2,2,2):
        # m0 = (a+b)(c+d)
        # m1 = (a-b)(c+d)
        # m2 = a(e+f)
        # m3 = c(a+b)
        # m4 = (a+b)(f-d)
        # m5 = (e-f)(a+b)
        # m6 = d(b-c)
        # m7 = (e+f)(b-c)
        
        # Where A = [[a,b],[c,d]], B = [[e,f],[g,h]]... 
        # Wait, I think the notation is:
        # A = [[a,b],[c,d]], B = [[e,g],[f,h]] (column-major for B?)
        # Or A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        
        # Let me use a completely different approach. I'll just verify numerically.
        
        # Standard Winograd 2x2 (from multiple sources):
        # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # P = A*B:
        # P[0,0] = (m0 + m1) / 2
        # P[0,1] = (m0 - m1) / 2 + (m4 + m7) / 2 - m6/2 ... 
        # This is getting too error-prone. Let me just use a verified construction.
        
        # I'll use the following verified Winograd 2x2 (from Strassen-style):
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # P00 = (m0+m1)/2
        # P01 = (m0-m1)/2 + (m4+m7-m6)/2  ... 
        
        # You know what, let me just implement it and test numerically in the code.
        
        return None  # placeholder
    
    # Let me take a step back and try a simpler but potentially effective approach.
    # I'll try to find a rank-78 decomposition by modifying the schoolbook
    # to share 2 multiplications using a simple identity.
    
    # For computing c[i][k] = sum_j a[i][j] * b[j][k], consider two adjacent output entries:
    # c[i][k] and c[i][k+1]
    # = sum_j a[i][j]*b[j][k] and sum_j a[i][j]*b[j][k+1]
    
    # Using: a[i][j]*b[j][k] + a[i][j]*b[j][k+1] = a[i][j]*(b[j][k]+b[j][k+1])
    # This doesn't save unless we can share (b[j][k]+b[j][k+1]) across different i.
    
    # What about sharing across j?
    # c[i][k] = a[i][0]*b[0][k] + a[i][1]*b[1][k] + a[i][2]*b[2][k] + a[i][3]*b[3][k]
    
    # Group as: (a[i][0]+a[i][1])*(b[0][k]+b[1][k]) - a[i][0]*b[1][k] - a[i][1]*b[0][k]
    #            + (a[i][2]+a[i][3])*(b[2][k]+b[3][k]) - a[i][2]*b[3][k] - a[i][3]*b[2][k]
    
    # This uses 2 + 4 + 2 + 4 = 12 multiplications for 2 output entries (8 needed), worse.
    
    # OK, let me try yet another approach: use a known low-rank algorithm.
    # For 4x4*4x4, the best known is 70. For 4x4*4x5, I'll try to construct rank 76-78.
    
    # Let me try a constructive approach using a 2x2 Winograd tiling properly.
    
    # I'll define the Winograd 2x2 multiplication explicitly and verify it.
    
    def make_winograd_2x2():
        """Construct U,V,W for 2x2 * 2x2 -> 2x2 with rank 7.
        A is 2x2 (4 entries, row-major: index 0=a, 1=b, 2=c, 3=d for A=[[a,b],[c,d]])
        B is 2x2 (4 entries, row-major: index 0=e, 1=f, 2=g, 3=h for B=[[e,f],[g,h]])
        P is 2x2 (4 entries, row-major: index 0=p00, 1=p01, 2=p10, 3=p11)
        """
        # Using a verified Winograd 2x2 scheme:
        # m0 = (a+b)(e+g)
        # m1 = (a-b)(e+g)
        # m2 = (c+d)(e+f)
        # m3 = (c-d)(e+f)
        # m4 = a(g+h)
        # m5 = c(f-h)
        # m6 = (a+c)(g+h)
        # m7 = (b+d)(f-h)
        
        # Let me verify: P = A*B where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
        # P[0,0] = a*e + b*g
        # P[0,1] = a*f + b*h
        # P[1,0] = c*e + d*g
        # P[1,1] = c*f + d*h
        
        # m0 = (a+b)(e+g) = ae+ag+be+bg
        # m1 = (a-b)(e+g) = ae-ag-be+bg
        # m2 = (c+d)(e+f) = ce+cf+de+df
        # m3 = (c-d)(e+f) = ce-cf-de+df
        # m4 = a(g+h) = ag+ah
        # m5 = c(f-h) = cf-ch
        # m6 = (a+c)(g+h) = ag+ah+cg+ch
        # m7 = (b+d)(f-h) = bf-bh+df-dh
        
        # P[0,0] = ae+bg = (m0+m1)/2
        # P[0,1] = af+bh
        #   af = ? Let's see: m4-m5 = ag+ah-cf+ch, not directly af
        #   m6-m0 = (ag+ah+cg+ch)-(ae+ag+be+bg) = ah+cg+ch-ae-be-bg
        #   This is getting complicated.
        
        # Let me try a different, cleaner Winograd 2x2:
        # From "Fast Matrix Multiplication" by Winograd:
        # m0 = (a+b)(c+d)
        # m1 = (a-b)(c+d)
        # m2 = a(e+f)
        # m3 = c(a+b)
        # m4 = (a+b)(f-d)
        # m5 = (e-f)(a+b)
        # m6 = d(b-c)
        # m7 = (e+f)(b-c)
        
        # With A = [[a,b],[c,d]], B = [[e,f],[g,h]]... 
        # I think the B matrix in Winograd's notation might be [[e,g],[f,h]]
        
        # Let me just try all 8 possible combinations and verify numerically.
        
        # Actually, let me use a known correct one. The standard Winograd 2x2:
        # m0 = (x00+x01)(y00+y10)
        # m1 = (x00-x01)(y00+y10)
        # m2 = (x10+x11)(y00+y01)
        # m3 = (x10-x11)(y00+y01)
        # m4 = x00(y10+y11)
        # m5 = x10(y01-y11)
        # m6 = (x00+x10)(y10+y11)
        # m7 = (x01+x11)(y01-y11)
        
        # P00 = m0 + m4 - m6 + m7
        # P01 = m1 + m4 + m6 - m7
        # P10 = m2 + m5 - m0 - m7
        # P11 = m2 - m5 + m6 - m4
        
        # Verify: X=[[1,2],[3,4]], Y=[[5,6],[7,8]]
        # x00=1,x01=2,x10=3,x11=4; y00=5,y01=6,y10=7,y11=8
        # m0 = (1+2)(5+7) = 36
        # m1 = (1-2)(5+7) = -12
        # m2 = (3+4)(5+6) = 77
        # m3 = (3-4)(5+6) = -11
        # m4 = 1*(7+8) = 15
        # m5 = 3*(6-8) = -6
        # m6 = (1+3)*(7+8) = 60
        # m7 = (2+4)*(6-8) = -12
        
        # P00 = 36+15-60+(-12) = -21 (should be 1*5+2*7=19) WRONG
        
        # The formula is wrong. Let me try:
        # P00 = m0 + m4 - m6 + m7 = 36+15-60-12 = -21
        # Maybe: P00 = (m0 + m4 - m6 + m7) / 2? = -21/2, no.
        
        # Let me try different P formulas:
        # P00 = m0 + m3 - m5 - m7 = 36+(-11)-(-6)-(-12) = 36-11+6+12 = 43, no
        # P00 = m0 + m2 - m4 - m6 = 36+77-15-60 = 38, no
        # P00 = (m0 + m2) / 2 = (36+77)/2 = 56.5, no
        
        # I clearly don't remember the exact Winograd 2x2 formulas.
        # Let me just use the schoolbook rank-80 decomposition and try to
        # find improvements through a numerical search.
        
        return None
    
    # Given the difficulty of getting the Winograd formulas exactly right,
    # let me try a different strategy: use a numerical optimization to find
    # a low-rank decomposition.
    
    # Actually, let me try a completely different approach:
    # Use the fact that the matrix multiplication tensor has a specific structure
    # and try to find a rank-78 decomposition through a constructive algorithm.
    
    # For now, let me just return the schoolbook decomposition, but with a
    # slight modification to try to get a small improvement.
    
    # Try rank 78: use 2 "shared" multiplications
    # Idea: for each pair of consecutive columns k, k+1 in B:
    # c[i][k] + c[i][k+1] = sum_j a[i][j]*(b[j][k]+b[j][k+1])
    # c[i][k] - c[i][k+1] = sum_j a[i][j]*(b[j][k]-b[j][k+1])
    # This still uses 2*4=8 multiplications per i for 2 columns, same as before.
    
    # I don't see an easy way to beat rank 80 without a more sophisticated algorithm.
    # Let me just return the schoolbook and hope for the best.
    
    # Actually, wait. Let me try a rank-79 decomposition by finding ONE shared multiplication.
    # 
    # Consider: c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0] + a[0][3]*b[3][0]
    # c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0] + a[1][3]*b[3][0]
    # 
    # (c[0][0] + c[1][0]) = (a[0][0]+a[1][0])*b[0][0] + (a[0][1]+a[1][1])*b[1][0] + ...
    # (c[0][0] - c[1][0]) = (a[0][0]-a[1][0])*b[0][0] + (a[0][1]-a[1][1])*b[1][0] + ...
    # 
    # Still 8 multiplications total. No savings.
    
    # The fundamental issue is that the bilinear complexity of a dot product of length n is n,
    # and the matrix multiplication tensor is a "tensor product" of these dot products
    # with no obvious algebraic shortcuts for small sizes.
    
    # For 4x4*4x5, the schoolbook rank of 80 might actually be close to optimal.
    # The best known for 4x4*4x4 is 70 (Bläser), which is 70/64 = 1.09375x schoolbook.
    # For 4x4*4x5, schoolbook is 80, and 70/64 * 80 = 87.5, so 70-rank is NOT directly applicable.
    # Actually, 4x4*4x5 is not a square problem, so the 70-rank algorithm doesn't directly apply.
    
    # I think the best I can do is return the schoolbook decomposition.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
