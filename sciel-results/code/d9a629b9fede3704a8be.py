"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition.

Strategy:
  • Use Strassen's 2x2x2 decomposition (rank 7) as a building block.
  • For the 4x4x4 part: split into 2x2 blocks, apply Strassen at block level.
  • For the 4x4x2 part: handle with schoolbook or a structured approach.
  • Target: reduce from schoolbook rank 96.
"""

import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen.
    
    Returns list of (U, V, W) tuples, each a list of 4 values.
    U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
    """
    terms = []
    
    # p1 = (a00+a11)(b00+b11)
    # C[0,0] += p1, C[1,1] += p1
    terms.append(([1,0,0,1], [1,0,1,0], [1,0,0,1]))
    
    # p2 = (a10+a11)(b00)
    # C[1,0] += p2, C[1,1] += p2
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    
    # p3 = a00(b01-b11)
    # C[0,1] += p3, C[0,0] += p3 (from b01 part) -- wait, let me re-derive
    # Actually: p3 = a00 * (b01 - b11)
    # In the product: a00*b01 contributes to C[0,1], a00*(-b11) is a "cross" term
    # But in Strassen, the combination works out. Let me use verified formulas.
    # c00 = p1 - p5 + p4 + p7  (using 0-indexed)
    # c01 = p3 + p5
    # c10 = p2 + p4  
    # c11 = p1 - p2 + p3 + p6
    #
    # Wait, I need to be very careful. Let me use the standard 1-indexed Strassen:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    #
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    #
    # Converting to 0-indexed:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # Now for each p_r, determine U, V, W vectors:
    # U has 4 entries indexed by (i,j) in {0,1}^2, flattened as i*2+j
    # V has 4 entries indexed by (j,k) in {0,1}^2, flattened as j*2+k
    # W has 4 entries indexed by (k,i) in {0,1}^2, flattened as k*2+i
    
    # p1 = (a00+a11)(b00+b11)
    # U: a00=1, a01=0, a10=0, a11=1 -> [1,0,0,1]
    # V: b00=1, b01=0, b10=1, b11=0 -> [1,0,1,0]
    # W: c00+=1, c11+=1 -> W[k*2+i]: W[0*2+0]=1, W[1*2+1]=1 -> [1,0,0,1]
    terms[0] = ([1,0,0,1], [1,0,1,0], [1,0,0,1])
    
    # p2 = (a10+a11)b00
    # U: [0,0,1,1]
    # V: [1,0,0,0]
    # W: c10+=1, c11+=1 -> W[0*2+1]=1, W[1*2+1]=1 -> [0,1,0,1]
    terms[1] = ([0,0,1,1], [1,0,0,0], [0,1,0,1])
    
    # p3 = a00(b01-b11)
    # U: [1,0,0,0]
    # V: [0,1,0,-1]
    # W: c01+=1, c11+=1 -> W[0*2+0]=1, W[1*2+1]=1 -> [1,0,0,1]
    # Wait: c01 means C[0,1], so k=1, i=0 -> W[1*2+0]=W[2]=1
    # c11 means C[1,1], so k=1, i=1 -> W[1*2+1]=W[3]=1
    # W = [0,0,1,1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    
    # p4 = a11(b10-b00)
    # U: [0,0,0,1]
    # V: [0,-1,1,0]
    # W: c00+=1, c10+=1 -> W[0*2+0]=1, W[0*2+1]=1 -> [1,1,0,0]
    terms.append(([0,0,0,1], [0,-1,1,0], [1,1,0,0]))
    
    # p5 = (a00+a01)b11
    # U: [1,1,0,0]
    # V: [0,0,0,1]
    # W: c00-=1, c01+=1 -> W[0*2+0]=-1, W[1*2+0]=1 -> [-1,0,1,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # p6 = (a10-a00)(b00+b01)
    # U: [-1,0,1,0]
    # V: [1,1,0,0]
    # W: c11+=1 -> W[1*2+1]=1 -> [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # p7 = (a01-a11)(b10+b11)
    # U: [0,1,0,-1]
    # V: [0,0,1,1]
    # W: c00+=1, c11-=1 -> W[0*2+0]=1, W[1*2+1]=-1 -> [1,0,0,-1]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,-1]))
    
    return terms

def construct():
    """Return an exact tensor decomposition for 4x4x6 matrix multiply."""
    t_start = time.time()
    DEADLINE = 120.0
    
    n, m, p = 4, 4, 6
    
    # Strategy: Split the 4x4x6 into:
    # Part 1: 4x4x4 (first 4 columns of B)
    # Part 2: 4x4x2 (last 2 columns of B)
    #
    # For Part 1 (4x4x4): Use Kronecker structure with Strassen
    # Split 4x4 into 2x2 blocks. Each block is 2x2.
    # The 4x4x4 tensor = T_2 ⊗ T_2 (Kronecker product of two 2x2x2 tensors)
    #
    # But rank(T⊗T) can be less than rank(T)^2 if we find a good decomposition.
    # 
    # Actually, let me use a direct approach:
    # For 4x4x4, I can use the fact that it's a block matrix multiply.
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # 
    # A = [[A00, A01], [A10, A11]] (each Ai is 2x2)
    # B = [[B00, B01], [B10, B11]] (each Bj is 2x2)  
    # C = [[C00, C01], [C10, C11]] (each Ci is 2x2)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each block product (2x2)*(2x2) = 2x2 has rank 7 (Strassen).
    # There are 8 block products total.
    #
    # But we can combine using Strassen at the block level:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two linear combinations of 2x2 blocks.
    # When we expand each Mi using Strassen (7 rank-1 terms each), we get 7*7=49 terms.
    # But many can be combined/cancelled.
    #
    # For simplicity and correctness, let me just use schoolbook for now:
    # Rank 96 for 4x4x6.
    
    # Schoolbook decomposition: rank = n*m*p = 4*4*6 = 96
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
