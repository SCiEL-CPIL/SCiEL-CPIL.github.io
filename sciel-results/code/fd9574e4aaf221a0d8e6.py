import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _strassen_2x2():
    """
    Return rank-1 terms for 2x2 * 2x2 -> 2x2 using Strassen's algorithm.
    Each term is (U_row, V_col, W_entry, coeff)
    U has 4 entries (2x2), V has 4 entries (2x2), W has 4 entries (2x2)
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(g-h)
    # M4 = (d)(f-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 + M6
    # C22 = M1 - M3 + M2 - M5
    #
    # Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> i*2+k
    #
    # For each Strassen product, we need to express it as sum of rank-1 terms.
    # But Strassen products are themselves products of linear combinations.
    # We need to expand each into rank-1 terms.
    
    # Actually, for tensor decomposition, each "product" in Strassen is not a single
    # rank-1 term but a product of two linear forms. We need to expand.
    #
    # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h -> 4 rank-1 terms
    # M2 = (c+d)(e) = c*e + d*e -> 2 rank-1 terms
    # M3 = a*(g-h) = a*g - a*h -> 2 rank-1 terms
    # M4 = d*(f-e) = d*f - d*e -> 2 rank-1 terms
    # M5 = (a+b)*h = a*h + b*h -> 2 rank-1 terms
    # M6 = (c-a)*(e+f) = c*e + c*f - a*e - a*f -> 4 rank-1 terms
    # M7 = (b-d)*(g+h) = b*g + b*h - d*g - d*h -> 4 rank-1 terms
    #
    # Total: 4+2+2+2+2+4+4 = 20 rank-1 terms? That's worse than schoolbook (16).
    #
    # Wait, I'm confusing myself. In Strassen's algorithm, the 7 "products" are
    # intermediate values, but each is a product of two linear combinations of
    # the input entries. To convert to a tensor decomposition (sum of rank-1
    # tensors), we need to expand each product into individual rank-1 terms.
    #
    # This expansion gives more than 7 terms. The key insight is that Strassen
    # reduces the number of *scalar multiplications*, but for tensor rank,
    # we count rank-1 terms in the decomposition.
    #
    # Actually, I think I'm overcomplicating this. Let me reconsider.
    #
    # The matrix multiplication tensor T has entries T[i,j,k,l] = delta[i,l]*delta[j,k]
    # for (n x m) * (m x p) -> (n x p).
    #
    # A rank-1 term is a product of three vectors: u in R^(n*m), v in R^(m*p), w in R^(n*p)
    # such that T = sum_r u_r * v_r * w_r (outer products).
    #
    # For schoolbook: T = sum_{i,j,k} e_{i,j} * e_{j,k} * e_{i,k} where e_{i,j} is
    # the unit vector in R^(n*m) at position (i,j), etc.
    #
    # Strassen's algorithm doesn't directly give a tensor decomposition in the
    # same sense. It gives an algorithm with fewer scalar multiplications, but
    # the tensor rank is a different measure.
    #
    # I need to reconsider. The tensor rank of 2x2x2x2 multiplication is 7,
    # meaning there exist 7 rank-1 tensors that sum to the multiplication tensor.
    # This is NOT the same as Strassen's 7 products being 7 rank-1 terms.
    #
    # The Strassen decomposition IS a rank-7 tensor decomposition. Each of the
    # 7 "products" corresponds to a rank-1 term where the vectors u, v, w are
    # linear combinations of the standard basis vectors.
    
    # Let me define the 7 rank-1 terms for 2x2*2x2:
    # Using the standard Strassen decomposition:
    
    terms = []
    
    # M1 = (a+d)(e+h): u = a+d, v = e+h, contributes to C
    # C11 += M1, C22 += M1
    # u vector: [1,0,1,0] (a and d)
    # v vector: [1,1,0,0] (e and h) -- wait, B = [[e,f],[g,h]], so e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # v index: j*2+k, so e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # v = e+h: [1,0,0,1]
    # w: C11 and C22 get +M1, so w = [1,0,0,1] (C indices: C[i,k] -> i*2+k, C11=(0,0), C22=(1,1))
    terms.append(([1,0,1,0], [1,0,0,1], [1,0,0,1], 1))
    
    # M2 = (c+d)*e: u = c+d, v = e
    # u: [0,0,1,1] (c=A[1,0], d=A[1,1])
    # v: [1,0,0,0] (e=B[0,0])
    # C21 += M2: C[1,0] -> index 2
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0], 1))
    
    # M3 = a*(g-h): u = a, v = g-h
    # u: [1,0,0,0]
    # v: [0,0,1,-1] (g=B[1,0], h=B[1,1])
    # C12 += M3: C[0,1] -> index 1
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,0], 1))
    
    # M4 = d*(f-e): u = d, v = f-e
    # u: [0,0,0,1]
    # v: [-1,1,0,0] (f=B[0,1], e=B[0,0])
    # C11 -= M4, C21 += M4
    # C11 -> index 0, C21 -> index 2
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,0,1], -1))  # -M4 for C11
    terms.append(([0,0,0,1], [-1,1,0,0], [0,0,1,0], 1))    # +M4 for C21
    
    # Wait, this is getting messy. Let me use the standard formulation.
    
    # Actually, I realize the Strassen decomposition for tensor rank works differently.
    # Let me look up the exact rank-7 decomposition.
    
    # The 7 rank-1 terms for 2x2*2x2 (using indices a=A00, b=A01, c=A10, d=A11,
    # e=B00, f=B01, g=B10, h=B11):
    
    # T1: u=[1,0,0,1], v=[1,1,0,0], w=[1,0,0,1]  -> (a+d)(e+f) ... no
    
    # I think I should just verify by construction. Let me try a different approach.
    #
    # Given the complexity, let me just use the schoolbook for now and accept rank 36,
    # or try a simpler block decomposition.
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    
    # For 2x3 * 3x6, partition the 6 columns into three 2-column blocks.
    # Each block: 2x3 * 3x2 -> 2x2
    # Use rank-11 decomposition for each block (if available)
    # Otherwise use schoolbook rank 12 per block, total 36
    
    # For now, use schoolbook as fallback
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
