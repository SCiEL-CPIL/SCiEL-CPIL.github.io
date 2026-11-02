import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # n=2 rows of A, m=2 cols of A = rows of B, p=15 cols of B
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (30, rank)
    # W index: k*n + i where k is column of B, i is row of A
    
    # We'll use Strassen decomposition for 2x2 * 2x2 blocks.
    # Correct Strassen formulas verified below.
    # 
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    #
    # Verified Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+c)h
    # M6 = (d-b)(e+f)
    # M7 = (b-c)(g+h)
    # C00 = M1 + M3 - M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    
    # Let me verify with a=1,b=2,c=3,d=4, e=5,f=6,g=7,h=8
    # C00 = 1*5 + 2*7 = 5+14 = 19
    # C01 = 1*6 + 2*8 = 6+16 = 22
    # C10 = 3*5 + 4*7 = 15+28 = 43
    # C11 = 3*6 + 4*8 = 18+32 = 50
    #
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+3)*8 = 32
    # M6 = (4-2)*(5+6) = 2*11 = 22
    # M7 = (2-3)*(7+8) = -1*15 = -15
    #
    # C00 = 65 + (-2) - 8 - 22 = 65 - 2 - 8 - 22 = 33. Should be 19. WRONG!
    
    # These formulas are definitely wrong. Let me derive correct ones.
    # 
    # Actually, I recall now that the standard Strassen uses:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C00 = M1 + M3 - M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    #
    # With A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    # A00=1, A01=2, A10=3, A11=4
    # B00=5, B01=6, B10=7, B11=8
    #
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-8) = -1
    # M4 = 4*(6-5) = 4
    # M5 = (1+3)*8 = 32
    # M6 = (4-1)(5+7) = 3*12 = 36
    # M7 = (2-4)(6+8) = -2*14 = -28
    #
    # C00 = 65 + (-1) - 4 - 36 = 65 - 1 - 4 - 36 = 24. Should be 19. STILL WRONG.
    
    # I'm clearly making systematic errors. Let me just use schoolbook for each 2x2x2 block.
    # Rank 4 per 2x2x2 block. 7 blocks + 1 column of width 1 (rank 2) = 7*4 + 2 = 30.
    # Wait, that's better than 60!
    
    # Schoolbook for 2x2 * 2xk: for each i in {0,1}, j in {0,1}, k in range:
    # C[i,k] += A[i,j] * B[j,k]
    # Each term is rank 1. Total terms: 2*2*k = 4k.
    # For k=2: rank 8. For k=1: rank 4.
    # 7*8 + 1*4 = 56 + 4 = 60. Same as before.
    
    # OK so schoolbook tiling doesn't help. I need Strassen to work.
    # Let me very carefully derive Strassen from scratch.
    #
    # We want to compute C = AB where A,B are 2x2.
    # C00 = a*e + b*g
    # C01 = a*f + b*h
    # C10 = c*e + d*g
    # C11 = c*f + d*h
    #
    # 8 scalar products in schoolbook. Strassen shows 7 suffice.
    #
    # The correct Strassen algorithm (I'll verify each step):
    # Let me use the formulation from a reliable source.
    #
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - e)
    # M5 = (a + c) * h
    # M6 = (d - b) * (e + f)
    # M7 = (b - c) * (g + h)
    #
    # C00 = M1 + M3 - M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    #
    # I've verified this fails. Let me try yet another set.
    #
    # From "Introduction to Algorithms" (CLRS), the Strassen algorithm:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A21)B22
    # M6 = (A22 - A11)(B11 + B21)
    # M7 = (A12 - A22)(B12 + B22)
    #
    # C11 = M1 + M3 - M4 - M6
    # C12 = M4 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 - M7
    #
    # Note: CLRS uses 1-indexing. So A11=a, A12=b, A21=c, A22=d.
    # B11=e, B12=f, B21=g, B22=h.
    # C11=C00, C12=C01, C21=C10, C22=C11.
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+c)h
    # M6 = (d-a)(e+g)
    # M7 = (b-d)(f+h)
    #
    # C00 = M1 + M3 - M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    #
    # Verify with a=1,b=2,c=3,d=4, e=5,f=6,g=7,h=8:
    # M1 = (1+4)(5+8) = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-8) = -1
    # M4 = 4*(6-5) = 4
    # M5 = (1+3)*8 = 32
    # M6 = (4-1)(5+7) = 3*12 = 36
    # M7 = (2-4)(6+8) = -2*14 = -28
    #
    # C00 = 65 + (-1) - 4 - 36 = 24. Should be 1*5+2*7=19. WRONG.
    
    # I give up trying to get Strassen right from memory. Let me just use schoolbook
    # with rank 60, which is guaranteed correct.
    
    rank = n * m * p  # 2*2*15 = 60
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
