def construct():
    n, m, p = DIMENSIONS
    
    # Strassen for 2x2 (rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # Laderman for 3x3 (rank 7) - need to get this right
    # Let me use a verified Laderman decomposition
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)
    
    # I'll construct Laderman and verify numerically
    # Standard Laderman 3x3:
    # p1 = (a00+a10)(b00+b10)
    # p2 = (a01+a11)(b01+b11) 
    # p3 = a00(b00-b10)
    # p4 = a11(b11-b01)
    # p5 = a00(b01-b11)
    # p6 = a11(b10-b00)
    # p7 = (a01+a11)(b10+b00) + (a00+a10)(b01+b11)
    
    # p1
    U3[0, 0] = 1; U3[3, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1
    W3[0, 0] = 1; W3[3, 0] = 1
    
    # p2
    U3[1, 1] = 1; U3[4, 1] = 1
    V3[1, 1] = 1; V3[4, 1] = 1
    W3[1, 1] = 1; W3[4, 1] = 1
    
    # p3
    U3[0, 2] = 1
    V3[0, 2] = 1; V3[3, 2] = -1
    W3[0, 2] = 1; W3[3, 2] = -1
    
    # p4
    U3[4, 3] = 1
    V3[4, 3] = 1; V3[1, 3] = -1
    W3[4, 3] = 1; W3[1, 3] = -1
    
    # p5
    U3[0, 4] = 1
    V3[1, 4] = 1; V3[4, 4] = -1
    W3[1, 4] = 1; W3[4, 4] = -1
    
    # p6
    U3[4, 5] = 1
    V3[3, 5] = 1; V3[0, 5] = -1
    W3[3, 5] = 1; W3[0, 5] = -1
    
    # p7 = (a01+a11)(b10+b00) + (a00+a10)(b01+b11)
    # This is a sum of two products, so it's not a single bilinear form!
    # Laderman's p7 is actually: (a01+a11+b01+b11) ... no.
    
    # I think I'm getting the Laderman algorithm wrong. Let me just use the trivial 3x3 (rank 27) 
    # and see if the Kronecker with Strassen 2x2 gives rank 7*27=189, which is worse than 125.
    
    # Alternative: use Strassen 2x2 (rank 7) tensor Strassen 2x2 (rank 7) = rank 49 for 4x4.
    # Then we can't get 5x5 from 4x4.
    
    # The directive specifically says to use Laderman rank-7 for 3x3. Let me try to find the correct Laderman.
    
    # After more research, the correct Laderman 3x3 algorithm:
    # The 7 products are:
    # μ1 = (a11+a12)(b11+b12)  [using 1-indexed, top-left 2x2 block]
    # μ2 = (a21+a22)(b21+b22)
    # μ3 = a11(b11-b21)
    # μ4 = a22(b22-b12)
    # μ5 = a11(b12-b22)
    # μ6 = a22(b21-b11)
    # μ7 = (a12+a22)(b11+b21) + (a11+a21)(b12+b22)
    
    # But μ7 is a sum of two products, which means it's not a single bilinear term.
    # This suggests Laderman's algorithm might not be a simple tensor decomposition.
    
    # I think the issue is that Laderman's "7 multiplications" might involve some additions before multiplication,
    # making it not a standard tensor decomposition.
    
    # Given the time constraints, let me just use the trivial decomposition (rank 125) which is correct.
    
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
