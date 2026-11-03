<<<<<<< SEARCH
    # === Strassen 2x2x2 core: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2] ===
    # Produces: C[0,0]=A00*B00+A01*B10, C[0,1]=A10*B00+A11*B10
    #           C[1,0]=A00*B01+A01*B11, C[1,1]=A10*B01+A11*B11

    # M1 = (A00+A11)*(B00+B11)
    # Contributes to C00 and C10
    add_term([(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(1,1)], 0)

    # M2 = (A10+A11)*B00
    # Contributes to C10
    add_term([(3,1),(4,1)], [(0,1)], [(1,1),(0,-1)], 1)

    # M3 = A00*(B01-B11)
    # Contributes to C01
    add_term([(0,1)], [(1,1),(4,-1)], [(2,1)], 2)

    # M4 = A11*(B10-B00)
    # Contributes to C10
    add_term([(4,1)], [(3,1),(0,-1)], [(1,1)], 3)

    # M5 = (A00+A01)*B11
    # Contributes to C01
    add_term([(0,1),(1,1)], [(4,1)], [(2,1)], 4)

    # M6 = (A10-A00)*(B00+B01)
    # Contributes to C10 and C11
    add_term([(3,1),(0,-1)], [(0,1),(1,1)], [(1,1),(3,1)], 5)

    # M7 = (A01-A11)*(B10+B11)
    # Contributes to C00 and C01
    add_term([(1,1),(4,-1)], [(3,1),(4,1)], [(0,1),(2,1)], 6)
=======
    # === Strassen 2x2x2 core: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2] ===
    # Using standard Strassen with C[k,i] = sum_j A[i,j]*B[j,k]
    # C[0,0]=A00*B00+A01*B10, C[1,0]=A10*B00+A11*B10
    # C[0,1]=A00*B01+A01*B11, C[1,1]=A10*B01+A11*B11

    # M1 = (A00+A11)*(B00+B11)
    # Used in: C00=+M1, C10=+M1, C01=-M1, C11=-M1 (will be combined with others)
    add_term([(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(1,1),(2,-1),(3,-1)], 0)

    # M2 = (A10+A11)*B00
    # Used in: C10=+M2, C00=-M2
    add_term([(3,1),(4,1)], [(0,1)], [(1,1),(0,-1)], 1)

    # M3 = A00*(B01-B11)
    # Used in: C01=+M3, C11=-M3
    add_term([(0,1)], [(1,1),(4,-1)], [(2,1),(3,-1)], 2)

    # M4 = A11*(B10-B00)
    # Used in: C00=+M4, C10=+M4
    add_term([(4,1)], [(3,1),(0,-1)], [(0,1),(1,1)], 3)

    # M5 = (A00+A01)*B11
    # Used in: C10=+M5, C01=+M5, C11=-M5
    add_term([(0,1),(1,1)], [(4,1)], [(1,1),(2,1),(3,-1)], 4)

    # M6 = (A10-A00)*(B00+B01)
    # Used in: C11=+M6
    add_term([(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)], 5)

    # M7 = (A01-A11)*(B10+B11)
    # Used in: C00=-M7, C01=+M7
    add_term([(1,1),(4,-1)], [(3,1),(4,1)], [(0,-1),(2,1)], 6)
>>>>>>> REPLACE

Let me verify C[0,0] = A00B00 + A01B10 + A02B20:
From Strassen terms:
T0: +M1 = +(A00+A11)(B00+B11) = A00B00+A00B11+A11B00+A11B11
T1: -M2 = -(A10+A11)B00 = -A10B00-A11B00
T3: +M4 = +A11(B10-B00) = A11B10-A11B00
T6: -M7 = -(A01-A11)(B10+B11) = -A01B10-A01B11+A11B10+A11B11

Sum of Strassen contributions to C00:
A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A11B10 - A11B00 - A01B10 - A01B11 + A11B10 + A11B11
= A00B00 + A00B11 - A10B00 + 2A11B10 - A11B00 - A01B10 - A01B11 + 2A11B11

Plus T7: A02B20

This is still wrong. I clearly have the Strassen combination formulas incorrect. Let me just abandon the Strassen approach and use a verified correct rank-17 or fall back to rank 18.

Given the complexity and my repeated errors, let me use the safest correct approach: rank-18 schoolbook.

