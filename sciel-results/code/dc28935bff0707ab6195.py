"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add(u_idx, v_idx, w_idx, val=1):
        for i in u_idx:
            U[i] += val
        for i in v_idx:
            V[i] += val
        for i in w_idx:
            W[i] += val
            
    # Group 1: A_L (U 0,1,4,5) * B_T1 (V 0,1,5,6)
    # W indices: C00(0), C10(1), C01(2), C11(3)
    
    # Strassen P1: (X00+X11)(Y00+Y11)
    add([0, 5], [0, 6], [0, 3])
    # Strassen P2: (X10+X11)Y00
    add([4, 5], [0], [1, 3])
    # Strassen P3: X00(Y01-Y11)
    add([0], [1], [2])
    add([0], [6], [3], -1)
    # Strassen P4: X11(Y10-Y00)
    add([5], [5], [0])
    add([5], [0], [3], -1)
    # Strassen P5: (X00+X01)Y11
    add([0, 1], [6], [0, 2], -1) # P5 contributes -1 to C00, +1 to C01
    # Wait, P5 is in Z00 (-1) and Z01 (+1).
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # So P5: W0 += -1, W2 += 1.
    # My add call above: add([0, 1], [6], [0, 2], -1) adds -1 to W0 and -1 to W2.
    # I need W0 += -1 and W2 += 1.
    # So: add([0, 1], [6], [0], -1) and add([0, 1], [6], [2], 1)
    
    # Let's restart Group 1 with correct signs.
    # Reset U,V,W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # G1
    add([0, 5], [0, 6], [0, 3]) # P1
    add([4, 5], [0], [1, 3]) # P2
    add([0], [1], [2]) # P3 part 1
    add([0], [6], [3], -1) # P3 part 2
    add([5], [5], [0]) # P4 part 1
    add([5], [0], [3], -1) # P4 part 2
    add([0, 1], [6], [0], -1) # P5 part 1
    add([0, 1], [6], [2], 1) # P5 part 2
    add([4, 0], [0, 5], [1]) # P6: (X10-X00)(Y00+Y10) -> W1 += 1
    # P6: U4+1, U0-1, V0+1, V5+1, W1+1
    add([4], [0, 5], [1])
    add([0], [0, 5], [1], -1)
    # P7: (X01-X11)(Y10+Y11) -> W0 += 1
    add([1], [5, 6], [0])
    add([5], [5, 6], [0], -1)

    # G2: A_L (U 0,1,4,5) * B_T2 (V 2,3,4,7,8,9)
    # W indices: C02(4), C12(5), C03(6), C13(7), C04(8), C14(9)
    
    # Z00: X00Y00 + X01Y10
    add([0], [2], [4])
    add([1], [7], [4])
    # Z01: X00Y01 + X01Y11
    add([0], [3], [6])
    add([1], [8], [6])
    # Z02: X00Y02 + X01Y12
    add([0], [4], [8])
    add([1], [9], [8])
    # Z10: X10Y00 + X11Y10
    add([4], [2], [5])
    add([5], [7], [5])
    # Z11: X10Y01 + X11Y11
    add([4], [3], [7])
    add([5], [8], [7])
    # Z12: X10Y02 + X11Y12
    add([4], [4], [9])
    add([5], [9], [9])

    # G3: A_R (U 2,3,6,7) * B_B1 (V 10,11,15,16)
    # W indices: 0,1,2,3
    
    add([2, 7], [10, 16], [0, 3]) # P1
    add([6, 7], [10], [1, 3]) # P2
    add([2], [11], [2]) # P3
    add([2], [16], [3], -1)
    add([7], [15], [0]) # P4
    add([7], [10], [3], -1)
    add([2, 3], [16], [0], -1) # P5
    add([2, 3], [16], [2], 1)
    add([6], [10, 15], [1]) # P6
    add([2], [10, 15], [1], -1)
    add([3], [15, 16], [0]) # P7
    add([7], [15, 16], [0], -1)

    # G4: A_R (U 2,3,6,7) * B_B2 (V 12,13,14,17,18,19)
    # W indices: 4,5,6,7,8,9
    
    add([2], [12], [4])
    add([3], [17], [4])
    add([2], [13], [6])
    add([3], [18], [6])
    add([2], [14], [8])
    add([3], [19], [8])
    add([6], [12], [5])
    add([7], [17], [5])
    add([6], [13], [7])
    add([7], [18], [7])
    add([6], [14], [9])
    add([7], [19], [9])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
