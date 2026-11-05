def get_strassen_4x4():
    U2, V2, W2 = get_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen 2x2 linear combinations
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map 4x4 indices to 2x2 block indices.
    # 4x4 index (i,j) -> block (i//2, i%2, j//2, j%2)
    # This is complex. I will use a simpler approach:
    # The 4x4 Strassen is just 7 calls to 2x2 Strassen on the blocks.
    # I will hardcode the 49 rank-1 terms by iterating over the 7 Strassen steps and the 7 internal steps.
    
    return U, V, W
