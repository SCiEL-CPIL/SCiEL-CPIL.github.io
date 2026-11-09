<<<<<<< SEARCH
    U2 = [
        # row 0
        [ 1,  0,  1,  0,  1,  0,  0],  # A00
        [ 0,  0,  0,  0,  1,  0,  1],  # A01
        # row 1
        [ 1,  1,  0,  0,  0, -1,  0],  # A10
        [ 1,  1,  0,  1,  0,  0, -1],  # A11
    ]
    
    V2 = [
        # row 0
        [ 1,  1,  0, -1,  0,  1,  0],  # B00
        [ 0,  0,  1,  0,  0,  0,  1],  # B01
        # row 1
        [ 1,  0, -1,  0,  1,  1,  0],  # B10
        [ 1,  0,  0,  1,  1,  0,  1],  # B11
    ]
    
    # W index = col*2 + row
    # col 0, row 0 (C00): P0 - P3 + P5 + P6
    # col 0, row 1 (C10): P1 + P3
    # col 1, row 0 (C01): P2 + P4
    # col 1, row 1 (C11): P0 - P2 + P1 + P4
    W2 = [
        # index 0: C00 (col=0, row=0)
        [ 1,  0,  0, -1,  0,  1,  1],
        # index 1: C10 (col=0, row=1)
        [ 0,  1,  0,  1,  0,  0,  0],
        # index 2: C01 (col=1, row=0)
        [ 0,  0,  1,  0,  1,  0,  0],
        # index 3: C11 (col=1, row=1)
        [ 1,  1, -1,  0,  1,  0,  0],
    ]
=======
    # U2[row][col][alpha]: coefficient for A[row,col] in product alpha
    U2 = [
        [
            [ 1,  0,  1,  0,  1, -1,  0],  # A00
            [ 0,  0,  0,  0,  1,  0, -1],  # A01
        ],
        [
            [ 0,  1,  0,  0,  0,  1,  0],  # A10
            [ 1,  1,  0,  1,  0,  0, -1],  # A11
        ]
    ]
    
    # V2[row][col][alpha]: coefficient for B[row,col] in product alpha
    V2 = [
        [
            [ 1,  1,  0, -1,  0,  1,  0],  # B00
            [ 0,  0,  1,  0,  0,  0,  1],  # B01
        ],
        [
            [ 0,  0, -1,  1,  0,  1,  0],  # B10
            [ 1,  0, -1,  0,  1,  0,  1],  # B11
        ]
    ]
    
    # W2[col][row][alpha]: coefficient for C[row,col] from product alpha
    # Note: The code accesses W2[K][I] where K is col-block, I is row-block.
    # W index in final matrix is col*4 + row.
    W2 = [
        [
            [ 1,  0,  0, -1,  0,  1,  1],  # C00 (col=0, row=0)
            [ 0,  1,  0,  1,  0,  0,  0],  # C10 (col=0, row=1)
        ],
        [
            [ 0,  0,  1,  0,  1,  0,  0],  # C01 (col=1, row=0)
            [ 1,  1, -1,  0,  1,  0,  0],  # C11 (col=1, row=1)
        ]
    ]
>>>>>>> REPLACE

And update the W loop:

