def add_2x2x3_rank10(a_offset, b_offset, c_offset, sign=1):
    i0, j0 = a_offset
    j1, k0 = b_offset
    c_i0, c_k0 = c_offset
    # Terms for 2x2x3 with rank 10
    # Using known decomposition
    terms = [
        # First 2 columns (Strassen-like)
        ({(i0, j0): 1, (i0+1, j0+1): 1}, {(j0, k0): 1, (j0+1, k0+1): 1}, {(i0, k0): 1, (i0+1, k0+1): 1}),
        ({(i0+1, j0): 1, (i0+1, j0+1): 1}, {(j0, k0): 1}, {(i0+1, k0): 1, (i0+1, k0+1): -1}),
        ({(i0, j0): 1}, {(j0, k0+1): 1, (j0+1, k0+1): -1}, {(i0, k0+1): 1, (i0+1, k0+1): 1}),
        ({(i0+1, j0+1): 1}, {(j0+1, k0): 1, (j0, k0): -1}, {(i0, k0): 1, (i0+1, k0): 1}),
        ({(i0, j0): 1, (i0, j0+1): 1}, {(j0+1, k0+1): 1}, {(i0, k0): -1, (i0, k0+1): 1}),
        ({(i0+1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k0+1): 1}, {(i0+1, k0+1): 1}),
        ({(i0, j0+1): 1, (i0+1, j0+1): -1}, {(j0+1, k0): 1, (j0+1, k0+1): 1}, {(i0, k0): 1}),
        # Additional 3 terms for the 3rd column
        ({(i0, j0): 1, (i0, j0+1): 1}, {(j0, k0+2): 1, (j0+1, k0+2): 1}, {(i0, k0+2): 1, (i0+1, k0+2): 1}),
        ({(i0+1, j0): 1, (i0+1, j0+1): 1}, {(j0, k0+2): 1}, {(i0+1, k0+2): 1, (i0+1, k0+2): -1}), # This is a guess
        ({(i0, j0): 1}, {(j0, k0+2): 1, (j0+1, k0+2): -1}, {(i0, k0+2): 1, (i0+1, k0+2): 1})
    ]
    # This is likely incorrect. I will use the 36-rank solution to be safe.
    pass
