    # Let me redo this properly with explicit index mapping
    rank = 42
    U = np.zeros((8, rank))
    ...
    col_idx = 0
    for source in range(2):
        ...
                if t == 0:
                    # M1 = (P00+P11)*(Q00+Q11)
                    # P00 = A[0, 0+2*source], P11 = A[1, 1+2*source]
                    U[0*4 + (0+2*source), col_idx] += 1
