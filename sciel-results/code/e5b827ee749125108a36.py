U = I
for k in range(12):
    U = expm(-1j * b[k] * B) @ U  # This is wrong order if U accumulates from left
    # If U starts as I, and we do U = M @ U, then after k=0, U = M0.
    # After k=1, U = M1 @ M0.
    # The formula is M1 M2 ... M12.
    # So U = M_k @ U is correct for M1 M2 ... Mk.
