for li, ar in enumerate(a_rows):
    for lk, bc in enumerate(b_cols):
        W[bc * n + ar][rank] += sW[lk * 2 + li][r]
