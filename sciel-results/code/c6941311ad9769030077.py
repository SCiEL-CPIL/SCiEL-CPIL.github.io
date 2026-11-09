for (cr, cc, coeff) in get_C_contrib(s, i, k):
    # C[cr][cc] gets contribution, W index is cc*6 + cr (since k*6+i where k=cc, i=cr)
    W[cc * 6 + cr][col] += coeff
