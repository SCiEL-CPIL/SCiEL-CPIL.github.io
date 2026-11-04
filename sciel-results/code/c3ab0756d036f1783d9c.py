k6 = k0 + 6
for i in range(2):
    for j in range(2):
        ii = i0 + i
        jj = j0 + j
        add_rank1_term([ii*m+jj], [1.0],
                       [jj*p+k6], [1.0],
                       [k6*n+ii], [1.0])
