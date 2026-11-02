    for I in range(3):
        for J in range(3):
            for K in range(3):
                for t in range(7):
                    # Process U
                    for (u_r, u_c), coeff in strassen_terms[t]['u']:
                        ...
                        U[idx_u][col] += coeff
                    ...
                    col += 1
