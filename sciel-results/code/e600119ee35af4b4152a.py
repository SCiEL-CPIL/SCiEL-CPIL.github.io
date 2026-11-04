<<<<<<< SEARCH
                    # W part: The result M[r, c] contributes to C blocks
                    # C[ib, kb, r, c] += C_coeffs[(ib,kb)] * M[r,c]
                    # W row index (trace convention): kb*3+c is the "k" index, ib*3+r is the "i" index
                    # W[k*6+i, col] where k = kb*3+c, i = ib*3+r
                    # So W row index: (kb*3+c) * 6 + (ib*3+r)
                    
                    for (ib, kb), c_coeff in C_coeffs.items():
                        w_k = kb * block_size + c
                        w_i = ib * block_size + r
                        w_idx = w_k * 6 + w_i
                        W[w_idx, col] += c_coeff
=======
                    # W part: The result M[r, c] contributes to C blocks
                    # C[ib, kb, r, c] += C_coeffs[(ib,kb)] * M[r,c]
                    # Standard convention: W row index is i*6+k
                    # i = ib*3+r, k = kb*3+c
                    # So W row index: (ib*3+r) * 6 + (kb*3+c)
                    
                    for (ib, kb), c_coeff in C_coeffs.items():
                        w_i = ib * block_size + r
                        w_k = kb * block_size + c
                        w_idx = w_i * 6 + w_k
                        W[w_idx, col] += c_coeff
>>>>>>> REPLACE
