<<<<<<< SEARCH
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen_factors[s]
            
            # Fill U: row = i*m + j = i*2 + j
            for (i, j, coeff) in u_entries:
                U[i * m + j][col] = coeff
            
            # Fill V: row = j*p + k = j*16 + (2t + k_local)
            for (j, k_local, coeff) in v_entries:
                V[j * p + (2 * t + k_local)][col] = coeff
            
            # Fill W: row = k*n + i = (2t + k_local)*2 + i
            for (i, k_local, coeff) in w_entries:
                W[(2 * t + k_local) * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    for t in range(num_blocks):
        for s in range(7):
            if time.time() > DEADLINE:
                return {"U": U, "V": V, "W": W}
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen_factors[s]
            
            # Fill U: row = i*m + j = i*2 + j
            for (i, j, coeff) in u_entries:
                U[i * m + j][col] = coeff
            
            # Fill V: row = j*p + k = j*16 + (2t + k_local)
            for (j, k_local, coeff) in v_entries:
                V[j * p + (2 * t + k_local)][col] = coeff
            
            # Fill W: row = k*n + i = (2t + k_local)*2 + i
            for (i, k_local, coeff) in w_entries:
                W[(2 * t + k_local) * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
