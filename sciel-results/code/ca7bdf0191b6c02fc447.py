    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # ...
        ks = k_start
        ke = k_start + 1
        
        # M1 = (A00 + A11) * (B0,ks + B1,ke)
        # C0,ks += M1, C1,ke += M1
        U[0, c0] += 1
        U[3, c0] += 1
        V[7 * 0 + ks, c0] += 1
        V[7 * 1 + ke, c0] += 1
        W[2 * ks + 0, c0] += 1
        W[2 * ke + 1, c0] += 1
