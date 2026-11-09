"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    U_list = []
    V_list = []
    W_list = []

    def add_term(i, j, k, coeff):
        """
        Add a term to the decomposition.
        i: row of A, col of C
        j: col of A, row of B
        k: col of B, row of C
        
        U index: i*m + j
        V index: j*p + k
        W index: k*n + i
        """
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        U_row[u_idx] = coeff
        V_row[v_idx] = coeff
        W_row[w_idx] = coeff
        
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    # Block decomposition: 2 + 2 + 1
    # Blocks are indexed 0, 1, 2
    # Block 0: indices 0-1
    # Block 1: indices 2-3
    # Block 2: index 4
    
    def get_global_indices(block_id, local_indices):
        """
        Maps local indices within a block to global indices.
        block_id: 0, 1, or 2
        local_indices: list of local indices for that block
        """
        globals = []
        for li in local_indices:
            if block_id == 0:
                globals.append(li)
            elif block_id == 1:
                globals.append(2 + li)
            else:
                globals.append(4)
        return globals

    def add_strassen_block(bi, bj, bk):
        """
        Adds terms for a 2x2x2 block multiplication using Strassen.
        bi, bj, bk: block IDs (must be 0 or 1 for 2x2 blocks)
        """
        if bi not in [0, 1] or bj not in [0, 1] or bk not in [0, 1]:
            return # Strassen only for 2x2 blocks

        # Local indices for 2x2 blocks are 0, 1
        # Strassen algorithm:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B00)
        # M4 = A11 * (B01 - B11)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # We iterate over the 7 Strassen multiplications.
        # Each multiplication involves sums of A and B entries.
        # We need to expand these into individual (i, j, k) terms.
        
        # Helper to get global indices for local (li, lj, lk)
        def g_idx(block, local):
            if block == 0:
                return local
            else:
                return 2 + local

        # Define the 7 Strassen products as lists of (i, j, k, sign) tuples
        # where the term is sign * A[bi, i][bj, j] * B[bj, j][bk, k]
        # Wait, Strassen combines A and B entries first.
        # e.g., M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        
        # Let's generate the terms for each M product
        # M1: (A00+A11)(B00+B11)
        # A indices: (0,0), (1,1) relative to block bi, bj
        # B indices: (0,0), (1,1) relative to block bj, bk
        
        strassen_products = [
            # M1
            [(0,0,0,1), (0,0,1,1), (1,1,0,1), (1,1,1,1)],
            # M2
            [(1,0,0,1), (1,1,0,1)],
            # M3
            [(0,0,1,1), (0,0,0,-1)],
            # M4
            [(1,1,1,1), (1,1,0,-1)],
            # M5
            [(0,0,1,1), (0,1,1,1)],
            # M6
            [(1,0,0,1), (1,0,1,1), (0,0,0,-1), (0,0,1,-1)],
            # M7
            [(0,1,1,1), (0,1,0,1), (1,1,1,-1), (1,1,0,-1)]
        ]
        
        # Coefficients for C blocks
        # C00: M1 + M4 - M5 + M7
        # C01: M3 + M5
        # C10: M2 + M4
        # C11: M1 - M2 + M3 + M6
        
        # We need to sum the contributions to each C entry.
        # C[li, lk] = sum over m of coeff_m * M_m
        # Each M_m is a sum of A[i,j]B[j,k]
        
        # Let's accumulate coefficients for each (li, lj, lk) in the block
        # local i, j, k are in {0, 1}
        coeffs = {}
        
        for m_idx, terms in enumerate(strassen_products):
            for (li, lj, lk, sign) in terms:
                key = (li, lj, lk)
                if key not in coeffs:
                    coeffs[key] = 0.0
                coeffs[key] += sign
                
        # Now apply the C block combination
        # C00 (li=0, lk=0): M1(1) + M4(1) - M5(1) + M7(1)
        # C01 (li=0, lk=1): M3(1) + M5(1)
        # C10 (li=1, lk=0): M2(1) + M4(1)
        # C11 (li=1, lk=1): M1(1) - M2(1) + M3(1) + M6(1)
        
        # We need to map the accumulated M coefficients to C coefficients.
        # Actually, the `coeffs` dict above sums the *input* terms to M.
        # But we need to sum the *output* contributions to C.
        
        # Let's redo this more carefully.
        # We want to add terms to the global U, V, W.
        # A term in C[li, lk] comes from A[li, lj] * B[lj, lk].
        # The coefficient of this term in the Strassen decomposition is determined by how A[li, lj] and B[lj, lk] participate in the M's and how the M's combine into C[li, lk].
        
        # Let's define the participation of each (li, lj) in A and (lj, lk) in B for each M.
        # M1: A00, A11; B00, B11
        # M2: A10, A11; B00
        # M3: A00; B10, B00
        # M4: A11; B01, B11
        # M5: A00, A01; B11
        # M6: A10, A00; B00, B01
        # M7: A01, A11; B10, B11
        
        # And the combination into C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's calculate the final coefficient for each (li, lj, lk) triple.
        # This is a linear combination.
        # coeff[li, lj, lk] = sum over M of (weight of M in C[li, lk]) * (coeff of A[li, lj] in M) * (coeff of B[lj, lk] in M)
        
        # Note: In Strassen, the "coeff" of A in M is 1 or -1.
        
        # M1: A00(1), A11(1); B00(1), B11(1)
        # M2: A10(1), A11(1); B00(1)
        # M3: A00(1); B10(1), B00(-1)  <-- Wait, M3 = A00(B10-B00)
        # M4: A11(1); B01(1), B11(-1)  <-- M4 = A11(B01-B11)
        # M5: A00(1), A01(1); B11(1)
        # M6: A10(1), A00(-1); B00(1), B01(1) <-- M6 = (A10-A00)(B00+B01)
        # M7: A01(1), A11(-1); B10(1), B11(1) <-- M7 = (A01-A11)(B10+B11)
        
        a_coeffs = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):1},
            2: {(0,0):1},
            3: {(1,1):1},
            4: {(0,0):1, (0,1):1},
            5: {(1,0):1, (0,0):-1},
            6: {(0,1):1, (1,1):-1}
        }
        
        b_coeffs = {
            0: {(0,0):1, (1,1):1},
            1: {(0,0):1},
            2: {(1,0):1, (0,0):-1},
            3: {(0,1):1, (1,1):-1},
            4: {(1,1):1},
            5: {(0,0):1, (0,1):1},
            6: {(1,0):1, (1,1):1}
        }
        
        c_weights = {
            (0,0): {0:1, 3:1, 4:-1, 6:1}, # M1, M4, -M5, M7
            (0,1): {2:1, 4:1},             # M3, M5
            (1,0): {1:1, 3:1},             # M2, M4
            (1,1): {0:1, 1:-1, 2:1, 5:1}   # M1, -M2, M3, M6
        }
        
        for li in range(2):
            for lk in range(2):
                weights = c_weights[(li, lk)]
                for lj in range(2):
                    total_coeff = 0.0
                    for m_idx, w in weights.items():
                        a_c = a_coeffs[m_idx].get((li, lj), 0)
                        b_c = b_coeffs[m_idx].get((lj, lk), 0)
                        if a_c != 0 and b_c != 0:
                            total_coeff += w * a_c * b_c
                            
                    if total_coeff != 0:
                        g_i = g_idx(bi, li)
                        g_j = g_idx(bj, lj)
                        g_k = g_idx(bk, lk)
                        add_term(g_i, g_j, g_k, total_coeff)

    def add_schoolbook_block(bi, bj, bk):
        """
        Adds terms for a block multiplication using schoolbook.
        bi, bj, bk: block IDs
        """
        size_i = 2 if bi in [0, 1] else 1
        size_j = 2 if bj in [0, 1] else 1
        size_k = 2 if bk in [0, 1] else 1
        
        def g_idx(block, local):
            if block == 0:
                return local
            elif block == 1:
                return 2 + local
            else:
                return 4

        for li in range(size_i):
            for lj in range(size_j):
                for lk in range(size_k):
                    g_i = g_idx(bi, li)
                    g_j = g_idx(bj, lj)
                    g_k = g_idx(bk, lk)
                    add_term(g_i, g_j, g_k, 1.0)

    # Iterate over all block combinations
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # Check if it's a 2x2x2 block
                if bi in [0, 1] and bj in [0, 1] and bk in [0, 1]:
                    add_strassen_block(bi, bj, bk)
                else:
                    add_schoolbook_block(bi, bj, bk)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
