"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # Target rank is 39
    # We will build the matrices U, V, W column by column.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        """
        u_idx: (i, j) for A[i, j]
        v_idx: (j, k) for B[j, k]
        w_idx: (i, k) for C[i, k]
        """
        # Convert to flat indices
        # A is n x m -> index i*m + j
        # B is m x p -> index j*p + k
        # W is p x n -> index k*n + i
        
        u_flat = u_idx[0] * m + u_idx[1]
        v_flat = v_idx[0] * p + v_idx[1]
        w_flat = w_idx[1] * n + w_idx[0]
        
        U.append([0.0] * (len(U) + 1)) # Placeholder, we will fill later or use list of lists
        # It is more efficient to build lists of rows.
        # Let's collect terms first.
        pass

    # Better approach: collect (u_flat, v_flat, w_flat, coeff) tuples
    terms = []
    
    # Helper to add Strassen block
    def add_strassen_block(a_i_range, a_j_range, b_j_range, b_k_range, w_i_range, w_k_range):
        # a_i_range: range of i for A
        # a_j_range: range of j for A
        # b_j_range: range of j for B
        # b_k_range: range of k for B
        # w_i_range: range of i for W
        # w_k_range: range of k for W
        
        # Map local indices to global
        # A local (0,0) -> Global (a_i_range.start + 0, a_j_range.start + 0)
        # B local (0,0) -> Global (b_j_range.start + 0, b_k_range.start + 0)
        # W local (0,0) -> Global (w_k_range.start + 0, w_i_range.start + 0)
        
        # Strassen products
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # We need to expand these into linear terms.
        # P1 = ae + ah + de + dh
        # P2 = ce + de
        # P3 = af - ah
        # P4 = dg - de
        # P5 = ah + bh
        # P6 = ce + cf - ae - af
        # P7 = bg + bh - dg - dh
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Let's define the linear combinations for each output element
        # Output C[0,0] (local) corresponds to global W[w_i_range.start, w_k_range.start]
        
        def add_linear_combo(w_local_i, w_local_k, coeffs):
            # coeffs is a dict mapping (a_local_i, a_local_j, b_local_j, b_local_k) to coeff
            # Actually, we just add terms to the global list
            for (ai, aj), (bj, bk), coeff in coeffs:
                global_i = a_i_range.start + ai
                global_j_a = a_j_range.start + aj
                global_j_b = b_j_range.start + bj
                global_k = b_k_range.start + bk
                
                # Check if indices are valid (they should be)
                u_flat = global_i * m + global_j_a
                v_flat = global_j_b * p + global_k
                w_flat = global_k * n + global_i
                
                terms.append((u_flat, v_flat, w_flat, coeff))
                
        # Define the 7 products as sums of rank-1 terms
        # Note: Strassen is an exact decomposition.
        
        # P1 terms
        p1 = [((0,0), (0,0), 1), ((0,1), (0,1), 1), ((1,0), (0,0), 1), ((1,1), (1,1), 1)]
        # P2 terms
        p2 = [((1,0), (0,0), 1), ((1,1), (0,0), 1)]
        # P3 terms
        p3 = [((0,0), (0,1), 1), ((0,1), (0,1), -1)]
        # P4 terms
        p4 = [((1,1), (1,0), 1), ((1,1), (0,0), -1)]
        # P5 terms
        p5 = [((0,1), (1,1), 1), ((1,1), (1,1), 1)]
        # P6 terms
        p6 = [((1,0), (0,0), 1), ((1,0), (0,1), 1), ((0,0), (0,0), -1), ((0,0), (0,1), -1)]
        # P7 terms
        p7 = [((0,1), (1,0), 1), ((0,1), (1,1), 1), ((1,1), (1,0), -1), ((1,1), (1,1), -1)]
        
        # Combine for C outputs
        # C00 = P1 + P4 - P5 + P7
        c00 = {}
        for t in p1: c00[(t[0], t[1])] = c00.get((t[0], t[1]), 0) + t[2]
        for t in p4: c00[(t[0], t[1])] = c00.get((t[0], t[1]), 0) + t[2]
        for t in p5: c00[(t[0], t[1])] = c00.get((t[0], t[1]), 0) - t[2]
        for t in p7: c00[(t[0], t[1])] = c00.get((t[0], t[1]), 0) + t[2]
        
        # C01 = P3 + P5
        c01 = {}
        for t in p3: c01[(t[0], t[1])] = c01.get((t[0], t[1]), 0) + t[2]
        for t in p5: c01[(t[0], t[1])] = c01.get((t[0], t[1]), 0) + t[2]
        
        # C10 = P2 + P4
        c10 = {}
        for t in p2: c10[(t[0], t[1])] = c10.get((t[0], t[1]), 0) + t[2]
        for t in p4: c10[(t[0], t[1])] = c10.get((t[0], t[1]), 0) + t[2]
        
        # C11 = P1 - P2 + P3 + P6
        c11 = {}
        for t in p1: c11[(t[0], t[1])] = c11.get((t[0], t[1]), 0) + t[2]
        for t in p2: c11[(t[0], t[1])] = c11.get((t[0], t[1]), 0) - t[2]
        for t in p3: c11[(t[0], t[1])] = c11.get((t[0], t[1]), 0) + t[2]
        for t in p6: c11[(t[0], t[1])] = c11.get((t[0], t[1]), 0) + t[2]
        
        # Add to terms
        for (ai, aj), (bj, bk), coeff in c00.items():
            if coeff != 0:
                add_linear_combo(0, 0, [((ai, aj), (bj, bk), coeff)])
        # Wait, add_linear_combo takes a list. Let's just iterate.
        # Refactor: add_linear_combo should just add one term.
        
        # Let's redo the addition loop directly in terms
        def add_terms_to_list(coeffs_dict):
            for (ai, aj), (bj, bk), coeff in coeffs_dict.items():
                if coeff != 0:
                    global_i = a_i_range.start + ai
                    global_j_a = a_j_range.start + aj
                    global_j_b = b_j_range.start + bj
                    global_k = b_k_range.start + bk
                    u_flat = global_i * m + global_j_a
                    v_flat = global_j_b * p + global_k
                    w_flat = global_k * n + global_i
                    terms.append((u_flat, v_flat, w_flat, coeff))
        
        add_terms_to_list(c00)
        add_terms_to_list(c01)
        add_terms_to_list(c10)
        add_terms_to_list(c11)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(range(0,2), range(0,2), range(0,2), range(0,2), range(0,2), range(0,2))
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(range(0,2), range(0,2), range(0,2), range(2,4), range(0,2), range(2,4))
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(range(0,2), range(0,2), range(0,2), range(4,6), range(0,2), range(4,6))
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6:7] -> C[0:2, 6:7]
    # This is 2x2 by 2x1. Rank 4.
    # C[i, 6] = A[i,0]B[0,6] + A[i,1]B[1,6]
    for i in range(2):
        for j in range(2):
            u_flat = i * m + j
            v_flat = j * p + 6
            w_flat = 6 * n + i
            terms.append((u_flat, v_flat, w_flat, 1))
            
    # Block 5: A[0:2, 2:3] * B[2:3, 0:7] -> C[0:2, 0:7]
    # A is 2x1, B is 1x7. Rank 14.
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(7):
            u_flat = i * m + 2
            v_flat = 2 * p + k
            w_flat = k * n + i
            terms.append((u_flat, v_flat, w_flat, 1))

    rank = len(terms)
    
    # Construct U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_flat, v_flat, w_flat, coeff) in enumerate(terms):
        U[u_flat][c] = coeff
        V[v_flat][c] = coeff
        W[w_flat][c] = coeff
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
