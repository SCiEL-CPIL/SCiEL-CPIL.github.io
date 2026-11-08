import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the known rank-49 recursive Strassen decomposition.
    2. Attempt to reduce the rank by identifying and eliminating linearly dependent terms
       or redundant terms through a series of local optimization passes.
    3. If a lower rank decomposition is found, return it; otherwise return rank-49.
    
    The evaluator scores as seed/rank, so lower rank is better.
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    
    # ============================================================
    # Step 1: Build the rank-49 recursive Strassen decomposition
    # ============================================================
    
    def build_strassen_49():
        """Build the rank-49 decomposition using recursive Strassen."""
        rank = 49
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Block indexing helpers
        def a_elem(bi, bj, r, c):
            return (2*bi + r, 2*bj + c)
        
        def b_elem(bi, bj, r, c):
            return (2*bi + r, 2*bj + c)
        
        def c_elem(bi, bj, r, c):
            return (2*bi + r, 2*bj + c)
        
        def get_U_index(row, col):
            return row * m + col
        
        def get_V_index(row, col):
            return row * p + col
        
        def get_W_index(row, col):
            return col * n + row
        
        # Define the 7 block-level Strassen terms
        L_terms = []
        R_terms = []
        contrib_terms = []
        
        # Term 0: P1 = (A11 + A22)(B11 + B22)
        L0 = [[{} for _ in range(2)] for _ in range(2)]
        R0 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
                er, ec = b_elem(0, 0, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R0[r][c] = {(er, ec): 1, (hr, hc): 1}
        contrib0 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib0[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib0[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L0)
        R_terms.append(R0)
        contrib_terms.append(contrib0)
        
        # Term 1: P2 = (A21 + A22)B11
        L1 = [[{} for _ in range(2)] for _ in range(2)]
        R1 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(1, 0, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
                er, ec = b_elem(0, 0, r, c)
                R1[r][c] = {(er, ec): 1}
        contrib1 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(1, 0, r, c)
                contrib1[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib1[(r, c)].append((cr2, cc2, -1))
        L_terms.append(L1)
        R_terms.append(R1)
        contrib_terms.append(contrib1)
        
        # Term 2: P3 = A11(B12 - B22)
        L2 = [[{} for _ in range(2)] for _ in range(2)]
        R2 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                L2[r][c] = {(ar, ac): 1}
                fr, fc = b_elem(0, 1, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
        contrib2 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 1, r, c)
                contrib2[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib2[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L2)
        R_terms.append(R2)
        contrib_terms.append(contrib2)
        
        # Term 3: P4 = A22(B21 - B11)
        L3 = [[{} for _ in range(2)] for _ in range(2)]
        R3 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                dr, dc = a_elem(1, 1, r, c)
                L3[r][c] = {(dr, dc): 1}
                gr, gc = b_elem(1, 0, r, c)
                er, ec = b_elem(0, 0, r, c)
                R3[r][c] = {(gr, gc): 1, (er, ec): -1}
        contrib3 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib3[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 0, r, c)
                contrib3[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L3)
        R_terms.append(R3)
        contrib_terms.append(contrib3)
        
        # Term 4: P5 = (A11 + A12)B22
        L4 = [[{} for _ in range(2)] for _ in range(2)]
        R4 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                br, bc = a_elem(0, 1, r, c)
                L4[r][c] = {(ar, ac): 1, (br, bc): 1}
                hr, hc = b_elem(1, 1, r, c)
                R4[r][c] = {(hr, hc): 1}
        contrib4 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib4[(r, c)] = [(cr, cc, -1)]
                cr2, cc2 = c_elem(0, 1, r, c)
                contrib4[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L4)
        R_terms.append(R4)
        contrib_terms.append(contrib4)
        
        # Term 5: P6 = (A21 - A11)(B11 + B12)
        L5 = [[{} for _ in range(2)] for _ in range(2)]
        R5 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                cr, cc = a_elem(1, 0, r, c)
                ar, ac = a_elem(0, 0, r, c)
                L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
                er, ec = b_elem(0, 0, r, c)
                fr, fc = b_elem(0, 1, r, c)
                R5[r][c] = {(er, ec): 1, (fr, fc): 1}
        contrib5 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(1, 1, r, c)
                contrib5[(r, c)] = [(cr, cc, 1)]
        L_terms.append(L5)
        R_terms.append(R5)
        contrib_terms.append(contrib5)
        
        # Term 6: P7 = (A12 - A22)(B21 + B22)
        L6 = [[{} for _ in range(2)] for _ in range(2)]
        R6 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                br, bc = a_elem(0, 1, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L6[r][c] = {(br, bc): 1, (dr, dc): -1}
                gr, gc = b_elem(1, 0, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
        contrib6 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib6[(r, c)] = [(cr, cc, 1)]
        L_terms.append(L6)
        R_terms.append(R6)
        contrib_terms.append(contrib6)
        
        # Strassen sub-product contributions
        sub_contrib = [
            [(0,0,1), (1,1,1)],    # M1
            [(1,0,1), (1,1,-1)],   # M2
            [(0,1,1), (1,1,1)],    # M3
            [(0,0,1), (1,0,1)],    # M4
            [(0,0,-1), (0,1,1)],   # M5
            [(1,1,1)],             # M6
            [(0,0,1)]              # M7
        ]
        
        # Build U, V, W
        for t in range(7):
            L_t = L_terms[t]
            R_t = R_terms[t]
            contrib_t = contrib_terms[t]
            
            a = L_t[0][0]
            b = L_t[0][1]
            c = L_t[1][0]
            d = L_t[1][1]
            e = R_t[0][0]
            f = R_t[0][1]
            g = R_t[1][0]
            h = R_t[1][1]
            
            # Define the 7 sub-products
            sub_products = []
            
            # M1: X = a+d, Y = e+h
            x1 = {}
            for key, coeff in a.items():
                x1[key] = x1.get(key, 0) + coeff
            for key, coeff in d.items():
                x1[key] = x1.get(key, 0) + coeff
            y1 = {}
            for key, coeff in e.items():
                y1[key] = y1.get(key, 0) + coeff
            for key, coeff in h.items():
                y1[key] = y1.get(key, 0) + coeff
            sub_products.append((x1, y1))
            
            # M2: X = c+d, Y = e
            x2 = {}
            for key, coeff in c.items():
                x2[key] = x2.get(key, 0) + coeff
            for key, coeff in d.items():
                x2[key] = x2.get(key, 0) + coeff
            y2 = dict(e)
            sub_products.append((x2, y2))
            
            # M3: X = a, Y = f-h
            x3 = dict(a)
            y3 = {}
            for key, coeff in f.items():
                y3[key] = y3.get(key, 0) + coeff
            for key, coeff in h.items():
                y3[key] = y3.get(key, 0) - coeff
            sub_products.append((x3, y3))
            
            # M4: X = d, Y = g-e
            x4 = dict(d)
            y4 = {}
            for key, coeff in g.items():
                y4[key] = y4.get(key, 0) + coeff
            for key, coeff in e.items():
                y4[key] = y4.get(key, 0) - coeff
            sub_products.append((x4, y4))
            
            # M5: X = a+b, Y = h
            x5 = {}
            for key, coeff in a.items():
                x5[key] = x5.get(key, 0) + coeff
            for key, coeff in b.items():
                x5[key] = x5.get(key, 0) + coeff
            y5 = dict(h)
            sub_products.append((x5, y5))
            
            # M6: X = c-a, Y = e+f
            x6 = {}
            for key, coeff in c.items():
                x6[key] = x6.get(key, 0) + coeff
            for key, coeff in a.items():
                x6[key] = x6.get(key, 0) - coeff
            y6 = {}
            for key, coeff in e.items():
                y6[key] = y6.get(key, 0) + coeff
            for key, coeff in f.items():
                y6[key] = y6.get(key, 0) + coeff
            sub_products.append((x6, y6))
            
            # M7: X = b-d, Y = g+h
            x7 = {}
            for key, coeff in b.items():
                x7[key] = x7.get(key, 0) + coeff
            for key, coeff in d.items():
                x7[key] = x7.get(key, 0) - coeff
            y7 = {}
            for key, coeff in g.items():
                y7[key] = y7.get(key, 0) + coeff
            for key, coeff in h.items():
                y7[key] = y7.get(key, 0) + coeff
            sub_products.append((x7, y7))
            
            # Add to U, V, W
            for k in range(7):
                x_k, y_k = sub_products[k]
                rank_idx = t * 7 + k
                
                # Add to U (from x_k, which are A elements)
                for (row, col), coeff in x_k.items():
                    u_idx = get_U_index(row, col)
                    U[u_idx, rank_idx] += coeff
                
                # Add to V (from y_k, which are B elements)
                for (row, col), coeff in y_k.items():
                    v_idx = get_V_index(row, col)
                    V[v_idx, rank_idx] += coeff
                
                # Add to W (contributions from sub_contrib[k] through contrib_t)
                for (pr, pc, sub_coeff) in sub_contrib[k]:
                    if (pr, pc) in contrib_t:
                        for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                            w_idx = get_W_index(out_row, out_col)
                            W[w_idx, rank_idx] += sub_coeff * block_coeff
        
        return U, V, W
    
    U, V, W = build_strassen_49()
    current_rank = 49
    
    # ============================================================
    # Step 2: Verify the decomposition
    # ============================================================
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for out_r in range(4):
            for out_c in range(4):
                computed = 0.0
                for k in range(4):
                    a_idx = out_r * 4 + k
                    b_idx = k * 4 + out_c
                    c_idx = out_c * 4 + out_r
                    for r in range(rank):
                        computed += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if abs(computed - 1) > 1e-10:
                    return False
        return True
    
    if not verify_decomposition(U, V, W, current_rank):
        # If verification fails, return as-is (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # ============================================================
    # Step 3: Attempt to reduce rank by eliminating redundant terms
    # ============================================================
    
    def can_remove_term(U, V, W, rank, term_idx):
        """Check if removing term_idx still maintains a valid decomposition."""
        if rank <= 1:
            return False
        
        # Create modified arrays without term_idx
        U_mod = U.copy()
        V_mod = V.copy()
        W_mod = W.copy()
        
        # Set the term to zero
        U_mod[:, term_idx] = 0
        V_mod[:, term_idx] = 0
        W_mod[:, term_idx] = 0
        
        # Check if this term is actually used (non-zero in at least one matrix)
        term_used = (np.any(U[:, term_idx] != 0) or 
                    np.any(V[:, term_idx] != 0) or 
                    np.any(W[:, term_idx] != 0))
        
        if not term_used:
            # This term is already zero, removing it doesn't change anything
            # But we can still reduce rank
            return True
        
        # Verify the decomposition still works without this term
        # This is expensive, so only do it if the term seems redundant
        # Quick check: is this term linearly dependent on others?
        
        # For now, do a full verification
        for out_r in range(4):
            for out_c in range(4):
                computed = 0.0
                for k in range(4):
                    a_idx = out_r * 4 + k
                    b_idx = k * 4 + out_c
                    c_idx = out_c * 4 + out_r
                    for r in range(rank):
                        if r == term_idx:
                            continue
                        computed += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if abs(computed - 1) > 1e-10:
                    return False
        
        return True
    
    def remove_term(U, V, W, rank, term_idx):
        """Remove a term from the decomposition."""
        # Keep only terms != term_idx
        keep = [i for i in range(rank) if i != term_idx]
        
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        return U_new, V_new, W_new, rank - 1
    
    # Try to remove terms one by one
    removed_any = True
    while removed_any and (time.time() - start_time) < DEADLINE - 10:
        removed_any = False
        for term_idx in range(current_rank - 1, -1, -1):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            if can_remove_term(U, V, W, current_rank, term_idx):
                U, V, W, current_rank = remove_term(U, V, W, current_rank, term_idx)
                removed_any = True
                print(f"Removed term {term_idx}, new rank: {current_rank}")
                break  # Restart the loop after removal
    
    # ============================================================
    # Step 4: Final verification
    # ============================================================
    
    if not verify_decomposition(U, V, W, current_rank):
        # If something went wrong, fall back to the original rank-49
        U, V, W = build_strassen_49()
        current_rank = 49
    
    elapsed = time.time() - start_time
    print(f"Final rank: {current_rank}, elapsed: {elapsed:.2f}s")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
