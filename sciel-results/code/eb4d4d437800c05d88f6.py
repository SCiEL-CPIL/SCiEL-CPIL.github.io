import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # We will build lists of (u_idx, v_idx, w_idx, coeff)
    # But since we sum, we can just accumulate into matrices.
    # However, rank is the number of terms.
    # We can use a list of tuples (u_idx, v_idx, w_idx) and then convert to matrices.
    
    terms = []
    
    # 1. Core 4x4x4 multiplication using Strassen (Rank 49)
    # We need the 7 Strassen products for 2x2x2
    # u_coeffs: list of 7 lists of 4 coeffs for A (2x2)
    # v_coeffs: list of 7 lists of 4 coeffs for B (2x2)
    # w_coeffs: list of 7 lists of 4 coeffs for C (2x2)
    
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # P1 = A00(B11 - B10)
    # P2 = (A00 + A01)B10
    # P3 = (A10 + A11)B00
    # P4 = A11(B01 - B00)
    # P5 = (A00 + A11)(B00 + B11)
    # P6 = (A01 - A11)(B10 + B11)
    # P7 = (A00 - A10)(B00 + B01)
    
    # C00 = P5 + P4 - P2 + P6
    # C01 = P1 + P2
    # C10 = P3 + P4
    # C11 = P5 + P1 - P3 + P7
    
    # Define the 7 products' linear forms
    # Format: (u_list, v_list, w_list)
    # u_list is coeff for A00, A01, A10, A11
    # v_list is coeff for B00, B01, B10, B11
    # w_list is coeff for C00, C01, C10, C11
    
    prod_defs = [
        # P1
        ([1, 0, 0, 0], [0, 0, -1, 1], [0, 1, 0, 0]), # C01 gets +1
        # P2
        ([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]), # C00 gets -1? No, C00 = ... - P2. C01 = P1 + P2.
        # Wait, w_list should contain the coefficients for C.
        # For P2, C00 coeff is -1, C01 coeff is 1.
        # So w_list = [-1, 1, 0, 0]
        # Let's re-evaluate w_list for each Pk based on C equations.
        
        # C00 = P5 + P4 - P2 + P6
        # C01 = P1 + P2
        # C10 = P3 + P4
        # C11 = P5 + P1 - P3 + P7
        
        # P1: C01 += 1. w=[0, 1, 0, 0]
        # P2: C00 -= 1, C01 += 1. w=[-1, 1, 0, 0]
        # P3: C10 += 1, C11 -= 1. w=[0, 0, 1, -1]
        # P4: C00 += 1, C10 += 1. w=[1, 0, 1, 0]
        # P5: C00 += 1, C11 += 1. w=[1, 0, 0, 1]
        # P6: C00 += 1. w=[1, 0, 0, 0]
        # P7: C11 += 1. w=[0, 0, 0, 1]
    ]
    
    # Correct u, v for each Pk:
    # P1 = A00(B11 - B10) -> u=[1,0,0,0], v=[0,0,-1,1]
    # P2 = (A00+A01)B10 -> u=[1,1,0,0], v=[0,0,1,0]
    # P3 = (A10+A11)B00 -> u=[0,0,1,1], v=[1,0,0,0]
    # P4 = A11(B01-B00) -> u=[0,0,0,1], v=[-1,1,0,0]
    # P5 = (A00+A11)(B00+B11) -> u=[1,0,0,1], v=[1,0,0,1]
    # P6 = (A01-A11)(B10+B11) -> u=[0,1,0,-1], v=[0,0,1,1]
    # P7 = (A00-A10)(B00+B01) -> u=[1,0,-1,0], v=[1,1,0,0]
    
    strassen_2x2 = [
        ([1,0,0,0], [0,0,-1,1], [0,1,0,0]),
        ([1,1,0,0], [0,0,1,0], [-1,1,0,0]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ([1,0,-1,0], [1,1,0,0], [0,0,0,1]),
    ]
    
    # Now apply this to 4x4x4
    # The 4x4 matrix is composed of four 2x2 blocks.
    # A_00 (top-left), A_01 (top-right), A_10 (bot-left), A_11 (bot-right)
    # Same for B and C.
    
    # We need to define 7 "Super-Products" for 4x4x4.
    # These are the same formulas, but A00 etc are now 2x2 blocks.
    # Each Super-Product is a 2x2x2 multiplication of the blocks.
    # So we iterate over the 7 super-products.
    # For each super-product, we iterate over the 7 sub-products of the 2x2x2.
    
    for sup_u, sup_v, sup_w in strassen_2x2:
        # sup_u are coeffs for A_00, A_01, A_10, A_11
        # sup_v are coeffs for B_00, B_01, B_10, B_11
        # sup_w are coeffs for C_00, C_01, C_10, C_11
        
        # The linear form for A in this super-product is:
        # L_A = sup_u[0] A_00 + sup_u[1] A_01 + sup_u[2] A_10 + sup_u[3] A_11
        # This is a 2x2 matrix (sum of 2x2 blocks).
        # Similarly for B.
        
        # The product P_sup = L_A * L_B is a 2x2 matrix.
        # This product is computed using Strassen 2x2x2 on the blocks?
        # No, L_A and L_B are 2x2 matrices.
        # We need to compute L_A * L_B.
        # We can use the schoolbook 4 multiplications or Strassen 7.
        # Since we are already in a recursive step, we should use Strassen again to minimize rank.
        # So for each sup-product, we invoke the 2x2x2 Strassen decomposition.
        
        # The "A" for this sub-problem is L_A (2x2).
        # The "B" for this sub-problem is L_B (2x2).
        # The "C" for this sub-problem is P_sup (2x2).
        
        # We need to map the local 2x2 indices to the global 4x4 indices.
        # Local A index (i,j) in L_A corresponds to global A index?
        # L_A = sum_b sup_u[b] A_b.
        # A_b is a 2x2 block.
        # Entry (i,j) of L_A is sum_b sup_u[b] A_b[i,j].
        # Global index of A_b[i,j]:
        # Block b: b=0 (00), 1 (01), 2 (10), 3 (11).
        # Block row br = b // 2, block col bc = b % 2.
        # Global row = br * 2 + i.
        # Global col = bc * 2 + j.
        
        for sub_u, sub_v, sub_w in strassen_2x2:
            # sub_u, sub_v, sub_w are coeffs for the 4 entries of the 2x2 matrices L_A, L_B, P_sup.
            # We need to expand this into scalar rank-1 terms.
            
            # For each of the 4 entries of L_A (i,j), we have a linear combination of global A entries.
            # coeff_A[i,j] = sum_b sup_u[b] * (entry (i,j) of A_b)
            # Entry (i,j) of A_b is at global index (br*2+i, bc*2+j).
            
            # We form the rank-1 term:
            # U_part: vector of coeffs for A.
            # V_part: vector of coeffs for B.
            # W_part: vector of coeffs for C.
            
            # Actually, we can just add the terms directly to the global U, V, W.
            # The term is:
            # (sum_{i,j} sub_u[i,j] * L_A[i,j]) * (sum_{k,l} sub_v[k,l] * L_B[k,l]) * (sum_{c,d} sub_w[c,d] * C_sup[c,d])
            
            # L_A[i,j] = sum_b sup_u[b] A_{b, i, j}
            # L_B[k,l] = sum_b sup_v[b] B_{b, k, l}
            # C_sup[c,d] contributes to C_{c,d} with weight sub_w[c,d] * sup_w[c,d]?
            # Wait, P_sup is the product.
            # C_global receives contribution from P_sup via sup_w.
            # C_sup[c,d] is the (c,d) entry of P_sup.
            # Global C index for block c (0..3) entry (c_row, c_col):
            # c = 0 (00), 1 (01), 2 (10), 3 (11).
            # Global row = (c//2)*2 + c_row.
            # Global col = (c%2)*2 + c_col.
            
            # So the weight for global C_{R,C} is:
            # sum_{c, c_row, c_col} sup_w[c] * sub_w[c_row, c_col] where R = (c//2)*2+c_row, C=(c%2)*2+c_col.
            
            # This is getting complex to implement generically.
            # Given the small size, we can just hardcode the 49 terms or generate them carefully.
            # Let's generate them.
            
            pass # Placeholder for logic

    # ... fringe terms ...
    
    return {"U": U, "V": V, "W": W}
