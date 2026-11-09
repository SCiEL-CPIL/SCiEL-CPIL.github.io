import numpy as np

def construct():
    n, m, p = 2, 5, 5
    # We will build the matrices by accumulating contributions from each block
    # The total rank will be the sum of the ranks of the individual block decompositions
    # because the supports (indices in U, V, W) for different blocks are disjoint
    # in terms of the "middle" index j?
    # No, U depends on i,j. V depends on j,k. W depends on k,i.
    # Different blocks use different j ranges or k ranges or i ranges?
    # A0 uses j=0,1. A1 uses j=2,3. A2 uses j=4.
    # B00 uses j=0,1, k=0,1. B01 uses j=0,1, k=2,3.
    # So for A0, we have B00 (j=0,1) and B01 (j=0,1).
    # The U indices for A0 are i*5+0, i*5+1.
    # The V indices for B00 are 0*5+0, 0*5+1, 1*5+0, 1*5+1.
    # The V indices for B01 are 0*5+2, 0*5+3, 1*5+2, 1*5+3.
    # Since the V indices are disjoint for B00 and B01, the decompositions can be independent.
    # Similarly, W indices for C00 are k*2+i with k=0,1.
    # W indices for C01 are k*2+i with k=2,3.
    # Disjoint.
    
    # So we can just sum up the ranks.
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal rank_total
        U_list.append([0] * rank_total + [coeff_u])
        V_list.append([0] * rank_total + [coeff_v])
        W_list.append([0] * rank_total + [coeff_w])
        rank_total += 1

    # This approach is inefficient for building large matrices.
    # Better to determine total rank first, then fill.
    
    # Let's calculate the total rank first.
    # Blocks:
    # A0 (2x2) * B00 (2x2) -> Strassen 7
    # A0 (2x2) * B01 (2x2) -> Strassen 7
    # A0 (2x2) * B02 (2x1) -> Rank 4
    # A1 (2x2) * B10 (2x2) -> Strassen 7
    # A1 (2x2) * B11 (2x2) -> Strassen 7
    # A1 (2x2) * B12 (2x1) -> Rank 4
    # A2 (2x1) * B20 (1x2) -> Rank 4
    # A2 (2x1) * B21 (1x2) -> Rank 4
    # A2 (2x1) * B22 (1x1) -> Rank 2
    
    total_rank = 7 + 7 + 4 + 7 + 7 + 4 + 4 + 4 + 2
    # 14 + 4 + 14 + 4 + 8 + 2 = 46
    
    U_np = np.zeros((n * m, total_rank), dtype=float)
    V_np = np.zeros((m * p, total_rank), dtype=float)
    W_np = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset: global i start
        # j_offset: global j start
        # k_offset: global k start
        
        # M1 = (A00+A11)(B00+B11)
        # A terms: A00 (i_off, j_off), A11 (i_off+1, j_off+1)
        # B terms: B00 (j_off, k_off), B11 (j_off+1, k_off+1)
        
        # We need to add 7 terms.
        # Term 1: (A00+A11) * (B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11 contributions
        
        # It is easier to just list the 7 multiplications and their contributions to C.
        
        # M1
        # U: A00 (+1), A11 (+1)
        # V: B00 (+1), B11 (+1)
        # W: C00 (+1), C01 (+1), C10 (+1), C11 (+1) -- Wait, M1 contributes to all 4?
        # C00 = M1 + ...
        # C01 = M3 + M1
        # C10 = M2 + M5
        # C11 = M1 - M2 + ...
        # So M1 contributes +1 to C00, +1 to C01, 0 to C10, +1 to C11?
        # Let's check standard Strassen again.
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00, C01, C11.
        
        # Let's define the 7 multipliers and their U, V, W coefficients.
        
        # M1: (A00+A11)(B00+B11)
        # U: A00:1, A11:1
        # V: B00:1, B11:1
        # W: C00:1, C01:1, C11:1
        
        # M2: (A10+A11)B00
        # U: A10:1, A11:1
        # V: B00:1
        # W: C10:1, C11:-1
        
        # M3: A00(B10-B11)
        # U: A00:1
        # V: B10:1, B11:-1
        # W: C01:1, C11:1
        
        # M4: A11(B01-B00)
        # U: A11:1
        # V: B01:1, B00:-1
        # W: C00:1
        
        # M5: (A00+A01)B11
        # U: A00:1, A01:1
        # V: B11:1
        # W: C00:-1, C10:1
        
        # M6: (A10-A00)(B00+B10)
        # U: A10:1, A00:-1
        # V: B00:1, B10:1
        # W: C11:1
        
        # M7: (A01-A11)(B01+B11)
        # U: A01:1, A11:-1
        # V: B01:1, B11:1
        # W: C00:1, C11:1
        
        # Now map local indices to global.
        # Local A indices: (0,0)->A00, (0,1)->A01, (1,0)->A10, (1,1)->A11
        # Global U idx: (i_offset + li) * m + (j_offset + lj)
        
        # Local B indices: (0,0)->B00, (0,1)->B01, (1,0)->B10, (1,1)->B11
        # Global V idx: (j_offset + lj) * p + (k_offset + lk)
        
        # Local C indices: (0,0)->C00, (0,1)->C01, (1,0)->C10, (1,1)->C11
        # Global W idx: (k_offset + lk) * n + (i_offset + li)
        
        # We iterate through the 7 multipliers.
        
        # M1
        U_np[i_offset*m + j_offset, col] += 1
        U_np[(i_offset+1)*m + (j_offset+1), col] += 1
        V_np[(j_offset)*p + k_offset, col] += 1
        V_np[(j_offset+1)*p + (k_offset+1), col] += 1
        W_np[(k_offset)*n + i_offset, col] += 1 # C00
        W_np[(k_offset+1)*n + i_offset, col] += 1 # C01
        W_np[(k_offset+1)*n + (i_offset+1), col] += 1 # C11
        col += 1
        
        # M2
        U_np[(i_offset+1)*m + j_offset, col] += 1 # A10
        U_np[(i_offset+1)*m + (j_offset+1), col] += 1 # A11
        V_np[(j_offset)*p + k_offset, col] += 1 # B00
        W_np[(k_offset)*n + (i_offset+1), col] += 1 # C10
        W_np[(k_offset+1)*n + (i_offset+1), col] += -1 # C11
        col += 1
        
        # M3
        U_np[i_offset*m + j_offset, col] += 1 # A00
        V_np[(j_offset+1)*p + k_offset, col] += 1 # B10
        V_np[(j_offset+1)*p + (k_offset+1), col] += -1 # B11
        W_np[(k_offset+1)*n + i_offset, col] += 1 # C01
        W_np[(k_offset+1)*n + (i_offset+1), col] += 1 # C11
        col += 1
        
        # M4
        U_np[(i_offset+1)*m + (j_offset+1), col] += 1 # A11
        V_np[(j_offset)*p + (k_offset+1), col] += 1 # B01
        V_np[(j_offset)*p + k_offset, col] += -1 # B00
        W_np[(k_offset)*n + i_offset, col] += 1 # C00
        col += 1
        
        # M5
        U_np[i_offset*m + j_offset, col] += 1 # A00
        U_np[i_offset*m + (j_offset+1), col] += 1 # A01
        V_np[(j_offset+1)*p + (k_offset+1), col] += 1 # B11
        W_np[(k_offset)*n + i_offset, col] += -1 # C00
        W_np[(k_offset)*n + (i_offset+1), col] += 1 # C10
        col += 1
        
        # M6
        U_np[(i_offset+1)*m + j_offset, col] += 1 # A10
        U_np[i_offset*m + j_offset, col] += -1 # A00
        V_np[(j_offset)*p + k_offset, col] += 1 # B00
        V_np[(j_offset+1)*p + k_offset, col] += 1 # B10
        W_np[(k_offset+1)*n + (i_offset+1), col] += 1 # C11
        col += 1
        
        # M7
        U_np[i_offset*m + (j_offset+1), col] += 1 # A01
        U_np[(i_offset+1)*m + (j_offset+1), col] += -1 # A11
        V_np[(j_offset)*p + (k_offset+1), col] += 1 # B01
        V_np[(j_offset+1)*p + (k_offset+1), col] += 1 # B11
        W_np[(k_offset)*n + i_offset, col] += 1 # C00
        W_np[(k_offset+1)*n + (i_offset+1), col] += 1 # C11
        col += 1

    # Helper to add 2x1x2 (A is 2x1, B is 1x2)
    # A: i in [0,1], j=0
    # B: j=0, k in [0,1]
    # C: i in [0,1], k in [0,1]
    # Terms: A0B0, A0B1, A1B0, A1B1
    def add_2x1x2(i_offset, j_offset, k_offset):
        nonlocal col
        # A0 B0
        U_np[i_offset*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + i_offset, col] = 1
        col += 1
        # A0 B1
        U_np[i_offset*m + j_offset, col] = 1
        V_np[j_offset*p + (k_offset+1), col] = 1
        W_np[(k_offset+1)*n + i_offset, col] = 1
        col += 1
        # A1 B0
        U_np[(i_offset+1)*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + (i_offset+1), col] = 1
        col += 1
        # A1 B1
        U_np[(i_offset+1)*m + j_offset, col] = 1
        V_np[j_offset*p + (k_offset+1), col] = 1
        W_np[(k_offset+1)*n + (i_offset+1), col] = 1
        col += 1

    # Helper to add 2x2x1 (A is 2x2, B is 2x1)
    # A: i in [0,1], j in [0,1]
    # B: j in [0,1], k=0
    # C: i in [0,1], k=0
    # Terms: A00B0, A01B1, A10B0, A11B1
    def add_2x2x1(i_offset, j_offset, k_offset):
        nonlocal col
        # A00 B0
        U_np[i_offset*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + i_offset, col] = 1
        col += 1
        # A01 B1
        U_np[i_offset*m + (j_offset+1), col] = 1
        V_np[(j_offset+1)*p + k_offset, col] = 1
        W_np[k_offset*n + i_offset, col] = 1
        col += 1
        # A10 B0
        U_np[(i_offset+1)*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + (i_offset+1), col] = 1
        col += 1
        # A11 B1
        U_np[(i_offset+1)*m + (j_offset+1), col] = 1
        V_np[(j_offset+1)*p + k_offset, col] = 1
        W_np[k_offset*n + (i_offset+1), col] = 1
        col += 1

    # Helper to add 2x1x1 (A is 2x1, B is 1x1)
    # A: i in [0,1], j=0
    # B: j=0, k=0
    # C: i in [0,1], k=0
    # Terms: A0B0, A1B0
    def add_2x1x1(i_offset, j_offset, k_offset):
        nonlocal col
        # A0 B0
        U_np[i_offset*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + i_offset, col] = 1
        col += 1
        # A1 B0
        U_np[(i_offset+1)*m + j_offset, col] = 1
        V_np[j_offset*p + k_offset, col] = 1
        W_np[k_offset*n + (i_offset+1), col] = 1
        col += 1

    # Call for each block
    # A0 (i=0, j=0,1)
    # B00 (j=0,1, k=0,1)
    add_strassen(0, 0, 0)
    
    # B01 (j=0,1, k=2,3)
    add_strassen(0, 0, 2)
    
    # B02 (j=0,1, k=4)
    add_2x2x1(0, 0, 4)
    
    # A1 (i=0, j=2,3)
    # B10 (j=2,3, k=0,1)
    add_strassen(0, 2, 0)
    
    # B11 (j=2,3, k=2,3)
    add_strassen(0, 2, 2)
    
    # B12 (j=2,3, k=4)
    add_2x2x1(0, 2, 4)
    
    # A2 (i=0, j=4)
    # B20 (j=4, k=0,1)
    add_2x1x2(0, 4, 0)
    
    # B21 (j=4, k=2,3)
    add_2x1x2(0, 4, 2)
    
    # B22 (j=4, k=4)
    add_2x1x1(0, 4, 4)

    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
