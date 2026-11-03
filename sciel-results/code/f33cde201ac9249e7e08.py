import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank 36 solution using Strassen for M_2,2,2 and extension for M_2,2,3.
    # The problem is decomposed into:
    # 1. M_2,4,2 part (cols 0,1 of C): Rank 14 (2 * M_2,2,2)
    # 2. M_2,4,3 part (cols 2,3,4 of C): Rank 22 (2 * M_2,2,3)
    # Total Rank 36.
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a Strassen decomposition for M_2,2,2
    # A indices: i in {0,1}, j in {j0, j0+1}
    # B indices: j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: k in {k0, k0+1}, i in {0,1}
    def add_strassen_222(j0, k0, offset):
        # Strassen products P1..P7
        # P1 = (A00+A11)(B00+B11) -> W[0,0] += 1, W[1,1] += 1? 
        # Let's map the 7 terms to the tensor equation.
        # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in the subblock.
        
        # Strassen coefficients for U, V, W
        # r=0: P1
        # U: (1, 1, 0, 0) for A00, A01, A10, A11? No, U is 2x2.
        # A = [[A00, A01], [A10, A11]]
        # P1 = (A00 + A11) * (B00 + B11)
        # U00=1, U11=1. V00=1, V11=1.
        # Contribution to C00: 1*1*W00. Contribution to C11: 1*1*W11.
        # We need W00=1, W11=1 for this term?
        # Let's use the standard Strassen identity:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We define 7 vectors (u_r, v_r, w_r)
        
        # r=0: P1 = (A00+A11)(B00+B11)
        # u: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00=1, A11=1
        # v: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00=1, B11=1
        # w: [1, 0, 0, 1] (C00, C01, C10, C11) -> C00=1, C11=1
        
        # r=1: P2 = (A10+A11)B00
        # u: [0, 0, 1, 1]
        # v: [1, 0, 0, 0]
        # w: [0, 0, 1, 0] (C10)
        
        # r=2: P3 = A00(B01-B11)
        # u: [1, 0, 0, 0]
        # v: [0, 1, 0, -1]
        # w: [0, 1, 0, 0] (C01)
        
        # r=3: P4 = A11(B10-B00)
        # u: [0, 0, 0, 1]
        # v: [0, 0, 1, -1]
        # w: [1, 0, 1, 0] (C00, C10)
        
        # r=4: P5 = (A00+A01)B11
        # u: [1, 1, 0, 0]
        # v: [0, 0, 0, 1]
        # w: [-1, 1, 0, 0] (C00, C01)
        
        # r=5: P6 = (A10-A00)(B00+B01)
        # u: [-1, 0, 1, 0]
        # v: [1, 1, 0, 0]
        # w: [0, 0, 0, 1] (C11)
        
        # r=6: P7 = (A01-A11)(B10+B11)
        # u: [0, 1, 0, -1]
        # v: [0, 0, 1, 1]
        # w: [1, 0, 0, 0] (C00)
        
        # Check C00: P1(1) + P4(1) - P5(1) + P7(1) = 1+1-1+1 = 2? No.
        # Strassen: C00 = P1 + P4 - P5 + P7.
        # My w vectors:
        # P1: w00=1
        # P4: w00=1
        # P5: w00=-1
        # P7: w00=1
        # Sum: 1+1-1+1 = 2.
        # Wait, Strassen formula:
        # P1 = (a+d)(e+h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P7 = (b-d)(g+h)
        # C00 = P1 + P4 - P5 + P7
        # Let's re-verify the coefficients.
        # P1: a e + a h + d e + d h
        # P4: d g - d e
        # P5: a h + b h
        # P7: b g + b h - d g - d h
        # Sum:
        # ae: 1
        # ah: 1 - 1 = 0
        # de: 1 - 1 = 0
        # dh: 1 - 1 = 0
        # dg: 1 - 1 = 0
        # bg: 1
        # bh: 1 - 1 = 0
        # ag: 0
        # Result: ae + bg. Correct.
        
        # So the w vectors are correct.
        
        # Map to global indices
        # i in {0, 1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # r=0
        U[j0, offset] = 1; U[j0+1, offset] = 1 # A00, A11? No, U is (i*m+j).
        # U index is i*m + j.
        # i=0, j=j0 -> 0*4 + j0
        # i=1, j=j0+1 -> 1*4 + j0+1
        
        # Let's define a helper to set U, V, W for a subblock
        def set_subblock(u_vals, v_vals, w_vals, i_offset, j_offset, k_offset, r_idx):
            # u_vals: 4 values for A00, A01, A10, A11
            # v_vals: 4 values for B00, B01, B10, B11
            # w_vals: 4 values for C00, C01, C10, C11
            
            # U
            U[i_offset*4 + j_offset, r_idx] += u_vals[0]
            U[i_offset*4 + j_offset+1, r_idx] += u_vals[1]
            U[(i_offset+1)*4 + j_offset, r_idx] += u_vals[2]
            U[(i_offset+1)*4 + j_offset+1, r_idx] += u_vals[3]
            
            # V
            V[j_offset*5 + k_offset, r_idx] += v_vals[0]
            V[j_offset*5 + k_offset+1, r_idx] += v_vals[1]
            V[(j_offset+1)*5 + k_offset, r_idx] += v_vals[2]
            V[(j_offset+1)*5 + k_offset+1, r_idx] += v_vals[3]
            
            # W
            W[k_offset*2 + i_offset, r_idx] += w_vals[0]
            W[(k_offset+1)*2 + i_offset, r_idx] += w_vals[1]
            W[k_offset*2 + (i_offset+1), r_idx] += w_vals[2]
            W[(k_offset+1)*2 + (i_offset+1), r_idx] += w_vals[3]

        # Strassen coefficients
        set_subblock([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], 0, j0, k0, offset)
        set_subblock([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], 0, j0, k0, offset+1)
        set_subblock([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0], 0, j0, k0, offset+2)
        set_subblock([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0], 0, j0, k0, offset+3)
        set_subblock([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0], 0, j0, k0, offset+4)
        set_subblock([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1], 0, j0, k0, offset+5)
        set_subblock([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0], 0, j0, k0, offset+6)

    # Helper to add trivial M_2,2,1 (4 mults) for extension
    def add_trivial_221(j0, k0, offset):
        # A00 B00 -> C00
        # A01 B00 -> C01? No, M_2,2,1 means B is 2x1.
        # Here we are extending M_2,2,2 to M_2,2,3 by adding a column.
        # The new column is k = k0+2.
        # We need A00 B_new0 -> C0_new
        # A01 B_new0 -> C0_new? No.
        # C_new0 = A00 B_new0 + A10 B_new1
        # C_new1 = A01 B_new0 + A11 B_new1
        
        # 4 terms:
        # A00 B_new0 -> C0_new
        # A10 B_new1 -> C0_new
        # A01 B_new0 -> C1_new
        # A11 B_new1 -> C1_new
        
        # U, V, W
        # r=0: A00 B_new0 -> C0_new
        set_subblock([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], 0, j0, k0+2, offset)
        # r=1: A10 B_new1 -> C0_new
        set_subblock([0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], 0, j0, k0+2, offset+1)
        # r=2: A01 B_new0 -> C1_new
        set_subblock([0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], 0, j0, k0+2, offset+2)
        # r=3: A11 B_new1 -> C1_new
        set_subblock([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], 0, j0, k0+2, offset+3)

    # Part 1: M_2,4,2 (j=0,1; k=0,1)
    # Left part (j=0,1): j0=0
    add_strassen_222(0, 0, 0)
    # Right part (j=2,3): j0=2
    add_strassen_222(2, 0, 7)
    
    # Part 2: M_2,4,3 (j=0,1; k=2,3,4) and (j=2,3; k=2,3,4)
    # Left part (j=0,1): j0=0
    # Strassen for k=2,3
    add_strassen_222(0, 2, 14)
    # Trivial for k=4
    add_trivial_221(0, 2, 21)
    
    # Right part (j=2,3): j0=2
    # Strassen for k=2,3
    add_strassen_222(2, 2, 25)
    # Trivial for k=4
    add_trivial_221(2, 2, 32)
    
    # Verify
    # The total rank is 14 + 7 + 4 + 7 + 4 = 36.
    # Wait, add_strassen_222 adds 7.
    # Part 1: 7 + 7 = 14.
    # Part 2 Left: 7 (Strassen) + 4 (Trivial) = 11.
    # Part 2 Right: 7 (Strassen) + 4 (Trivial) = 11.
    # Total 14 + 11 + 11 = 36.
    
    # Convert to lists
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
