import numpy as np
import time

def construct():
    deadline = time.time() + 115
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # We will build the decomposition with rank 78
    rank = 78
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for (i, j), c in u_dict.items():
            U[i * m + j, col] += c
        for (j, k), c in v_dict.items():
            V[j * p + k, col] += c
        for (k, i), c in w_dict.items():
            W[k * n + i, col] += c
        col += 1

    # Strassen 2x2 coefficients
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 - P6
    
    # We define the 7 steps of 2x2 Strassen
    # Each step is defined by u_coeffs, v_coeffs, w_coeffs
    # u_coeffs: {(0,0): 1, (1,1): 1} etc.
    
    s2_steps = [
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1} # Wait, w is for output Z
            # Z00 gets P1, Z11 gets P1.
            # So w should map (k,i) to coeff.
            # Z is 2x2. i in 0..1, k in 0..1.
            # Z00: i=0, k=0. Z11: i=1, k=1.
            # So w: {(0,0): 1, (1,1): 1}
        },
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(0,1): 1, (1,1): 1} # Z10 gets P2, Z11 gets -P2
            # Z10: i=1, k=0. Z11: i=1, k=1.
            # w: {(0,1): 1, (1,1): -1}
        },
        {
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(1,0): 1, (1,1): 1} # Z01 gets P3, Z11 gets P3
            # Z01: i=0, k=1. Z11: i=1, k=1.
            # w: {(1,0): 1, (1,1): 1}
        },
        {
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (0,1): 1} # Z00 gets P4, Z10 gets P4
            # Z00: i=0, k=0. Z10: i=1, k=0.
            # w: {(0,0): 1, (0,1): 1}
        },
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (1,0): 1} # Z00 gets -P5, Z01 gets P5
            # Z00: i=0, k=0. Z01: i=0, k=1.
            # w: {(0,0): -1, (1,0): 1}
        },
        {
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): -1} # Z11 gets -P6
            # Z11: i=1, k=1.
            # w: {(1,1): -1}
        },
        {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1} # Z00 gets P7, Z11 gets P7
            # Z00: i=0, k=0. Z11: i=1, k=1.
            # w: {(0,0): 1, (1,1): 1}
        }
    ]
    
    # Correcting w coefficients based on reconstruction formulas:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 - P6 + P7
    
    # P1: Z00 (+1), Z11 (+1) -> w: {(0,0): 1, (1,1): 1}
    # P2: Z10 (+1), Z11 (-1) -> w: {(0,1): 1, (1,1): -1}
    # P3: Z01 (+1), Z11 (+1) -> w: {(1,0): 1, (1,1): 1}
    # P4: Z00 (+1), Z10 (+1) -> w: {(0,0): 1, (0,1): 1}
    # P5: Z00 (-1), Z01 (+1) -> w: {(0,0): -1, (1,0): 1}
    # P6: Z11 (-1) -> w: {(1,1): -1}
    # P7: Z00 (+1), Z11 (+1) -> w: {(0,0): 1, (1,1): 1}
    
    # Note: In my s2_steps list above, I need to update the 'w' dicts.
    s2_steps[0]['w'] = {(0,0): 1, (1,1): 1}
    s2_steps[1]['w'] = {(0,1): 1, (1,1): -1}
    s2_steps[2]['w'] = {(1,0): 1, (1,1): 1}
    s2_steps[3]['w'] = {(0,0): 1, (0,1): 1}
    s2_steps[4]['w'] = {(0,0): -1, (1,0): 1}
    s2_steps[5]['w'] = {(1,1): -1}
    s2_steps[6]['w'] = {(0,0): 1, (1,1): 1}
    
    # Now, for the 4x4x4 part.
    # We have 7 Strassen steps M1..M7.
    # Each Ms is a 2x2 matrix product X*Y.
    # X and Y are linear combinations of 2x2 submatrices of A and B.
    
    # A submatrices: A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
    # B submatrices: B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define these as lists of (coeff, sub_index) for A and B.
    # sub_index: 0=A00, 1=A01, 2=A10, 3=A11
    # For B: 0=B00, 1=B01, 2=B10, 3=B11
    
    m4_steps = [
        {
            'A': [(1, 0), (1, 3)], # A00 + A11
            'B': [(1, 0), (1, 3)], # B00 + B11
            'C': [(1, 0), (1, 3)]  # C00 + C11 ? No, C is 2x2.
            # The output of M1 is a 2x2 matrix.
            # It contributes to C00 and C11.
            # C00 gets +1 * M1. C11 gets +1 * M1.
            # So the "w" part of the 4x4 Strassen step is:
            # C00: +1, C11: +1.
        },
        {
            'A': [(1, 2), (1, 3)], # A10 + A11
            'B': [(1, 0)],          # B00
            'C': [(1, 2), (-1, 3)]  # C10 +1, C11 -1
        },
        {
            'A': [(1, 0)],          # A00
            'B': [(1, 1), (-1, 3)], # B01 - B11
            'C': [(1, 1), (1, 3)]   # C01 +1, C11 +1
        },
        {
            'A': [(1, 3)],          # A11
            'B': [(1, 2), (-1, 0)], # B10 - B00
            'C': [(1, 0), (1, 2)]   # C00 +1, C10 +1
        },
        {
            'A': [(1, 0), (1, 1)],  # A00 + A01
            'B': [(1, 3)],          # B11
            'C': [(-1, 0), (1, 1)]  # C00 -1, C01 +1
        },
        {
            'A': [(1, 2), (-1, 0)], # A10 - A00
            'B': [(1, 0), (1, 1)],  # B00 + B01
            'C': [(-1, 3)]          # C11 -1
        },
        {
            'A': [(1, 1), (-1, 3)], # A01 - A11
            'B': [(1, 2), (1, 3)],  # B10 + B11
            'C': [(1, 0), (1, 3)]   # C00 +1, C11 +1
        }
    ]
    
    # For each of the 7 M4 steps, we have a 2x2 matrix product.
    # We expand this into 7 scalar products using s2_steps.
    
    for m4_idx, m4_step in enumerate(m4_steps):
        # Construct the linear combinations for X and Y
        # X is a 2x2 matrix.
        # X[i,j] = sum of coeff * A_sub[i,j]
        # A_sub is 2x2.
        # A00 is A[0:2, 0:2]. A01 is A[0:2, 2:4].
        # So A_sub[i,j] corresponds to A[i, j] or A[i, j+2] etc.
        
        # We need to map the "local" 2x2 indices to global 4x4 indices.
        # For X (from A):
        # Local (li, lj) -> Global (gi, gj)
        # If A_sub is A00: gi=li, gj=lj
        # If A_sub is A01: gi=li, gj=lj+2
        # If A_sub is A10: gi=li+2, gj=lj
        # If A_sub is A11: gi=li+2, gj=lj+2
        
        # Similarly for Y (from B):
        # Local (li, lj) -> Global (gi, gj)
        # B00: gi=li, gj=lj
        # B01: gi=li, gj=lj+2
        # B10: gi=li+2, gj=lj
        # B11: gi=li+2, gj=lj+2
        
        # And for Z (output C):
        # Local (li, lj) -> Global (gi, gj)
        # C00: gi=li, gj=lj
        # C01: gi=li, gj=lj+2
        # C10: gi=li+2, gj=lj
        # C11: gi=li+2, gj=lj+2
        
        # We iterate through the 7 scalar steps of the 2x2 Strassen
        for s2_idx, s2_step in enumerate(s2_steps):
            # s2_step has 'u', 'v', 'w' for the 2x2 matrices X, Y, Z.
            # We need to combine these with the m4_step definitions.
            
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Process u (X)
            for (li, lj), c_local in s2_step['u'].items():
                # c_local is the coeff in X[li, lj]
                # We need to find which A_sub contributes to X[li, lj]
                # X is a linear combination of A_subs.
                # m4_step['A'] is a list of (coeff, sub_idx)
                # The sub_idx determines the offset.
                # However, the structure of Strassen is that X is formed by summing A_subs.
                # So X[li, lj] = sum_{(c, sub_idx) in m4_step['A']} c * A_sub[li, lj]
                
                for c_sub, sub_idx in m4_step['A']:
                    # Determine global index
                    if sub_idx == 0: # A00
                        gi, gj = li, lj
                    elif sub_idx == 1: # A01
                        gi, gj = li, lj + 2
                    elif sub_idx == 2: # A10
                        gi, gj = li + 2, lj
                    elif sub_idx == 3: # A11
                        gi, gj = li + 2, lj + 2
                    
                    coeff = c_local * c_sub
                    key = (gi, gj)
                    if key in u_dict:
                        u_dict[key] += coeff
                    else:
                        u_dict[key] = coeff
            
            # Process v (Y)
            for (li, lj), c_local in s2_step['v'].items():
                for c_sub, sub_idx in m4_step['B']:
                    if sub_idx == 0: # B00
                        gi, gj = li, lj
                    elif sub_idx == 1: # B01
                        gi, gj = li, lj + 2
                    elif sub_idx == 2: # B10
                        gi, gj = li + 2, lj
                    elif sub_idx == 3: # B11
                        gi, gj = li + 2, lj + 2
                    
                    coeff = c_local * c_sub
                    key = (gi, gj)
                    if key in v_dict:
                        v_dict[key] += coeff
                    else:
                        v_dict[key] = coeff
            
            # Process w (Z)
            # w in s2_step is for Z[li, lj]
            # Z is a 2x2 matrix.
            # The output C is a linear combination of Z.
            # m4_step['C'] is a list of (coeff, sub_idx) for C.
            # C_sub is 2x2.
            # So C[gi, gj] gets contribution from Z[li, lj] with coeff c_local * c_sub.
            
            for (li, lj), c_local in s2_step['w'].items():
                for c_sub, sub_idx in m4_step['C']:
                    if sub_idx == 0: # C00
                        gi, gj = li, lj
                    elif sub_idx == 1: # C01
                        gi, gj = li, lj + 2
                    elif sub_idx == 2: # C10
                        gi, gj = li + 2, lj
                    elif sub_idx == 3: # C11
                        gi, gj = li + 2, lj + 2
                    
                    # Note: w_dict key is (k, i) where k is col, i is row.
                    # In C, row is i, col is k.
                    # So key is (gj, gi).
                    coeff = c_local * c_sub
                    key = (gj, gi)
                    if key in w_dict:
                        w_dict[key] += coeff
                    else:
                        w_dict[key] = coeff
            
            add_term(u_dict, v_dict, w_dict)
            
    # Now for the 4x4x2 part.
    # A_L is A[0:4, 0:4]. B_t2 is B[0:4, 4:6].
    # A_L = [A_L1 (4x2), A_L2 (4x2)]
    # B_t2 = [B_t2a (2x2); B_t2b (2x2)]
    
    # A_L1 is A[0:4, 0:2].
    # A_L2 is A[0:4, 2:4].
    # B_t2a is B[0:2, 4:6].
    # B_t2b is B[2:4, 4:6].
    
    # Term 1: A_L1 B_t2a
    # A_L1 (4x2) * B_t2a (2x2)
    # Split A_L1 into A_L1a (2x2) and A_L1b (2x2).
    # A_L1a = A[0:2, 0:2].
    # A_L1b = A[2:4, 0:2].
    
    # A_L1a B_t2a: 2x2 * 2x2.
    # Use s2_steps.
    
    # We need to map local indices to global.
    # For A_L1a: local (li, lj) -> global (li, lj).
    # For A_L1b: local (li, lj) -> global (li+2, lj).
    # For B_t2a: local (li, lj) -> global (li, lj+4).
    # For C: local (li, lj) -> global (li, lj+4).
    
    for s2_idx, s2_step in enumerate(s2_steps):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # u: A_L1a
        for (li, lj), c_local in s2_step['u'].items():
            u_dict[(li, lj)] = c_local
        
        # v: B_t2a
        for (li, lj), c_local in s2_step['v'].items():
            v_dict[(li, lj + 4)] = c_local
        
        # w: C
        for (li, lj), c_local in s2_step['w'].items():
            w_dict[(lj + 4, li)] = c_local
            
        add_term(u_dict, v_dict, w_dict)
        
    # Term 2: A_L1b B_t2a
    # A_L1b (2x2) * B_t2a (2x2)
    # A_L1b = A[2:4, 0:2].
    
    for s2_idx, s2_step in enumerate(s2_steps):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # u: A_L1b
        for (li, lj), c_local in s2_step['u'].items():
            u_dict[(li + 2, lj)] = c_local
        
        # v: B_t2a
        for (li, lj), c_local in s2_step['v'].items():
            v_dict[(li, lj + 4)] = c_local
        
        # w: C
        for (li, lj), c_local in s2_step['w'].items():
            w_dict[(lj + 4, li + 2)] = c_local
            
        add_term(u_dict, v_dict, w_dict)
        
    # Term 3: A_L2 B_t2b
    # A_L2 (4x2) * B_t2b (2x2)
    # Split A_L2 into A_L2a (2x2) and A_L2b (2x2).
    # A_L2a = A[0:2, 2:4].
    # A_L2b = A[2:4, 2:4].
    # B_t2b = B[2:4, 4:6].
    
    # Term 3a: A_L2a B_t2b
    for s2_idx, s2_step in enumerate(s2_steps):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # u: A_L2a
        for (li, lj), c_local in s2_step['u'].items():
            u_dict[(li, lj + 2)] = c_local
        
        # v: B_t2b
        for (li, lj), c_local in s2_step['v'].items():
            v_dict[(li + 2, lj + 4)] = c_local
        
        # w: C
        for (li, lj), c_local in s2_step['w'].items():
            w_dict[(lj + 4, li)] = c_local
            
        add_term(u_dict, v_dict, w_dict)
        
    # Term 3b: A_L2b B_t2b
    for s2_idx, s2_step in enumerate(s2_steps):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # u: A_L2b
        for (li, lj), c_local in s2_step['u'].items():
            u_dict[(li + 2, lj + 2)] = c_local
        
        # v: B_t2b
        for (li, lj), c_local in s2_step['v'].items():
            v_dict[(li + 2, lj + 4)] = c_local
        
        # w: C
        for (li, lj), c_local in s2_step['w'].items():
            w_dict[(lj + 4, li + 2)] = c_local
            
        add_term(u_dict, v_dict, w_dict)
        
    # Term 4: A_R B_bottom
    # A_R is A[0:4, 4]. (4x1)
    # B_bottom is B[4, 0:6]. (1x6)
    # C_part is A_R * B_bottom. (4x6)
    # This is an outer product.
    # C[i, k] = A[i, 4] * B[4, k].
    # Rank 1.
    
    u_dict = {}
    v_dict = {}
    w_dict = {}
    
    for i in range(4):
        u_dict[(i, 4)] = 1.0
    for k in range(6):
        v_dict[(4, k)] = 1.0
    for i in range(4):
        for k in range(6):
            w_dict[(k, i)] = 1.0
            
    add_term(u_dict, v_dict, w_dict)
    
    # Check if we used all rank
    # Should be 49 + 14 + 14 + 1 = 78
    
    return {"U": U, "V": V, "W": W}
