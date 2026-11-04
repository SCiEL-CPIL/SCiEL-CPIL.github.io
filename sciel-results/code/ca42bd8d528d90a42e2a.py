import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin
    
    n, m, p = 4, 4, 6
    
    # Outer Strassen Steps for 4x4 * 4x6
    # Blocks are 2x2 (A) and 2x3 (B, C)
    # Indices: 00, 01, 10, 11
    # A blocks: 00, 01, 10, 11 (2x2)
    # B blocks: 00, 01, 10, 11 (2x3)
    # C blocks: 00, 01, 10, 11 (2x3)
    
    # Note: B and C blocks are 2x3.
    # B00: rows 0-1, cols 0-2
    # B01: rows 0-1, cols 3-5
    # B10: rows 2-3, cols 0-2
    # B11: rows 2-3, cols 3-5
    
    # C00: rows 0-1, cols 0-2
    # C01: rows 0-1, cols 3-5
    # C10: rows 2-3, cols 0-2
    # C11: rows 2-3, cols 3-5
    
    # Strassen Steps:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
    
    outer_steps = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1, '11': 1})
    ]
    
    # Inner Terms for 2x2 * 2x3
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    
    inner_terms = [
        # Strassen for cols 0,1
        ({0:1, 3:1}, {0:1, 4:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1}),
        ({0:1}, {1:1, 4:-1}, {1:1}),
        ({3:1}, {3:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {3:1, 4:1}, {0:1, 3:1}),
        # Schoolbook for col 2
        ({0:1}, {2:1}, {2:1}),
        ({1:1}, {5:1}, {2:1}),
        ({2:1}, {2:1}, {5:1}),
        ({3:1}, {5:1}, {5:1})
    ]
    
    rank = 77
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Helper to get global index for A block
    def get_a_idx(block, i1, j1):
        # block is '00', '01', '10', '11'
        # i1, j1 are 0 or 1
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j
        
    # Helper to get global index for B block
    def get_b_idx(block, j1, k1):
        # block is '00', '01', '10', '11'
        # j1 is 0 or 1, k1 is 0, 1, 2
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k
        
    # Helper to get global index for C block
    def get_c_idx(block, i1, k1):
        # block is '00', '01', '10', '11'
        # i1 is 0 or 1, k1 is 0, 1, 2
        # C index is k * n + i
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i
        
    col = 0
    for s in range(7):
        a_coeffs, b_coeffs, c_coeffs = outer_steps[s]
        
        for t in range(11):
            a_local, b_local, c_local = inner_terms[t]
            
            # Build U vector
            # For each local A index (li, lj) with coeff ca
            # We need to sum over outer blocks ba in a_coeffs
            # Global A index = get_a_idx(ba, li, lj)
            # Contribution = ca * a_coeffs[ba]
            
            # But wait, the local A indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
            # So we need to map local index to (i1, j1).
            local_a_coords = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            local_b_coords = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2)}
            local_c_coords = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2)}
            
            for la, ca in a_local.items():
                i1, j1 = local_a_coords[la]
                for ba, outer_ca in a_coeffs.items():
                    global_a = get_a_idx(ba, i1, j1)
                    U[global_a, col] += ca * outer_ca
                    
            for lb, cb in b_local.items():
                j1, k1 = local_b_coords[lb]
                for bb, outer_cb in b_coeffs.items():
                    global_b = get_b_idx(bb, j1, k1)
                    V[global_b, col] += cb * outer_cb
                    
            for lc, cc in c_local.items():
                i1, k1 = local_c_coords[lc]
                for bc, outer_cc in c_coeffs.items():
                    global_c = get_c_idx(bc, i1, k1)
                    W[global_c, col] += cc * outer_cc
                    
            col += 1
            
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
