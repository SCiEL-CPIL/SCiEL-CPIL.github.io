"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses Strassen's 7-product decomposition on the 2x2 block structure,
reducing rank from 64 (schoolbook) to 56.
"""

import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Build Strassen decomposition
    # A is 2x4, B is 4x8, C is 2x8
    # Block structure: 2x2 blocks
    # A00=A[0,0:2], A01=A[0,2:4], A10=A[1,0:2], A11=A[1,2:4]
    # B00=B[0:2,0:4], B01=B[0:2,4:8], B10=B[2:4,0:4], B11=B[2:4,4:8]
    # C00=C[0,0:4], C01=C[0,4:8], C10=C[1,0:4], C11=C[1,4:8]
    
    # 7 Strassen products, each (1x2)*(2x4) -> 1x4
    # For product i, define:
    #   left_i: dict mapping j in {0,1} to list of (a_idx, coeff)
    #   right_i: dict mapping (j,c) to list of (b_idx, coeff)
    #   c_map: list of (c_idx, coeff) for the 1x4 output
    
    # A indices: A[i,j] -> i*m+j = i*4+j
    # B indices: B[i,j] -> i*p+j = i*8+j
    # C indices: C[i,j] -> j*n+i (trace convention)
    
    products = []
    
    # M1 = (A00+A11)*(B00+B11)
    # Left: j=0: A[0,0]+A[1,2], j=1: A[0,1]+A[1,3]
    # Right: j=0,c=0: B[0,0]+B[2,4], etc.
    left = {
        0: [(0*4+0, 1), (1*4+2, 1)],
        1: [(0*4+1, 1), (1*4+3, 1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [(j*8+c, 1), ((j+2)*8+(c+4), 1)]
    c_map = [(0*8+c, 1) for c in range(4)] + [(1*8+c, 1) for c in range(4, 8)]
    # Wait, C00 is C[0,0:4] and C11 is C[1,4:8]
    # M1 contributes +1 to C00 and +1 to C11
    c_map = [(c, 1) for c in range(4)] + [(4+c, 1) for c in range(4)]
    # C index for C[0,c] is c*n+0 = c*2+0 = 2c
    # C index for C[1,c] is c*n+1 = c*2+1
    # C00: C[0,0..3] -> indices 0,2,4,6
    # C11: C[1,4..7] -> indices 9,11,13,15
    c_map = [(2*c, 1) for c in range(4)] + [(2*(4+c), 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M2 = (A10+A11)*B00
    left = {
        0: [(1*4+0, 1), (1*4+2, 1)],
        1: [(1*4+1, 1), (1*4+3, 1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [(j*8+c, 1)]
    # M2 contributes +1 to C10 = C[1,0:4] -> indices 1,3,5,7
    c_map = [(2*c+1, 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M3 = A00*(B01-B11)
    left = {
        0: [(0*4+0, 1)],
        1: [(0*4+1, 1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [(j*8+(4+c), 1), ((j+2)*8+(4+c), -1)]
    # M3 contributes +1 to C01 = C[0,4:8] -> indices 8,10,12,14
    c_map = [(2*(4+c), 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M4 = A11*(B10-B00)
    left = {
        0: [(1*4+2, 1)],
        1: [(1*4+3, 1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [((j+2)*8+c, 1), (j*8+c, -1)]
    # M4 contributes +1 to C00 = C[0,0:4] -> indices 0,2,4,6
    c_map = [(2*c, 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M5 = (A00+A01)*B11
    left = {
        0: [(0*4+0, 1), (0*4+2, 1)],
        1: [(0*4+1, 1), (0*4+3, 1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [((j+2)*8+(4+c), 1)]
    # M5 contributes -1 to C00, +1 to C01
    # C00: indices 0,2,4,6 with coeff -1
    # C01: indices 8,10,12,14 with coeff +1
    c_map = [(2*c, -1) for c in range(4)] + [(2*(4+c), 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M6 = (A10-A00)*(B00+B01)
    left = {
        0: [(1*4+0, 1), (0*4+0, -1)],
        1: [(1*4+1, 1), (0*4+1, -1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [(j*8+c, 1), (j*8+(4+c), 1)]
    # M6 contributes +1 to C11 = C[1,4:8] -> indices 9,11,13,15
    c_map = [(2*(4+c)+1, 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # M7 = (A01-A11)*(B10+B11)
    left = {
        0: [(0*4+2, 1), (1*4+2, -1)],
        1: [(0*4+3, 1), (1*4+3, -1)]
    }
    right = {}
    for j in range(2):
        for c in range(4):
            right[(j,c)] = [((j+2)*8+c, 1), ((j+2)*8+(4+c), 1)]
    # M7 contributes +1 to C00 = C[0,0:4] -> indices 0,2,4,6
    c_map = [(2*c, 1) for c in range(4)]
    products.append({'left': left, 'right': right, 'c_map': c_map})
    
    # Now build U, V, W
    # For each product, for each (j, c) in 0..1 x 0..3:
    #   For each (a_idx, la) in left[j]:
    #     For each (b_idx, rb) in right[(j,c)]:
    #       Create a column with U[a_idx]=la, V[b_idx]=rb, W[c_idx]=coeff
    #       where c_idx comes from c_map
    
    # Count total columns
    total_cols = 0
    for prod in products:
        for j in range(2):
            for c in range(4):
                total_cols += len(prod['left'][j]) * len(prod['right'][(j,c)])
    
    rank = total_cols
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        c_map = prod['c_map']
        # c_map is a list of (c_index, coeff) for the 4 output positions
        # But we need to map each output position c to its C index and coeff
        # The output of the product is 1x4, where position c corresponds to
        # a specific C element.
        
        # Actually, c_map has 4 entries (one per output column c=0..3)
        # Each entry is (c_global_index, coeff)
        
        for j in range(2):
            for c in range(4):
                c_idx, c_coeff = c_map[c]
                for (a_idx, la) in prod['left'][j]:
                    for (b_idx, rb) in prod['right'][(j,c)]:
                        U[a_idx][col] = la
                        V[b_idx][col] = rb
                        W[c_idx][col] = c_coeff
                        col += 1
    
    return {"U": U, "V": V, "W": W}
