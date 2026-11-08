import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 4x4 -> 7 products of 2x2, each 2x2 -> 7 scalar products.
    Output uses trace convention W[k*n+i, r] where k=col, i=row.
    """
    deadline = time.time() + 140
    n = 4
    N = n * n  # 16
    rank = 49
    
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]
    
    # Flat index: A[i,j] -> i*n + j = i*4 + j
    # Block (br,bc) entry (r,s) -> A[2*br+r, 2*bc+s] -> flat (2*br+r)*4 + (2*bc+s)
    
    def bidx(br, bc, r, s):
        return (2*br + r) * 4 + (2*bc + s)
    
    # Strassen 2x2 block-level: 7 products
    # Each product: X = sum of block refs with coeffs, Y = sum of block refs with coeffs
    # C_map: which output blocks get which coeff
    strassen_block = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M, C11+=M
        {'X': [((0,0),1), ((1,1),1)], 'Y': [((0,0),1), ((1,1),1)], 'C': [((0,0),1), ((1,1),1)]},
        # M2 = (A10+A11)B00 -> C10+=M, C11-=M
        {'X': [((1,0),1), ((1,1),1)], 'Y': [((0,0),1)], 'C': [((1,0),1), ((1,1),-1)]},
        # M3 = A00(B01-B11) -> C01+=M, C11+=M
        {'X': [((0,0),1)], 'Y': [((0,1),1), ((1,1),-1)], 'C': [((0,1),1), ((1,1),1)]},
        # M4 = A11(B10-B00) -> C00+=M, C10+=M
        {'X': [((1,1),1)], 'Y': [((1,0),1), ((0,0),-1)], 'C': [((0,0),1), ((1,0),1)]},
        # M5 = (A00+A01)B11 -> C00-=M, C01+=M
        {'X': [((0,0),1), ((0,1),1)], 'Y': [((1,1),1)], 'C': [((0,0),-1), ((0,1),1)]},
        # M6 = (A10-A00)(B00+B01) -> C11+=M
        {'X': [((1,0),1), ((0,0),-1)], 'Y': [((0,0),1), ((0,1),1)], 'C': [((1,1),1)]},
        # M7 = (A01-A11)(B10+B11) -> C00+=M
        {'X': [((0,1),1), ((1,1),-1)], 'Y': [((1,0),1), ((1,1),1)], 'C': [((0,0),1)]},
    ]
    
    # Strassen 2x2 scalar-level: 7 products
    # X_entries: (r, s, coeff) for 2x2 matrix entries
    # Y_entries: (r, s, coeff) for 2x2 matrix entries
    # Z_contrib: {(zr, zs): coeff} for 2x2 output entries
    strassen_scalar = [
        # z1 = (x00+x11)(y00+y11) -> Z00:1, Z11:1
        {'X': [(0,0,1), (1,1,1)], 'Y': [(0,0,1), (1,1,1)], 'Z': {(0,0):1, (1,1):1}},
        # z2 = (x10+x11)y00 -> Z10:1, Z11:-1
        {'X': [(1,0,1), (1,1,1)], 'Y': [(0,0,1)], 'Z': {(1,0):1, (1,1):-1}},
        # z3 = x00(y01-y11) -> Z01:1, Z11:1
        {'X': [(0,0,1)], 'Y': [(0,1,1), (1,1,-1)], 'Z': {(0,1):1, (1,1):1}},
        # z4 = x11(y10-y00) -> Z00:1, Z10:1
        {'X': [(1,1,1)], 'Y': [(1,0,1), (0,0,-1)], 'Z': {(0,0):1, (1,0):1}},
        # z5 = (x00+x01)y11 -> Z00:-1, Z01:1
        {'X': [(0,0,1), (0,1,1)], 'Y': [(1,1,1)], 'Z': {(0,0):-1, (0,1):1}},
        # z6 = (x10-x00)(y00+y01) -> Z11:1
        {'X': [(1,0,1), (0,0,-1)], 'Y': [(0,0,1), (0,1,1)], 'Z': {(1,1):1}},
        # z7 = (x01-x11)(y10+y11) -> Z00:1
        {'X': [(0,1,1), (1,1,-1)], 'Y': [(1,0,1), (1,1,1)], 'Z': {(0,0):1}},
    ]
    
    r_idx = 0
    for bp in strassen_block:
        # Build X_form and Y_form: 2x2 dicts mapping flat_idx -> coeff
        X_form = {(r,s): {} for r in range(2) for s in range(2)}
        Y_form = {(r,s): {} for r in range(2) for s in range(2)}
        
        for (br, bc), coeff in bp['X']:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    X_form[(r,s)][idx] = X_form[(r,s)].get(idx, 0) + coeff
        
        for (br, bc), coeff in bp['Y']:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    Y_form[(r,s)][idx] = Y_form[(r,s)].get(idx, 0) + coeff
        
        for sp in strassen_scalar:
            U_dict = {}
            V_dict = {}
            W_dict = {}
            
            # Build U from X_entries
            for (er, es, ecoeff) in sp['X']:
                for flat_idx, val in X_form[(er,es)].items():
                    U_dict[flat_idx] = U_dict.get(flat_idx, 0) + ecoeff * val
            
            # Build V from Y_entries
            for (er, es, ecoeff) in sp['Y']:
                for flat_idx, val in Y_form[(er,es)].items():
                    V_dict[flat_idx] = V_dict.get(flat_idx, 0) + ecoeff * val
            
            # Build W from Z_contrib and C_map
            for (zr, zs), zcoeff in sp['Z'].items():
                for (cb_r, cb_c), ccoeff in bp['C']:
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zs
                    # Trace convention: W[k*n+i] where k=col, i=row
                    w_idx = col * n + row
                    W_dict[w_idx] = W_dict.get(w_idx, 0) + zcoeff * ccoeff
            
            for flat_idx, val in U_dict.items():
                if val != 0:
                    U[flat_idx][r_idx] = int(val)
            for flat_idx, val in V_dict.items():
                if val != 0:
                    V[flat_idx][r_idx] = int(val)
            for flat_idx, val in W_dict.items():
                if val != 0:
                    W[flat_idx][r_idx] = int(val)
            
            r_idx += 1
            
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    return {"U": U, "V": V, "W": W}
