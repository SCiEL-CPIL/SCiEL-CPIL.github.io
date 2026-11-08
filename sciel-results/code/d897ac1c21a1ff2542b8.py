import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 145
    
    n = 4
    N = n * n  # 16
    
    # Build the 4x4 matrix multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent
    # A_idx = i*n + j, B_idx = j*n + k, C_idx = k*n + i (trace convention)
    T = np.zeros((N, N, N))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a_idx = i * n + j
                b_idx = j * n + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each of the 7 Strassen terms, define:
    # a_coeffs: dict {(row,col): coeff} for the 2x2 A block
    # b_coeffs: dict {(row,col): coeff} for the 2x2 B block
    # c_coeffs: list of (row, col, coeff) for the 2x2 C block
    
    strassen = [
        # s=0: P1
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': [(0,0, 1), (1,1, 1)]},
        # s=1: P2
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': [(1,0, 1), (1,1, -1)]},
        # s=2: P3
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': [(0,1, 1), (1,1, 1)]},
        # s=3: P4
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': [(0,0, 1), (1,0, 1)]},
        # s=4: P5
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': [(0,0, -1), (0,1, 1)]},
        # s=5: P6
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': [(1,1, 1)]},
        # s=6: P7
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': [(0,0, 1)]},
    ]
    
    rank = 49
    U = np.zeros((N, rank))
    V = np.zeros((N, rank))
    W = np.zeros((N, rank))
    
    # For 4x4, we have 2x2 blocks of 2x2 matrices
    # Block (br, bc) contains elements at global positions:
    # (2*br + er, 2*bc + ec) for er, ec in {0,1}
    
    # A_idx = i*4 + j where i is global row, j is global col
    # B_idx = j*4 + k where j is global row, k is global col  
    # C_idx = k*4 + i where k is global col, i is global row (trace convention)
    
    def a_idx(br, bc, er, ec):
        """Global index for A element in block (br,bc) at position (er,ec)"""
        i = 2*br + er
        j = 2*bc + ec
        return i * n + j
    
    def b_idx(br, bc, er, ec):
        """Global index for B element in block (br,bc) at position (er,ec)"""
        j = 2*br + er
        k = 2*bc + ec
        return j * n + k
    
    def c_idx(br, bc, er, ec):
        """Global index for C element in block (br,bc) at position (er,ec) using trace convention"""
        i = 2*br + er
        k = 2*bc + ec
        return k * n + i
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            st = strassen[s]
            
            # U contributions from A
            for (er, ec), coeff_a in st['a'].items():
                for br in range(2):
                    for bc in range(2):
                        # The outer Strassen term t determines which blocks to combine
                        # But wait - we need to think about this differently
                        pass
    
    # Actually, let me reconsider the approach.
    # 
    # The recursive Strassen works as follows:
    # At the top level, we have 4x4 = 2x2 blocks of 2x2 matrices.
    # We apply Strassen to the 2x2 block structure:
    #   P1 = (A00 + A11)(B00 + B11)
    #   P2 = (A10 + A11)B00
    #   etc.
    # Each Pi is a 2x2 matrix product, which we compute using Strassen again (7 scalar mults).
    #
    # So for each outer term t (0-6) and inner term s (0-6):
    # - The outer term determines which 2x2 blocks of A and B are combined
    # - The inner term determines which scalar elements within those blocks are combined
    #
    # For outer term t, the A-side combines certain blocks with certain coefficients.
    # For each such block (br, bc) with coefficient ca, and for the inner term s,
    # we combine elements within that block according to the inner Strassen formula.
    
    # Let me define the outer Strassen structure more explicitly.
    # Outer term t operates on the 2x2 block structure:
    # A blocks: A00, A01, A10, A11 (where Aij is the 2x2 block at position (i,j))
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    outer_strassen = [
        # t=0: P1 = (A00+A11)(B00+B11)
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': [(0,0, 1), (1,1, 1)]},
        # t=1: P2 = (A10+A11)B00
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': [(1,0, 1), (1,1, -1)]},
        # t=2: P3 = A00(B01-B11)
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': [(0,1, 1), (1,1, 1)]},
        # t=3: P4 = A11(B10-B00)
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': [(0,0, 1), (1,0, 1)]},
        # t=4: P5 = (A00+A01)B11
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': [(0,0, -1), (0,1, 1)]},
        # t=5: P6 = (A10-A00)(B00+B01)
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': [(1,1, 1)]},
        # t=6: P7 = (A01-A11)(B10+B11)
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': [(0,0, 1)]},
    ]
    
    # Inner Strassen for 2x2 scalar matrices (same structure)
    inner_strassen = [
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': [(0,0, 1), (1,1, 1)]},
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': [(1,0, 1), (1,1, -1)]},
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': [(0,1, 1), (1,1, 1)]},
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': [(0,0, 1), (1,0, 1)]},
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': [(0,0, -1), (0,1, 1)]},
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': [(1,1, 1)]},
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': [(0,0, 1)]},
    ]
    
    U = np.zeros((N, rank))
    V = np.zeros((N, rank))
    W = np.zeros((N, rank))
    
    for t in range(7):
        ot = outer_strassen[t]
        for s in range(7):
            r = t * 7 + s
            ist = inner_strassen[s]
            
            # U: combine A blocks according to outer, then elements within each block according to inner
            for (obr, obc), ocoeff in ot['a'].items():
                for (ier, iec), icoeff in ist['a'].items():
                    total = ocoeff * icoeff
                    if total == 0:
                        continue
                    i = 2*obr + ier
                    j = 2*obc + iec
                    U[i*n + j, r] += total
            
            # V: combine B blocks according to outer, then elements within each block according to inner
            for (obr, obc), ocoeff in ot['b'].items():
                for (ier, iec), icoeff in ist['b'].items():
                    total = ocoeff * icoeff
                    if total == 0:
                        continue
                    j = 2*obr + ier
                    k = 2*obc + iec
                    V[j*n + k, r] += total
            
            # W: combine C blocks according to outer, then elements within each block according to inner
            for (obr, obc, ocoeff) in ot['c']:
                for (ier, iec, icoeff) in ist['c']:
                    total = ocoeff * icoeff
                    if total == 0:
                        continue
                    i = 2*obr + ier
                    k = 2*obc + iec
                    W[k*n + i, r] += total
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
