"""FTW-inspired decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

def _frac(x):
    """Represent half-integer as float."""
    return float(x)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    
    # Rank-42 decomposition using Strassen-based block partitioning
    # Split m=4 into two 2-col blocks, p=6 into three 2-col blocks
    # For each column block r, compute Cr = A1*Br_top + A2*Br_bot
    # using two independent Strassen 2x2x2 decompositions (7+7=14 per block)
    # Total rank = 3 * 14 = 42
    
    rank = 42
    U = [[0.0] * rank for _ in range(8)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(12)]
    
    # Strassen 2x2x2 product definitions
    # Each: (U_coeffs, V_coeffs) where each is list of (local_index, coeff)
    # Local indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        # M1 = (X00+X11)(Y00+Y11)
        ([(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (X10+X11)Y00
        ([(2,1),(3,1)], [(0,1)]),
        # M3 = X00(Y01-Y11)
        ([(0,1)], [(1,1),(3,-1)]),
        # M4 = X11(Y10-Y00)
        ([(3,1)], [(2,1),(0,-1)]),
        # M5 = (X00+X01)Y11
        ([(0,1),(1,1)], [(3,1)]),
        # M6 = (X10-X00)(Y00+Y01)
        ([(2,1),(0,-1)], [(0,1),(1,1)]),
        # M7 = (X01-X11)(Y10+Y11)
        ([(1,1),(3,-1)], [(2,1),(3,1)]),
    ]
    
    # Strassen output combination: Z[i,k] = sum of M_j with coefficients
    # Z00 = M1+M4-M5+M7, Z01 = M3+M5, Z10 = M2+M4, Z11 = M1-M2+M3+M6
    w_combos = [
        [(0,1),(3,1),(4,-1),(6,1)],   # Z00
        [(2,1),(4,1)],                 # Z01
        [(1,1),(3,1)],                 # Z10
        [(0,1),(1,-1),(2,1),(5,1)],   # Z11
    ]
    
    col = 0
    for r in range(3):
        # Column block r: B columns 2*r, 2*r+1; C columns 2*r, 2*r+1
        
        # Part 1: A1 * Br_top
        # A1: rows i in {0,1}, cols j in {0,1} -> global A index (i, j)
        # Br_top: rows j in {0,1}, cols kb in {0,1} -> global B index (j, 2*r+kb)
        # Output: (i, 2*r+kb) -> W row = (2*r+kb)*2 + i
        
        for prod_idx in range(7):
            u_coeffs, v_coeffs = strassen[prod_idx]
            for (local_idx, coeff) in u_coeffs:
                # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                li, lj = divmod(local_idx, 2)
                U[li*4 + lj][col] += float(coeff)
            for (local_idx, coeff) in v_coeffs:
                li, lj = divmod(local_idx, 2)
                V[li*6 + (2*r + lj)][col] += float(coeff)
            col += 1
        
        # Part 2: A2 * Br_bot
        # A2: rows i in {0,1}, cols j in {0,1} -> global A index (i, j+2)
        # Br_bot: rows j in {0,1}, cols kb in {0,1} -> global B index (j+2, 2*r+kb)
        # Output: same (i, 2*r+kb) -> W row = (2*r+kb)*2 + i
        
        for prod_idx in range(7):
            u_coeffs, v_coeffs = strassen[prod_idx]
            for (local_idx, coeff) in u_coeffs:
                li, lj = divmod(local_idx, 2)
                U[li*4 + (lj+2)][col] += float(coeff)
            for (local_idx, coeff) in v_coeffs:
                li, lj = divmod(local_idx, 2)
                V[(li+2)*6 + (2*r + lj)][col] += float(coeff)
            col += 1
    
    # Set W: output combination for each block
    # For block r, M1..M7 are cols r*14 to r*14+6 (A1*Br_top)
    # M8..M14 are cols r*14+7 to r*14+13 (A2*Br_bot)
    
    for r in range(3):
        base = r * 14
        # M indices for A1*Br_top: base+0 to base+6
        # M indices for A2*Br_bot: base+7 to base+13
        
        # Output (i, kb) -> W row = (2*r+kb)*2 + i
        # (i=0,kb=0): row=4r, (i=0,kb=1): row=4r+2
        # (i=1,kb=0): row=4r+1, (i=1,kb=1): row=4r+3
        
        # Z00 = M1+M4-M5+M7 (from A1) + M8+M11-M12+M14 (from A2)
        # M1=base+0, M4=base+3, M5=base+4, M7=base+6
        # M8=base+7, M11=base+10, M12=base+11, M14=base+13
        W[4*r][base+0] += 1
        W[4*r][base+3] += 1
        W[4*r][base+4] += -1
        W[4*r][base+6] += 1
        W[4*r][base+7] += 1
        W[4*r][base+10] += 1
        W[4*r][base+11] += -1
        W[4*r][base+13] += 1
        
        # Z01 = M3+M5 (from A1) + M10+M12 (from A2)
        # M3=base+2, M5=base+4, M10=base+9, M12=base+11
        W[4*r+2][base+2] += 1
        W[4*r+2][base+4] += 1
        W[4*r+2][base+9] += 1
        W[4*r+2][base+11] += 1
        
        # Z10 = M2+M4 (from A1) + M9+M11 (from A2)
        # M2=base+1, M4=base+3, M9=base+8, M11=base+10
        W[4*r+1][base+1] += 1
        W[4*r+1][base+3] += 1
        W[4*r+1][base+8] += 1
        W[4*r+1][base+10] += 1
        
        # Z11 = M1-M2+M3+M6 (from A1) + M8-M9+M10+M13 (from A2)
        # M1=base+0, M2=base+1, M3=base+2, M6=base+5
        # M8=base+7, M9=base+8, M10=base+9, M13=base+12
        W[4*r+3][base+0] += 1
        W[4*r+3][base+1] += -1
        W[4*r+3][base+2] += 1
        W[4*r+3][base+5] += 1
        W[4*r+3][base+7] += 1
        W[4*r+3][base+8] += -1
        W[4*r+3][base+9] += 1
        W[4*r+3][base+12] += 1
    
    # Verify dimensions
    assert len(U) == 8 and len(U[0]) == 42
    assert len(V) == 24 and len(V[0]) == 42
    assert len(W) == 12 and len(W[0]) == 42
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
