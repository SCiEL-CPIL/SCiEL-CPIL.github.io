"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time

# EVOLVE-BLOCK-START
def _winograd_2x2_ops():
    """
    Return the 5 bilinear operations for 2x2 matrix multiplication using Winograd.
    Each op is (u_coeffs, v_coeffs, w_coeff) where:
      u_coeffs: dict mapping (i,j) -> coeff for A[i,j]
      v_coeffs: dict mapping (j,k) -> coeff for B[j,k]
      w_coeff: coefficient for C[i,k]
    Actually, each bilinear op contributes to multiple C entries.
    Let me restructure: each of the 5 multiplications m0..m4 is a product of
    a linear form in A and a linear form in B, and the result C is a linear
    combination of these 5 products.
    
    So we return:
      a_factors: list of 5 dicts {(i,j): coeff}
      b_factors: list of 5 dicts {(j,k): coeff}
      c_combos: list of 5 dicts {(i,k): coeff}
    """
    # m0 = (a00+a01)*(b00+b10)
    # m1 = (a00-a01)*(b00-b10)
    # m2 = (a10+a11)*(b01+b11)
    # m3 = (a10-a11)*(b01-b11)
    # m4 = (a00+a10+a11)*(b00+b10+b11)
    #
    # c00 = m0 + m1 - m4
    # c01 = m2 - m1 + m4
    # c10 = m2 + m1 - m4
    # c11 = m0 - m2 + m4
    
    a_factors = [
        {(0,0): 1, (0,1): 1},           # m0
        {(0,0): 1, (0,1): -1},          # m1
        {(1,0): 1, (1,1): 1},           # m2
        {(1,0): 1, (1,1): -1},          # m3
        {(0,0): 1, (1,0): 1, (1,1): 1}  # m4
    ]
    b_factors = [
        {(0,0): 1, (1,0): 1},           # m0
        {(0,0): 1, (1,0): -1},          # m1
        {(0,1): 1, (1,1): 1},           # m2
        {(0,1): 1, (1,1): -1},          # m3
        {(0,0): 1, (1,0): 1, (1,1): 1}  # m4
    ]
    c_combos = [
        {(0,0): 1, (1,1): 1},           # m0 contributes to c00 and c11
        {(0,0): 1, (0,1): -1, (1,0): 1},# m1: c00+1, c01-1, c10+1
        {(0,1): 1, (1,0): 1},           # m2: c01+1, c10+1
        {},                              # m3: doesn't directly appear in C formula? 
        {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}  # m4: c00-1, c01+1, c10-1, c11+1
    ]
    # Wait, let me re-verify the Winograd 2x2 formula.
    # Standard Winograd for 2x2:
    # m0 = (a00+a01)(b00+b10)
    # m1 = (a00-a01)(b00-b10)  
    # m2 = (a10+a11)(b01+b11)
    # m3 = (a10-a11)(b01-b11)
    # m4 = (a00+a10+a11)(b00+b10+b11)
    #
    # c00 = m0 + m1 - m4
    # c01 = m2 - m1 + m4
    # c10 = m2 + m1 - m4
    # c11 = m0 - m2 + m4
    #
    # m3 is not used! That's a problem. Let me use the correct 5-multiply Winograd.
    # Actually the standard Winograd FT for 2x2 uses:
    # m0 = (a00+a01)(b00+b10)
    # m1 = (a00-a01)(b00-b10)
    # m2 = (a10+a11)(b01+b11)
    # m3 = (a10-a11)(b01-b11)
    # m4 = (a00+a10+a11)(b00+b10+b11)
    #
    # c00 = m0 + m1 - m4
    # c01 = m2 - m1 + m4
    # c10 = m2 + m1 - m4
    # c11 = m0 - m2 + m4
    #
    # Indeed m3 is not in the output. So we only need 4 multiplications?
    # No wait, that can't be right. Let me verify with a simple case.
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    # m0 = (1+0)(1+0) = 1
    # m1 = (1-0)(1-0) = 1
    # m2 = (0+1)(0+1) = 1
    # m3 = (0-1)(0-1) = 1
    # m4 = (1+0+1)(1+0+1) = 4
    # c00 = 1+1-4 = -2 ≠ 1. Something is wrong.
    
    # Let me look up the correct Winograd 2x2 formula.
    # The correct one is:
    # m0 = (a00+a01)(b00+b10)
    # m1 = (a00-a01)(b00-b10)
    # m2 = (a10+a11)(b01+b11)
    # m3 = (a10-a11)(b01-b11)
    # m4 = (a00+a10+a11)(b00+b10+b11)
    #
    # c00 = m0 + m1 - m4
    # c01 = m2 - m1 + m4
    # c10 = m2 + m1 - m4
    # c11 = m0 - m2 + m4
    
    # Hmm, let me try another known formula. Actually I think I'm confusing things.
    # Let me just use a verified 5-multiply 2x2 scheme.
    # 
    # Actually, for 2x2 matrix multiplication, the minimum rank is 7 (Strassen-like),
    # not 5. Winograd's algorithm for 2x2 uses 5 multiplications but with additions
    # in the input/output. Let me verify.
    #
    # Winograd FT for 2x2:
    # u0 = a00 + a01
    # u1 = a00 - a01
    # u2 = a10 + a11
    # u3 = a10 - a11
    # u4 = a00 + a10 + a11
    #
    # v0 = b00 + b10
    # v1 = b00 - b10
    # v2 = b01 + b11
    # v3 = b01 - b11
    # v4 = b00 + b10 + b11
    #
    # m0 = u0 * v0
    # m1 = u1 * v1
    # m2 = u2 * v2
    # m3 = u3 * v3
    # m4 = u4 * v4
    #
    # c00 = m0 + m1 - m4
    # c01 = m2 - m1 + m4
    # c10 = m2 + m1 - m4
    # c11 = m0 - m2 + m4
    
    # Test with identity:
    # A=I, B=I: a00=1,a01=0,a10=0,a11=1; b00=1,b01=0,b10=0,b11=1
    # u0=1, u1=1, u2=1, u3=-1, u4=2
    # v0=1, v1=1, v2=1, v3=-1, v4=2
    # m0=1, m1=1, m2=1, m3=1, m4=4
    # c00=1+1-4=-2. Still wrong!
    
    # I must be getting the formula wrong. Let me use a different approach.
    # For 2x2, let's just use schoolbook (4 mults) since finding a 5-mult
    # that actually works is tricky. Or use a known correct decomposition.
    
    # Actually, I recall now: the Winograd 2x2 algorithm does NOT reduce the
    # number of scalar multiplications below 4 for general 2x2. It's useful
    # for larger matrices via tensor decomposition. For 2x2 specifically,
    # the bilinear rank is 7 (not 5).
    
    # So let me abandon the Winograd 2x2 idea and instead look for other
    # structural reductions. 
    
    # Alternative: use a block structure that exploits the rectangular nature.
    # For 4x5 * 5x5, maybe there's a known lower rank decomposition.
    
    # Let me just return schoolbook for now and note that Winograd 2x2
    # doesn't give a reduction for 2x2 blocks.
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook baseline: rank = 100
    # Try to find a better decomposition.
    
    # For now, use schoolbook as the safe fallback.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
