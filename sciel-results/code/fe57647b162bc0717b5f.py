"""Matrix multiplication 2,2,16 — rank-minimal exact tensor decomposition.

Uses a Strassen-based construction: group the 16 columns of the right matrix
into 8 blocks of 2. Each 2x2 block multiplication uses rank 7 (Strassen),
giving total rank = 8 * 7 = 56.
"""
import time
import sys

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    
    For A (2x2) * B (2x2) = C:
      A = [[a, b], [c, d]]
      B = [[e, f], [g, h]]
      
    Strassen:
      m1 = (a+d)(e+h)
      m2 = (c+d)e
      m3 = a(f-h)
      m4 = d(g-e)
      m5 = (a+b)h
      m6 = (c-a)(e+f)
      m7 = (b-d)(g+h)
      
      C[0,0] = m1 + m4 - m5 + m7
      C[0,1] = m3 + m5
      C[1,0] = m2 + m4
      C[1,1] = m1 - m2 + m3 + m6
    """
    # Each term: (U_rows, V_cols, W_rows) where
    # U rows are indices into A (2x2 flattened row-major: 0=a,1=b,2=c,3=d)
    # V cols are indices into B (2x2 flattened row-major: 0=e,1=f,2=g,3=h)
    # W rows are indices into C (2x2 flattened row-major: 0=C00,1=C01,2=C10,3=C11)
    #
    # We represent each of the 7 terms as a list of (coeff, u_idx, v_idx) for U/V
    # and (coeff, w_idx) for W, but actually we need the tensor structure.
    #
    # The multiplication tensor for 2x2x2 has entries T[a,b,c] where:
    #   A index a in {0,1,2,3} (row-major: a=0->A[0,0], a=1->A[0,1], a=2->A[1,0], a=3->A[1,1])
    #   B index b in {0,1,2,3} (row-major: b=0->B[0,0], b=1->B[0,1], b=2->B[1,0], b=3->B[1,1])
    #   C index c in {0,1,2,3} (row-major: c=0->C[0,0], c=1->C[0,1], c=2->C[1,0], c=3->C[1,1])
    #
    # T[a,b,c] = 1 if C[c] depends on A[a]*B[b]
    #
    # C[0,0] = a*e + b*g  => T[0,0,0]=1, T[1,2,0]=1
    # C[0,1] = a*f + b*h  => T[0,1,1]=1, T[1,3,1]=1
    # C[1,0] = c*e + d*g  => T[2,0,2]=1, T[3,2,2]=1
    # C[1,1] = c*f + d*h  => T[2,1,3]=1, T[3,3,3]=1
    #
    # Strassen decomposition: rank 7 tensors (u_term, v_term, w_term)
    # where u_term is a vector in R^4 (A-space), v_term in R^4 (B-space), w_term in R^4 (C-space)
    # and sum over terms of u_term[a]*v_term[b]*w_term[c] = T[a,b,c]
    
    # m1 = (a+d)(e+h): u=(1,0,0,1), v=(1,0,0,1), contributes to C via m1
    # m2 = (c+d)e: u=(0,0,1,1), v=(1,0,0,0)
    # m3 = a(f-h): u=(1,0,0,0), v=(0,1,0,-1)
    # m4 = d(g-e): u=(0,0,0,1), v=(-1,0,1,0)
    # m5 = (a+b)h: u=(1,1,0,0), v=(0,0,0,1)
    # m6 = (c-a)(e+f): u=(-1,0,1,0), v=(1,1,0,0)
    # m7 = (b-d)(g+h): u=(0,1,0,-1), v=(0,0,1,1)
    
    # C contributions:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # w vectors (in C-space, index 0=C00, 1=C01, 2=C10, 3=C11):
    # m1: w = (1, 0, 0, 1)
    # m2: w = (0, 0, 1, -1)
    # m3: w = (0, 1, 0, 1)
    # m4: w = (1, 0, 1, 0)
    # m5: w = (-1, 1, 0, 0)
    # m6: w = (0, 0, 0, 1)
    # m7: w = (1, 0, 0, 0)
    
    terms = [
        # (u_vec, v_vec, w_vec)
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # m1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),  # m2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),  # m3
        ((0,0,0,1), (-1,0,1,0), (1,0,1,0)),  # m4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),  # m5
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),  # m6
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),  # m7
    ]
    return terms


def construct():
    """Construct rank-56 decomposition of 2x2 x 2x16 matrix multiplication.
    
    U: shape (n*m, rank) = (4, 56)
    V: shape (m*p, rank) = (32, 56)
    W: shape (p*n, rank) = (32, 56)
    
    Convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Number of 2-column blocks
    num_blocks = p // 2  # 8
    assert num_blocks * 2 == p
    
    strassen_terms = _strassen_2x2()
    num_strassen = len(strassen_terms)  # 7
    
    rank = num_blocks * num_strassen  # 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]    # (4, 56)
    V = [[0] * rank for _ in range(m * p)]    # (32, 56)
    W = [[0] * rank for _ in range(p * n)]    # (32, 56)
    
    for block in range(num_blocks):
        for s in range(num_strassen):
            col = block * num_strassen + s
            u_vec, v_vec, w_vec = strassen_terms[s]
            
            # A is 2x2, flattened row-major: index a -> row a//2, col a%2
            # U index: i*m + j where i is A-row, j is A-col
            # A[a] where a = i*2 + j (row-major 2x2)
            for a in range(4):
                if u_vec[a] != 0:
                    i = a // 2
                    j = a % 2
                    U[i * m + j][col] = u_vec[a]
            
            # B for this block is 2x2: rows 0,1 of B; columns 2*block, 2*block+1
            # V index: j*p + k where j is B-row, k is B-col (global)
            # B local index b -> local col b%2, row b//2
            # Global col = 2*block + (b % 2)
            for b in range(4):
                if v_vec[b] != 0:
                    j = b // 2  # B row
                    k_local = b % 2  # B local col
                    k_global = 2 * block + k_local
                    V[j * p + k_global][col] = v_vec[b]
            
            # C for this block is 2x2: rows 0,1; cols 2*block, 2*block+1
            # W index: k*n + i where k is C-col (global), i is C-row
            # C local index c -> local col c%2, row c//2
            # Global col = 2*block + (c % 2)
            for c in range(4):
                if w_vec[c] != 0:
                    i = c // 2  # C row
                    k_local = c % 2  # C local col
                    k_global = 2 * block + k_local
                    W[k_global * n + i][col] = w_vec[c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
