import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Local indices 0,1 correspond to 1,2
    # A indices: (i, j) -> i in 0,1; j in 0,1
    # B indices: (j, k) -> j in 0,1; k in 0,1
    # C indices: (i, k) -> i in 0,1; k in 0,1
    
    # Coefficients for U (A), V (B), W (C)
    # We need to express M_l = sum U_ij V_jk W_ki
    
    # Let's define the 7 terms explicitly
    # Term 1: (A11+A22)(B11+B22)
    # U: A11, A22. V: B11, B22. W: C11, C22, C12, C21?
    # Actually, we just add the rank-1 components.
    # M1 contributes to C11, C12, C21, C22.
    # C11 += M1
    # C12 += M1
    # C21 += M1
    # C22 += M1
    
    # This is getting complex to manage manually.
    # Better to generate the 7 rank-1 tensors for each block.
    
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal col
        # Local i: 0,1 -> Global i: i_offset + 0, i_offset + 1
        # Local j: 0,1 -> Global j: j_offset + 0, j_offset + 1
        # Local k: 0,1 -> Global k: k_offset + 0, k_offset + 1
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # Strassen 7 products
        # 1. (A11+A22)(B11+B22)
        # 2. (A21+A22)B11
        # 3. A11(B12-B22)
        # 4. A22(B21-B11)
        # 5. (A11+A12)B22
        # 6. (A21-A11)(B11+B12)
        # 7. (A12-A22)(B21+B22)
        
        # We will add 7 columns to U, V, W
        # For each product, we add a rank-1 term?
        # No, each product is a scalar multiplication of a matrix?
        # No, the product is a scalar $M_l$.
        # $M_l = (\sum U_{ij} A_{ij}) (\sum V_{jk} B_{jk})$.
        # Then $C_{ik} += W_{ki} M_l$.
        
        # So for each of the 7 terms, we define:
        # u_vec: coefficients for A
        # v_vec: coefficients for B
        # w_vec: coefficients for C
        
        # Term 1:
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C12=1, C21=1, C22=1
        
        # Term 2:
        # U: A21=1, A22=1
        # V: B11=1
        # W: C21=1, C22=-1
        
        # Term 3:
        # U: A11=1
        # V: B12=1, B22=-1
        # W: C12=1, C22=1
        
        # Term 4:
        # U: A22=1
        # V: B21=1, B11=-1
        # W: C11=1, C21=1
        
        # Term 5:
        # U: A11=1, A12=1
        # V: B22=1
        # W: C11=-1, C22=1
        
        # Term 6:
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C22=1
        
        # Term 7:
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=1, C12=-1
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: (i,j) pairs, v: (j,k) pairs, w: (k,i) pairs
            # Local indices 0,1
            # 1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}),
            # 2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # 3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            # 4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # 5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,1): 1}),
            # 6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # 7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): -1})
        ]
        
        for u_dict, v_dict, w_dict in terms:
            # Add to U
            for (li, lj), val in u_dict.items():
                gi = i_offset + li
                gj = j_offset + lj
                U[gi * m + gj][col] += val
            # Add to V
            for (lj, lk), val in v_dict.items():
                gj = j_offset + lj
                gk = k_offset + lk
                V[gj * p + gk][col] += val
            # Add to W
            for (lk, li), val in w_dict.items():
                gk = k_offset + lk
                gi = i_offset + li
                W[gk * n + gi][col] += val
            col += 1

    # 8 blocks
    # j_offsets: 0, 2
    # k_offsets: 0, 2, 4, 6
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(0, j_off, k_off)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
