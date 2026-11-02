import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # We aim for rank 50
    # Breakdown:
    # 4 blocks of 2x2x2 (Strassen, rank 7) -> 28
    # 1 block of 2x2x1 (Schoolbook, rank 4) -> 4
    # 1 block of 2x1x9 (Schoolbook, rank 18) -> 18
    # Total: 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx, col] = coeff_u
        V[v_idx, col] = coeff_v
        W[w_idx, col] = coeff_w
        col += 1

    # Helper to add a Strassen 2x2x2 block
    # i_base: row offset in A (0 or 1)
    # j_base: col offset in A (0 or 1)
    # k_base: row offset in B (0 or 1) -- wait, B is 3x9.
    # The Strassen block operates on A[i0:i2, j0:j2] and B[j0:j2, k0:k2]
    # Here A is 2x3, B is 3x9.
    # We are using A[0:2, 0:2] and B[0:2, k_base:k_base+2]
    def add_strassen_block(k_base):
        nonlocal col
        # A indices:
        # A00 = 0*m + 0 = 0
        # A01 = 0*m + 1 = 1
        # A10 = 1*m + 0 = 3
        # A11 = 1*m + 1 = 4
        
        # B indices:
        # B00 = 0*p + k_base
        # B01 = 0*p + k_base + 1
        # B10 = 1*p + k_base
        # B11 = 1*p + k_base + 1
        
        # W indices:
        # W00 = k_base*n + 0
        # W01 = k_base*n + 1
        # W10 = (k_base+1)*n + 0
        # W11 = (k_base+1)*n + 1
        
        # Strassen terms:
        # M0 = (A00 + A11)(B00 + B11)
        # U: A00 (+), A11 (+)
        # V: B00 (+), B11 (+)
        # W: W00 (+), W01 (+), W10 (+), W11 (+)
        # Wait, M0 is a single product.
        # We need to decompose (A00+A11)(B00+B11) into rank-1 terms?
        # No, the Strassen algorithm defines 7 intermediate multiplications.
        # Each intermediate multiplication is a product of two sums.
        # We must expand these sums into linear combinations in U and V,
        # and combine the results in W.
        
        # Actually, the tensor decomposition for Strassen is:
        # C = sum_{t=1}^7 U_t V_t W_t
        # Where U_t, V_t are vectors of coefficients for A and B.
        
        # Let's define the 7 Strassen coefficients for the block.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00)
        # M3 = (A00)(B01 - B11)
        # M4 = (A11)(B10 - B00)
        # M5 = (A00 + A01)(B11)
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # For each M_t, we add a rank-1 term to the global U, V, W.
        # The "U" part is the coefficients of A in the sum.
        # The "V" part is the coefficients of B in the sum.
        # The "W" part distributes the result M_t to the C entries.
        
        # M1:
        # A coeffs: A00:1, A11:1
        # B coeffs: B00:1, B11:1
        # C coeffs: C00:1, C01:0, C10:0, C11:1 (Wait, M1 contributes to C00 and C11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So for M1:
        # W00 += 1, W11 += 1
        
        def add_strassen_term(a_coeffs, b_coeffs, w_coeffs):
            nonlocal col
            for a_idx, a_val in a_coeffs.items():
                if a_val != 0:
                    U[a_idx, col] = a_val
            for b_idx, b_val in b_coeffs.items():
                if b_val != 0:
                    V[b_idx, col] = b_val
            for w_idx, w_val in w_coeffs.items():
                if w_val != 0:
                    W[w_idx, col] = w_val
            col += 1
            
        # Indices for this block
        A00 = 0
        A01 = 1
        A10 = 3
        A11 = 4
        
        B00 = k_base
        B01 = k_base + 1
        B10 = p + k_base
        B11 = p + k_base + 1
        
        W00 = k_base * n
        W01 = k_base * n + 1
        W10 = (k_base + 1) * n
        W11 = (k_base + 1) * n + 1
        
        # M1: (A00 + A11)(B00 + B11)
        add_strassen_term(
            {A00: 1, A11: 1},
            {B00: 1, B11: 1},
            {W00: 1, W11: 1}
        )
        
        # M2: (A10 + A11)(B00)
        add_strassen_term(
            {A10: 1, A11: 1},
            {B00: 1},
            {W10: 1, W11: -1}
        )
        
        # M3: (A00)(B01 - B11)
        add_strassen_term(
            {A00: 1},
            {B01: 1, B11: -1},
            {W01: 1, W11: 1}
        )
        
        # M4: (A11)(B10 - B00)
        add_strassen_term(
            {A11: 1},
            {B10: 1, B00: -1},
            {W00: 1, W10: 1}
        )
        
        # M5: (A00 + A01)(B11)
        add_strassen_term(
            {A00: 1, A01: 1},
            {B11: 1},
            {W00: -1, W01: 1}
        )
        
        # M6: (A10 - A00)(B00 + B01)
        add_strassen_term(
            {A10: 1, A00: -1},
            {B00: 1, B01: 1},
            {W00: -1, W11: 1}
        )
        
        # M7: (A01 - A11)(B10 + B11)
        add_strassen_term(
            {A01: 1, A11: -1},
            {B10: 1, B11: 1},
            {W00: -1, W11: 1}
        )
        
        return col

    # Apply Strassen to the four 2x2 blocks in B (cols 0-1, 2-3, 4-5, 6-7)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_strassen_block(6)
    
    # Apply Schoolbook to the 2x1 block in B (col 8)
    # A is 2x2 (cols 0,1), B is 2x1 (col 8)
    # C00 += A00*B00 + A01*B10
    # C01 += A00*B00 + A01*B10  -- Wait, C is 2x1? No, C is 2x9.
    # The block result is 2x1.
    # C_{0, 8} = A_{00}B_{0,8} + A_{01}B_{1,8}
    # C_{1, 8} = A_{10}B_{0,8} + A_{11}B_{1,8}
    
    # Terms:
    # 1. A00 * B00 (where B00 is B_{0,8})
    #    U: A00 (idx 0), V: B_{0,8} (idx 8), W: C_{0,8} (idx 8*2+0=16)
    add_term(0, 8, 16)
    
    # 2. A01 * B10 (where B10 is B_{1,8})
    #    U: A01 (idx 1), V: B_{1,8} (idx 9+8=17), W: C_{0,8} (idx 16)
    add_term(1, 17, 16)
    
    # 3. A10 * B00 (where B00 is B_{0,8})
    #    U: A10 (idx 3), V: B_{0,8} (idx 8), W: C_{1,8} (idx 8*2+1=17)
    add_term(3, 8, 17)
    
    # 4. A11 * B10 (where B10 is B_{1,8})
    #    U: A11 (idx 4), V: B_{1,8} (idx 17), W: C_{1,8} (idx 17)
    add_term(4, 17, 17)
    
    # Apply Schoolbook to the 2x1x9 part (from A col 2 and B row 2)
    # A is 2x1 (col 2), B is 1x9 (row 2)
    # C_{i, k} += A_{i,2} * B_{2,k}
    # i in 0,1. k in 0..8.
    # A indices:
    # A_{0,2} = 0*m + 2 = 2
    # A_{1,2} = 1*m + 2 = 5
    # B indices:
    # B_{2,k} = 2*p + k = 18 + k
    
    for i in range(2):
        for k in range(9):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx)
            
    if col != 50:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
