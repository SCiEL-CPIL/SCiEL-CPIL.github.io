import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 135  # Safety margin before 150s kill
    
    # Correct Laderman rank-17 decomposition for 4x4x4
    # Split A, B, C into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]  
    # C = [[C00, C01], [C10, C11]]
    #
    # 17 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # M8 = (A00+A10)B10
    # M9 = (A01+A11)B01
    # M10 = (A00+A10)(B01+B11)
    # M11 = (A01+A11)(B00+B10)
    # M12 = A01*B10
    # M13 = A10*B01
    # M14 = (A00+A11)(B01+B10)
    # M15 = (A01+A10)(B00+B11)
    # M16 = (A00+A11)(B00-B11)
    # M17 = (A01+A10)(B01-B10)
    #
    # C00 = M1 - M4 + M6 - M7 + M16
    # C01 = M3 + M4 + M9 - M14
    # C10 = M2 + M5 - M8 + M13
    # C11 = M1 - M2 + M3 + M5 + M17
    
    # Actually, let me use a verified implementation.
    # The Laderman algorithm has a specific set of 17 rank-1 tensors.
    # Let me define them explicitly as (a_coeffs, b_coeffs, c_coeffs)
    
    # Each product k has:
    #   a_k: dict mapping (i,j) -> coeff for A[i,j]
    #   b_k: dict mapping (i,j) -> coeff for B[i,j]  
    #   c_k: dict mapping (i,j) -> coeff for C[i,j]
    
    products = []
    
    # M1: (A00+A11)(B00+B11)
    a = {(0,0):1, (3,3):1}
    b = {(0,0):1, (3,3):1}
    c = {(0,0):1, (3,3):1}
    products.append((a, b, c))
    
    # M2: (A10+A11)B00
    a = {(2,0):1, (3,3):1}
    b = {(0,0):1}
    c = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    # Wait, this isn't right. M2 contributes to C10 and C11 blocks.
    # Let me be more careful.
    
    # Actually, the cleanest approach is to use the known Laderman coefficients.
    # Let me use a different, verified approach: implement the rank-17 decomposition
    # using the block structure explicitly.
    
    # For 4x4x4 with Laderman, we can think of it as:
    # 7 Strassen products for the "main" 2x2 block multiplication,
    # plus 10 additional products to handle the cross terms.
    
    # Let me just verify: is rank 17 achievable? Yes, Laderman proved it.
    # Let me use a concrete implementation from literature.
    
    # Here's a verified rank-17 decomposition for 4x4x4:
    # Using the standard Laderman scheme.
    
    # I'll define the 17 products with their A, B, C coefficient vectors.
    # A is 4x4 = 16 entries, indexed 0-15 (row-major)
    # B is 4x4 = 16 entries, indexed 0-15
    # C is 4x4 = 16 entries, indexed 0-15
    
    def idx(i, j):
        return i * 4 + j
    
    # Product 1: (A00+A11)(B00+B11) -> C00 + C11
    a1 = {idx(0,0):1, idx(3,3):1}
    b1 = {idx(0,0):1, idx(3,3):1}
    c1 = {idx(0,0):1, idx(3,3):1}
    products.append((a1, b1, c1))
    
    # Product 2: (A10+A11)(B00) -> C10 + C11  
    a2 = {idx(2,0):1, idx(3,3):1}
    b2 = {idx(0,0):1}
    c2 = {idx(2,0):1, idx(2,1):1, idx(3,0):1, idx(3,1):1}
    products.append((a2, b2, c2))
    
    # This is getting complex. Let me use a simpler verification approach.
    # I'll test the current program's decomposition and if it works, use it.
    # If not, I'll use Strassen (rank 98) which is definitely correct.
    
    # Let me test the current program's 17-product decomposition:
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Test the products from the original program
    orig_products = []
    
    # P0
    a0 = {(0,0):1, (1,1):1, (2,2):1, (3,3):1}
    b0 = {(0,0):1, (1,1):1, (2,2):1, (3,3):1}
    c0 = {(i,j):1 for i in range(4) for j in range(4)}
    orig_products.append((a0, b0, c0))
    
    # P1
    a1 = {(0,1):1, (1,0):1, (2,3):1, (3,2):1}
    b1 = {(0,1):1, (1,0):1, (2,3):1, (3,2):1}
    c1 = {(0,1):1, (1,0):1, (2,3):1, (3,2):1}
    orig_products.append((a1, b1, c1))
    
    # P2
    a2 = {(0,2):1, (1,3):1, (2,0):1, (3,1):1}
    b2 = {(0,2):1, (1,3):1, (2,0):1, (3,1):1}
    c2 = {(0,2):1, (1,3):1, (2,0):1, (3,1):1}
    orig_products.append((a2, b2, c2))
    
    # P3
    a3 = {(0,3):1, (1,2):1, (2,1):1, (3,0):1}
    b3 = {(0,3):1, (1,2):1, (2,1):1, (3,0):1}
    c3 = {(0,3):1, (1,2):1, (2,1):1, (3,0):1}
    orig_products.append((a3, b3, c3))
    
    # P4
    a4 = {(0,0):1, (1,1):1}
    b4 = {(0,0):1, (1,1):1}
    c4 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    orig_products.append((a4, b4, c4))
    
    # P5
    a5 = {(0,0):1, (2,2):1}
    b5 = {(0,0):1, (2,2):1}
    c5 = {(0,0):1, (0,2):1, (2,0):1, (2,2):1}
    orig_products.append((a5, b5, c5))
    
    # P6
    a6 = {(0,0):1, (3,3):1}
    b6 = {(0,0):1, (3,3):1}
    c6 = {(0,0):1, (0,3):1, (3,0):1, (3,3):1}
    orig_products.append((a6, b6, c6))
    
    # P7
    a7 = {(1,1):1, (2,2):1}
    b7 = {(1,1):1, (2,2):1}
    c7 = {(1,1):1, (1,2):1, (2,1):1, (2,2):1}
    orig_products.append((a7, b7, c7))
    
    # P8
    a8 = {(1,1):1, (3,3):1}
    b8 = {(1,1):1, (3,3):1}
    c8 = {(1,1):1, (1,3):1, (3,1):1, (3,3):1}
    orig_products.append((a8, b8, c8))
    
    # P9
    a9 = {(2,2):1, (3,3):1}
    b9 = {(2,2):1, (3,3):1}
    c9 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    orig_products.append((a9, b9, c9))
    
    # P10
    a10 = {(0,1):1, (2,3):1}
    b10 = {(0,1):1, (2,3):1}
    c10 = {(0,1):1, (0,3):1, (2,1):1, (2,3):1}
    orig_products.append((a10, b10, c10))
    
    # P11
    a11 = {(0,2):1, (3,1):1}
    b11 = {(0,2):1, (3,1):1}
    c11 = {(0,2):1, (0,1):1, (3,2):1, (3,1):1}
    orig_products.append((a11, b11, c11))
    
    # P12
    a12 = {(0,3):1, (1,2):1}
    b12 = {(0,3):1, (1,2):1}
    c12 = {(0,3):1, (0,2):1, (1,3):1, (1,2):1}
    orig_products.append((a12, b12, c12))
    
    # P13
    a13 = {(1,0):1, (3,2):1}
    b13 = {(1,0):1, (3,2):1}
    c13 = {(1,0):1, (1,2):1, (3,0):1, (3,2):1}
    orig_products.append((a13, b13, c13))
    
    # P14
    a14 = {(1,3):1, (2,0):1}
    b14 = {(1,3):1, (2,0):1}
    c14 = {(1,3):1, (1,0):1, (2,3):1, (2,0):1}
    orig_products.append((a14, b14, c14))
    
    # P15
    a15 = {(2,1):1, (3,0):1}
    b15 = {(2,1):1, (3,0):1}
    c15 = {(2,1):1, (2,0):1, (3,1):1, (3,0):1}
    orig_products.append((a15, b15, c15))
    
    # P16
    a16 = {(0,1):1, (1,0):1}
    b16 = {(0,1):1, (1,0):1}
    c16 = {(0,1):1, (1,0):1}
    orig_products.append((a16, b16, c16))
    
    C_computed = np.zeros((4, 4))
    for k, (a_k, b_k, c_k) in enumerate(orig_products):
        prod_a = sum(coeff * A_test[i, j] for (i, j), coeff in a_k.items())
        prod_b = sum(coeff * B_test[i, j] for (i, j), coeff in b_k.items())
        for (i, j), coeff in c_k.items():
            C_computed[i, j] += coeff * prod_a * prod_b
    
    laderman_ok = np.allclose(C_computed, C_expected, atol=1e-10)
    
    if laderman_ok and time.time() - start_time < DEADLINE:
        # Use the rank-17 decomposition twice for rank 34
        rank = 34
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for k, (a_k, b_k, c_k) in enumerate(orig_products):
            # Left half (B columns 0-3)
            for (i, j), coeff in a_k.items():
                U[i * 4 + j, k] = coeff
            for (i, j), coeff in b_k.items():
                V[i * 8 + j, k] = coeff
            for (i, j), coeff in c_k.items():
                W[j * 4 + i, k] = coeff
            
            # Right half (B columns 4-7)
            for (i, j), coeff in a_k.items():
                U[i * 4 + j, 17 + k] = coeff
            for (i, j), coeff in b_k.items():
                V[i * 8 + (j + 4), 17 + k] = coeff
            for (i, j), coeff in c_k.items():
                W[(j + 4) * 4 + i, 17 + k] = coeff
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to Strassen rank 98
        return _strassen_decomposition()

def _strassen_decomposition():
    """Strassen-based decomposition with rank 98"""
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 rank-7 decomposition
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (A10+A11)(B00)
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3 = (A00)(B10-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4 = (A11)(B01-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5 = (A00+A01)(B11)
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6 = (A01-A10)(B00+B10)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (A10-A11)(B01+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block decomposition specs for 4x4x4 using Strassen
    A_specs = [
        [(1, 0), (1, 3)],   # M1: A00 + A11
        [(1, 2), (1, 3)],   # M2: A10 + A11
        [(1, 0)],           # M3: A00
        [(1, 3)],           # M4: A11
        [(1, 0), (1, 1)],   # M5: A00 + A01
        [(1, 2), (-1, 0)],  # M6: A10 - A00
        [(1, 1), (-1, 3)],  # M7: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # M1: B00 + B11
        [(1, 0)],           # M2: B00
        [(1, 1), (-1, 3)],  # M3: B10 - B11
        [(1, 2), (-1, 0)],  # M4: B01 - B00
        [(1, 3)],           # M5: B11
        [(1, 0), (1, 1)],   # M6: B00 + B10
        [(1, 2), (1, 3)],   # M7: B01 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],   # M1: C00 + C11
        [(1, 2), (-1, 3)],  # M2: C10 - C11
        [(1, 1), (1, 3)],   # M3: C01 + C11
        [(1, 0), (1, 2)],   # M4: C00 + C01
        [(-1, 0), (1, 1)],  # M5: -C00 + C10
        [(1, 3)],           # M6: C11
        [(1, 0)],           # M7: C00
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
