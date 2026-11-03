import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # We'll build a tensor decomposition for 3x4 * 4x4 -> 3x4
    # Strategy: Split A (3x4) into rows. 
    # Rows 0,1 of A form a 2x4 matrix A_top. Row 2 is A_bottom (1x4).
    # B is 4x4.
    # C_top = A_top @ B (2x4), C_bottom = A_bottom @ B (1x4).
    
    # For A_top (2x4) @ B (4x4):
    # Split B into 2x2 blocks: B = [[B00, B01], [B10, B11]] where each is 2x2.
    # A_top = [A_L, A_R] where A_L is 2x2, A_R is 2x2.
    # C_top = [A_L, A_R] @ [[B00, B01], [B10, B11]]
    # C_top[:, 0:2] = A_L B00 + A_R B10
    # C_top[:, 2:4] = A_L B01 + A_R B11
    
    # We can use Strassen for each 2x2x2x2 multiplication? No, Strassen multiplies two 2x2 matrices.
    # Here we have 2x2 @ 2x2. That's exactly Strassen's case. Rank 7.
    # So for C_top[:, 0:2], we compute X1..X7 from A_L, A_R? No.
    # Wait, Strassen computes P = Q @ R. Here Q=A_L (2x2), R=B00 (2x2).
    # This gives 7 products.
    # Similarly for A_L @ B01 (7 products).
    # Similarly for A_R @ B10 (7 products).
    # Similarly for A_R @ B11 (7 products).
    # Total 28 products for the top 2 rows of C.
    
    # BUT, we can do better by reusing Strassen intermediates if we structure it carefully?
    # Actually, A_L @ B00 and A_R @ B10 are independent matrix multiplications.
    # However, we are summing them: A_L B00 + A_R B10.
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix? No.
    # It's (A_L, A_R) [B00; B10].
    # Let M = A_top (2x4), N = [B00; B10] (4x2).
    # C_top_left = M @ N.
    # Dimensions: 2x4 * 4x2 -> 2x2.
    # Rank of 2x4x2 multiplication tensor?
    # Schoolbook: 2*4*2 = 16.
    # Can we do better?
    # If we use Strassen on blocks:
    # Split M into M_L (2x2), M_R (2x2).
    # Split N into N_T (2x2), N_B (2x2).
    # C = M_L N_T + M_R N_B.
    # This is the sum of two 2x2x2 multiplications.
    # Rank of sum? If they are independent, rank adds. 7+7=14.
    # Is 14 better than 16? Yes.
    
    # So for C_top_left (2x2 result), we need 14 terms.
    # Similarly for C_top_right (2x2 result), we need 14 terms.
    # Total for top 2 rows of C: 28 terms.
    
    # Now for C_bottom (1x4 result).
    # C_bottom = A_bottom (1x4) @ B (4x4).
    # Split B into 2x2 blocks.
    # C_bottom[:, 0:2] = A_bottom_L (1x2) @ B00 (2x2) + A_bottom_R (1x2) @ B10 (2x2).
    # This is a 1x4 * 4x2 -> 1x2 multiplication.
    # Schoolbook: 1*4*2 = 8.
    # Can we do better?
    # Split A_bottom into L (1x2), R (1x2).
    # Split N = [B00; B10] into T (2x2), B (2x2).
    # C = L T + R B.
    # Sum of two 1x2x2 multiplications.
    # Rank of 1x2x2: 1*2*2 = 4.
    # Total 4+4=8. No gain over schoolbook for this specific block structure?
    # Wait, 1x2 @ 2x2 is just a row vector times matrix.
    # Rank of 1x2x2 tensor is 4? 
    # Let a = (a1, a2), B = [[b11, b12], [b21, b22]].
    # c1 = a1 b11 + a2 b21
    # c2 = a1 b12 + a2 b22
    # Can we do this in < 4 products?
    # If a1=0, c1=a2 b21, c2=a2 b22. 2 products.
    # Generally, rank of 1x2x2 is 4? No, it's 3?
    # Actually, matrix multiplication rank for 1x2x2 is 4?
    # Let's check.
    # If we use 3 products:
    # p1 = a1 b11
    # p2 = a2 b21
    # p3 = a1 b12 + a2 b22 ? No, mixed terms.
    # It is known that rank of 1xn x nxm is nm if n,m > 1? 
    # Actually, rank of 1x2x2 is 4? 
    # Wait, 1x1xn is n. 1x2x1 is 2.
    # For 1x2x2, we have 4 outputs? No, 2 outputs.
    # Inputs: 2 from A, 4 from B. Total 6 variables.
    # Bilinear forms.
    # It turns out rank is 4.
    
    # So for C_bottom, schoolbook is 8 terms (4 for left 2 cols, 4 for right 2 cols).
    # Total terms: 28 (top) + 8 (bottom) = 36.
    # Schoolbook total: 3*4*4 = 48.
    # 36 < 48. This is an improvement.

    # Let's verify the 2x4x2 rank reduction.
    # M (2x4), N (4x2).
    # M = [M_L, M_R] (2x2 each). N = [N_T; N_B] (2x2 each).
    # C = M_L N_T + M_R N_B.
    # We compute M_L N_T using Strassen (7 terms).
    # We compute M_R N_B using Strassen (7 terms).
    # Sum the results.
    # This works because Strassen gives the exact product.
    
    # We need to construct U, V, W matrices.
    # U shape: (n*m, rank) = (12, R)
    # V shape: (m*p, rank) = (16, R)
    # W shape: (p*n, rank) = (12, R)
    # Indices:
    # U row: i*m + j (i in 0..2, j in 0..3)
    # V row: j*p + k (j in 0..3, k in 0..3)
    # W row: k*n + i (k in 0..3, i in 0..2)
    
    # We will generate the Strassen factors for each 2x2x2 block.
    # Strassen for A(2x2) @ B(2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these linear combinations to the global U, V, W.
    # Each product term in Strassen corresponds to a rank-1 component in the tensor decomposition?
    # No, Strassen uses 7 multiplications, but the additions/subtractions are part of the decomposition.
    # In tensor decomposition, each column of U,V,W corresponds to one scalar multiplication.
    # The entries in U, V, W can be any constants (integers, fractions).
    # So for each Strassen "M" term, we can define a column in U, V, W?
    # Wait, Strassen computes M = (A_part) * (B_part).
    # This is a single scalar product of two linear forms?
    # No, M1 = (A00+A11) * (B00+B11).
    # This is the product of two linear forms.
    # So yes, each M corresponds to a rank-1 term in the bilinear form sense?
    # Yes, if we consider the scalar product.
    # So each Strassen step adds 7 columns to U, V, W.
    
    # Let's define a helper to add a Strassen block.
    # Inputs:
    # A_indices: 4 indices (i1,j1), (i1,j2), (i2,j1), (i2,j2) for the 2x2 A block.
    # B_indices: 4 indices (j1,k1), (j1,k2), (j2,k1), (j2,k2) for the 2x2 B block.
    # C_indices: 4 indices (i1,k1), (i1,k2), (i2,k1), (i2,k2) for the 2x2 C block.
    # Note: C is accumulated, so we add to existing columns? No, we add new columns.
    # The output C is the sum of contributions from all blocks.
    # So for each Strassen block, we add 7 new columns to U, V, W.
    # The W entries for these columns will be placed at the C indices.
    
    # However, we are summing two Strassen blocks: M_L N_T and M_R N_B.
    # They contribute to the same C block.
    # So we just add 7 columns for the first, and 7 columns for the second.
    
    # Let's code this.
    
    U = []
    V = []
    W = []
    
    def add_column(u_val, v_val, w_val):
        # u_val, v_val, w_val are dicts mapping index to value
        # We need to append to lists.
        pass
    
    # Better to use lists of dicts or sparse representation, then convert to dense arrays.
    # Since rank is small (~36), we can just build lists of length rank.
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_factors, v_factors, w_factors):
        nonlocal rank
        U_cols.append(u_factors)
        V_cols.append(v_factors)
        W_cols.append(w_factors)
        rank += 1
    
    # Strassen template
    # A: 2x2, B: 2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    
    def add_strassen_block(a_idx, b_idx, c_idx):
        # a_idx: [(i0,j0), (i0,j1), (i1,j0), (i1,j1)]
        # b_idx: [(j0,k0), (j0,k1), (j1,k0), (j1,k1)]
        # c_idx: [(i0,k0), (i0,k1), (i1,k0), (i1,k1)]
        
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00->1, A11->1
        # V: B00->1, B11->1
        # W: C00->1, C11->1
        add_term({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1})
        
        # M2 = (A10+A11)B00
        add_term({a10:1, a11:1}, {b00:1}, {c10:1, c11:-1})
        
        # M3 = A00(B10-B11)
        add_term({a00:1}, {b10:1, b11:-1}, {c01:1, c10:1})
        
        # M4 = A11(B01-B10)
        add_term({a11:1}, {b01:1, b10:-1}, {c00:1, c10:1})
        
        # M5 = (A00+A01)B11
        add_term({a00:1, a01:1}, {b11:1}, {c00:-1, c01:1})
        
        # M6 = (A10-A00)(B00+B10)
        add_term({a10:1, a00:-1}, {b00:1, b10:1}, {c11:1})
        
        # M7 = (A01-A11)(B01+B11)
        add_term({a01:1, a11:-1}, {b01:1, b11:1}, {c00:1})
        
    # Define indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    # Top 2 rows of A: i in 0,1.
    # A_L: i in 0,1; j in 0,1.
    # A_R: i in 0,1; j in 2,3.
    # B00: j in 0,1; k in 0,1.
    # B10: j in 2,3; k in 0,1.
    # B01: j in 0,1; k in 2,3.
    # B11: j in 2,3; k in 2,3.
    
    # C_top_left: i in 0,1; k in 0,1.
    # C_top_right: i in 0,1; k in 2,3.
    
    # Block 1: A_L @ B00
    a_L = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    b_00 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    c_00 = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    add_strassen_block(a_L, b_00, c_00)
    
    # Block 2: A_R @ B10
    a_R = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    b_10 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    add_strassen_block(a_R, b_10, c_00)
    
    # Block 3: A_L @ B01
    b_01 = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    c_01 = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    add_strassen_block(a_L, b_01, c_01)
    
    # Block 4: A_R @ B11
    b_11 = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    add_strassen_block(a_R, b_11, c_01)
    
    # Bottom row of A: i=2.
    # A_bottom_L: i=2, j in 0,1.
    # A_bottom_R: i=2, j in 2,3.
    
    # C_bottom_left: i=2, k in 0,1.
    # C_bottom_right: i=2, k in 2,3.
    
    # These are 1x2 @ 2x2 multiplications.
    # We can use schoolbook for these?
    # Rank 4 each.
    
    def add_1x2x2_block(a_idx, b_idx, c_idx):
        # a_idx: [j0, j1] (indices in U for row i=2)
        # b_idx: [(j0,k0), (j0,k1), (j1,k0), (j1,k1)]
        # c_idx: [k0, k1]
        a0, a1 = a_idx
        b00, b01, b10, b11 = b_idx
        c0, c1 = c_idx
        
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        add_term({a0:1}, {b00:1}, {c0:1})
        add_term({a1:1}, {b10:1}, {c0:1})
        add_term({a0:1}, {b01:1}, {c1:1})
        add_term({a1:1}, {b11:1}, {c1:1})
        
    a_bot_L = [u_idx(2,0), u_idx(2,1)]
    a_bot_R = [u_idx(2,2), u_idx(2,3)]
    
    c_bot_0 = [w_idx(0,2), w_idx(1,2)]
    c_bot_1 = [w_idx(2,2), w_idx(3,2)]
    
    add_1x2x2_block(a_bot_L, b_00, c_bot_0)
    add_1x2x2_block(a_bot_R, b_10, c_bot_0)
    
    add_1x2x2_block(a_bot_L, b_01, c_bot_1)
    add_1x2x2_block(a_bot_R, b_11, c_bot_1)
    
    # Now convert to dense matrices
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for col in range(rank):
        for u_ind, val in U_cols[col].items():
            U_mat[u_ind][col] = val
        for v_ind, val in V_cols[col].items():
            V_mat[v_ind][col] = val
        for w_ind, val in W_cols[col].items():
            W_mat[w_ind][col] = val
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}

# Optional: Verification function (not required by interface, but good for sanity)
def _verify():
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    n, m, p = 3, 4, 4
    # Tensor T[i,j,k] = 1 if i,j,k valid
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few random entries
    
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # True value: 1
        # Computed value:
        val = 0.0
        for r in range(U.shape[1]):
            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if abs(val - 1.0) > 1e-9:
            print(f"Failed at {i},{j},{k}: got {val}")
            return False
    return True

if __name__ == "__main__":
    if _verify():
        print("Verification passed.")
    else:
        print("Verification failed.")
