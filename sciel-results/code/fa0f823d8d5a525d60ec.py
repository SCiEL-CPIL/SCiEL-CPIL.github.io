import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 matrices.
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    # u_idx, v_idx, w_idx mapping
    # P = X Y
    # M1 = (X00+X11)(Y00+Y11) -> P00+M1, P11-M1? No P11 has -M2+M3+M6+M1.
    # Standard:
    # M1 = (X00 + X11) (Y00 + Y11)
    # M2 = (X10 + X11) Y00
    # M3 = X00 (Y01 - Y11)
    # M4 = X11 (Y10 - Y00)
    # M5 = (X00 + X01) Y11
    # M6 = (X10 - X00) (Y00 + Y01)
    # M7 = (X01 - X11) (Y10 + Y11)
    
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Define terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_idx: coeff}
    # local_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    def map_idx(global_start_r, global_start_c, local_r, local_c):
        return (global_start_r + local_r) * 4 + (global_start_c + local_c)

    # Base terms for a single 2x2 block
    base_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    # 4 blocks to apply
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Note: B rows 2:4)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Note: B cols 2:4)
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Wait, the current code logic:
    # 1. A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1)
    # 2. A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1)
    # 3. A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3)
    # 4. A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3)
    
    # Offsets:
    # Block 0: A_row=0, A_col=0; B_row=0, B_col=0; C_row=0, C_col=0. Base Col 0.
    # Block 1: A_row=0, A_col=2; B_row=2, B_col=0; C_row=0, C_col=0. Base Col 7.
    # Block 2: A_row=0, A_col=0; B_row=0, B_col=2; C_row=0, C_col=2. Base Col 14.
    # Block 3: A_row=0, A_col=2; B_row=2, B_col=2; C_row=0, C_col=2. Base Col 21.

    blocks = [
        # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, base_col)
        (0, 0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0, 7),
        (0, 0, 0, 2, 0, 2, 14),
        (0, 2, 2, 2, 0, 2, 21)
    ]
    
    for block_idx, (ar, ac, br, bc, cr, cc, bcol) in enumerate(blocks):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = bcol + t_idx
            
            for lu, coeff_u in u_loc.items():
                lr = lu // 2
                lc = lu % 2
                # A is 2x4. Index i*m + k. Here m=4? No, A is n x m. n=2, m=4.
                # A row index: ar + lr. A col index: ac + lc.
                # Flat index: (ar + lr) * 4 + (ac + lc)
                u_idx = (ar + lr) * 4 + (ac + lc)
                U[u_idx, col] += coeff_u
                
            for lv, coeff_v in v_loc.items():
                lr = lv // 2
                lc = lv % 2
                # B is m x p. m=4, p=4.
                # B row index: br + lr. B col index: bc + lc.
                # Flat index: (br + lr) * 4 + (bc + lc)
                v_idx = (br + lr) * 4 + (bc + lc)
                V[v_idx, col] += coeff_v
                
            for lw, coeff_w in w_loc.items():
                lr = lw // 2
                lc = lw % 2
                # C is n x p. n=2, p=4.
                # C row index: cr + lr. C col index: cc + lc.
                # Trace convention: W[k*n + i] where k is col, i is row.
                # So index is (cc + lc) * 2 + (cr + lr)
                w_idx = (cc + lc) * 2 + (cr + lr)
                W[w_idx, col] += coeff_w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Generate test matrices
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
        exit(1)
        
    print("OK rank", U.shape[1])
