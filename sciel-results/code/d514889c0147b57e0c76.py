import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition (Rank 7)
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2: (A10+A11)B00
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[1, 1] = 1; W_out[3, 1] = -1
    
    # M3: A00(B01-B11)
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[2, 2] = 1; W_out[3, 2] = 1
    
    # M4: A11(B10-B00)
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5: (A00+A01)B11
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1

    # 3x3 Schoolbook decomposition (Rank 27)
    U_in = np.zeros((9, 27), dtype=int)
    V_in = np.zeros((9, 27), dtype=int)
    W_in = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_in[i * 3 + j, col] = 1
                V_in[j * 3 + k, col] = 1
                W_in[k * 3 + i, col] = 1
                col += 1
                
    rank = 7 * 27
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(27):
            if time.time() > DEADLINE:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            c = c_out * 27 + c_in
            
            # Kronecker product logic
            # U[r, c] = U_out[r_out, c_out] * U_in[r_in, c_in]
            # r = (r_out_block * 3 + r_in_block) * 6 + ...
            
            # Vectorized approach for speed
            # U_out is (4,7), U_in is (9,27)
            # We need to map (r_out, r_in) to r
            # r_out is index in 2x2 block (0..3)
            # r_in is index in 3x3 block (0..8)
            # r_out = i_out * 2 + j_out
            # r_in = i_in * 3 + j_in
            # r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
            
            # Pre-compute indices
            r_outs = np.arange(4)
            i_outs = r_outs // 2
            j_outs = r_outs % 2
            
            r_ins = np.arange(9)
            i_ins = r_ins // 3
            j_ins = r_ins % 3
            
            # For each c_out, c_in
            u_out_vals = U_out[:, c_out]
            u_in_vals = U_in[:, c_in]
            
            # Non-zero indices
            nz_out = np.nonzero(u_out_vals)[0]
            nz_in = np.nonzero(u_in_vals)[0]
            
            for ro in nz_out:
                io = i_outs[ro]
                jo = j_outs[ro]
                val_o = u_out_vals[ro]
                
                for ri in nz_in:
                    ii = i_ins[ri]
                    ji = j_ins[ri]
                    val_i = u_in_vals[ri]
                    
                    r = (io * 3 + ii) * 6 + (jo * 3 + ji)
                    U[r, c] += val_o * val_i
                    
            # Similarly for V and W
            v_out_vals = V_out[:, c_out]
            v_in_vals = V_in[:, c_in]
            
            nz_out = np.nonzero(v_out_vals)[0]
            nz_in = np.nonzero(v_in_vals)[0]
            
            for ro in nz_out:
                jo = ro // 2
                ko = ro % 2
                val_o = v_out_vals[ro]
                
                for ri in nz_in:
                    ji = ri // 3
                    ki = ri % 3
                    val_i = v_in_vals[ri]
                    
                    r = (jo * 3 + ji) * 6 + (ko * 3 + ki)
                    V[r, c] += val_o * val_i
                    
            w_out_vals = W_out[:, c_out]
            w_in_vals = W_in[:, c_in]
            
            nz_out = np.nonzero(w_out_vals)[0]
            nz_in = np.nonzero(w_in_vals)[0]
            
            for ro in nz_out:
                ko = ro // 2
                io = ro % 2
                val_o = w_out_vals[ro]
                
                for ri in nz_in:
                    ki = ri // 3
                    ii = ri % 3
                    val_i = w_in_vals[ri]
                    
                    r = (ko * 3 + ki) * 6 + (io * 3 + ii)
                    W[r, c] += val_o * val_i

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
