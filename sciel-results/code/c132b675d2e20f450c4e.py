    for r in range(16):
        r2 = r // 4
        r1 = r % 4
        
        i2 = r2 // 2
        j2 = r2 % 2
        i1 = r1 // 2
        j1 = r1 % 2
        
        i4 = i2 * 2 + i1
        j4 = j2 * 2 + j1
        
        global_u = i4 * 5 + j4
        
        # Similarly for V and W
        v_r2 = r // 4
        v_r1 = r % 4
        j2_v = v_r2 // 2
        k2_v = v_r2 % 2
        j1_v = v_r1 // 2
        k1_v = v_r1 % 2
        j4_v = j2_v * 2 + j1_v
        k4_v = k2_v * 2 + k1_v
        global_v = j4_v * 5 + k4_v
        
        w_r2 = r // 4
        w_r1 = r % 4
        k2_w = w_r2 // 2
        i2_w = w_r2 % 2
        k1_w = w_r1 // 2
        i1_w = w_r1 % 2
        k4_w = k2_w * 2 + k1_w
        i4_w = i2_w * 2 + i1_w
        global_w = k4_w * 4 + i4_w
        
        for c in range(49):
            U[global_u, col + c] = U4[r, c]
            V[global_v, col + c] = V4[r, c]
            W[global_w, col + c] = W4[r, c]
