        # M1
        W[c_idx[0]][c_start+0] += 1
        W[c_idx[3]][c_start+0] += 1
        
        # M2
        W[c_idx[2]][c_start+1] += 1
        W[c_idx[3]][c_start+1] += -1
        
        # M3
        W[c_idx[1]][c_start+2] += 1
        W[c_idx[3]][c_start+2] += 1
        
        # M4
        W[c_idx[0]][c_start+3] += 1
        W[c_idx[2]][c_start+3] += 1
        
        # M5
        W[c_idx[0]][c_start+4] += -1
        W[c_idx[3]][c_start+4] += 1
        
        # M6
        W[c_idx[3]][c_start+5] += 1
        
        # M7
        W[c_idx[0]][c_start+6] += 1
        W[c_idx[1]][c_start+6] += -1
