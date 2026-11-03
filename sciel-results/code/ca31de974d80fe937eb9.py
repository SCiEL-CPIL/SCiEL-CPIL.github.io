import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    u_macro = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, -1],
        [0, 1, 0, -1] 
    ]
    v_macro = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    w_macro = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    u_micro = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, -1],
        [0, 1, 0, -1] 
    ]
    v_micro = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    w_micro = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    U = []
    V = []
    W = []
    
    for k in range(7):
        for r in range(7):
            u_vec = []
            v_vec = []
            w_vec = []
            
            for i in range(4):
                coeff_u = u_macro[k][i]
                if coeff_u != 0:
                    u_vec.extend([coeff_u * x for x in u_micro[r]])
                else:
                    u_vec.extend([0, 0, 0, 0])
                    
            for i in range(4):
                coeff_v = v_macro[k][i]
                if coeff_v != 0:
                    v_vec.extend([coeff_v * x for x in v_micro[r]])
                else:
                    v_vec.extend([0, 0, 0, 0])
                    
            for i in range(4):
                coeff_w = w_macro[k][i]
                if coeff_w != 0:
                    w_vec.extend([coeff_w * x for x in w_micro[r]])
                else:
                    w_vec.extend([0, 0, 0, 0])
                    
            # Pad V and W with 0s for the 5th column
            v_vec.extend([0, 0, 0, 0])
            w_vec.extend([0, 0, 0, 0])
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    for t in range(16):
        i = t // 4
        k = t % 4
        
        u_row = [0] * 16
        u_row[i*4 + k] = 1
        
        v_row = [0] * 20
        v_row[k*5 + 4] = 1
        
        w_row = [0] * 20
        w_row[16 + i] = 1
        
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
        
    return {"U": U, "V": V, "W": W}
