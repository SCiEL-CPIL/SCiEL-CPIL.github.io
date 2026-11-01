import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for P1..P7
    # U coeffs: [A00, A01, A10, A11]
    # V coeffs: [e, f, g, h]
    # W coeffs for C00, C10, C01, C11
    
    # s=1
    U_coeffs_1 = [1, 0, 0, 1]
    V_coeffs_1 = [1, 0, 0, 1] # e, f, g, h
    W_coeffs_1 = [1, 0, 0, 1] # C00, C10, C01, C11
    
    # s=2
    U_coeffs_2 = [0, 0, 1, 1]
    V_coeffs_2 = [1, 0, 0, 0]
    W_coeffs_2 = [0, 1, 0, -1]
    
    # s=3
    U_coeffs_3 = [1, 0, 0, 0]
    V_coeffs_3 = [0, 1, 0, -1]
    W_coeffs_3 = [0, 0, 1, 1]
    
    # s=4
    U_coeffs_4 = [0, 0, 0, 1]
    V_coeffs_4 = [-1, 0, 1, 0]
    W_coeffs_4 = [1, 1, 0, 0]
    
    # s=5
    U_coeffs_5 = [1, 1, 0, 0]
    V_coeffs_5 = [0, 0, 0, 1]
    W_coeffs_5 = [-1, 0, 1, 0]
    
    # s=6
    U_coeffs_6 = [-1, 0, 1, 0]
    V_coeffs_6 = [1, 1, 0, 0]
    W_coeffs_6 = [0, 0, 0, 1]
    
    # s=7
    U_coeffs_7 = [0, 1, 0, -1]
    V_coeffs_7 = [0, 0, 1, 1]
    W_coeffs_7 = [1, 0, 0, 0]
    
    all_U = [U_coeffs_1, U_coeffs_2, U_coeffs_3, U_coeffs_4, U_coeffs_5, U_coeffs_6, U_coeffs_7]
    all_V = [V_coeffs_1, V_coeffs_2, V_coeffs_3, V_coeffs_4, V_coeffs_5, V_coeffs_6, V_coeffs_7]
    all_W = [W_coeffs_1, W_coeffs_2, W_coeffs_3, W_coeffs_4, W_coeffs_5, W_coeffs_6, W_coeffs_7]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            u_c = all_U[s]
            v_c = all_V[s]
            w_c = all_W[s]
            
            # U
            for idx in range(4):
                if u_c[idx] != 0:
                    U[idx][col] = u_c[idx]
                    
            # V
            # e, f, g, h indices for block b
            # e: B[0, 2b] -> 0*6 + 2b
            # f: B[0, 2b+1] -> 0*6 + 2b+1
            # g: B[1, 2b] -> 1*6 + 2b
            # h: B[1, 2b+1] -> 1*6 + 2b+1
            
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if v_c[idx] != 0:
                    V[v_indices[idx]][col] = v_c[idx]
                    
            # W
            # C00, C10, C01, C11 indices for block b
            # C00: i=0, k=2b -> 2b*2 + 0
            # C10: i=1, k=2b -> 2b*2 + 1
            # C01: i=0, k=2b+1 -> (2b+1)*2 + 0
            # C11: i=1, k=2b+1 -> (2b+1)*2 + 1
            
            w_indices = [2*(2*b), 2*(2*b)+1, 2*(2*b+1), 2*(2*b+1)+1]
            for idx in range(4):
                if w_c[idx] != 0:
                    W[w_indices[idx]][col] = w_c[idx]
                    
    if time.time() > deadline:
        return None
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    if res:
        print("Rank:", len(res["U"][0]))
