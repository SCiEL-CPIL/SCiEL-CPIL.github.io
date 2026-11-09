import time

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 5.0
    
    n, m, p = 4, 4, 9
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 2x2x2 Strassen tensors
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]
    ]
    
    A_blocks = {
        'A00': (0, 0), 'A01': (0, 2),
        'A10': (2, 0), 'A11': (2, 2)
    }
    
    B1_blocks = {
        'B100': (0, 0), 'B101': (0, 2),
        'B110': (2, 0), 'B111': (2, 2)
    }
    
    C1_blocks = {
        'C100': (0, 0), 'C101': (0, 2),
        'C110': (2, 0), 'C111': (2, 2)
    }
    
    Ms = [
        ([('A00', 1), ('A11', 1)], [('B100', 1), ('B111', 1)], [('C100', 1), ('C111', 1)]),
        ([('A10', 1), ('A11', 1)], [('B100', 1)], [('C110', 1)]),
        ([('A00', 1)], [('B101', 1), ('B111', -1)], [('C101', 1)]),
        ([('A11', 1)], [('B110', 1), ('B100', -1)], [('C100', 1)]),
        ([('A00', 1), ('A01', 1)], [('B111', 1)], [('C100', -1), ('C101', 1)]),
        ([('A10', 1), ('A00', -1)], [('B100', 1), ('B101', 1)], [('C111', 1)]),
        ([('A01', 1), ('A11', -1)], [('B110', 1), ('B111', 1)], [('C100', 1)])
    ]
    
    for a_blocks, b_blocks, c_blocks in Ms:
        for c in range(7):
            if time.time() > deadline - margin:
                return {"U": U, "V": V, "W": W}
                
            u_vec = [U2[i][c] for i in range(4)]
            v_vec = [V2[i][c] for i in range(4)]
            w_vec = [W2[i][c] for i in range(4)]
            
            for (a_name, a_sign) in a_blocks:
                i_s, j_s = A_blocks[a_name]
                for i in range(2):
                    for j in range(2):
                        U[i_s + i][j_s + j][col] = u_vec[i*2+j] * a_sign
            for (b_name, b_sign) in b_blocks:
                j_s, k_s = B1_blocks[b_name]
                for j in range(2):
                    for k in range(2):
                        V[j_s + j][k_s + k][col] = v_vec[j*2+k] * b_sign
            for (c_name, c_sign) in c_blocks:
                k_s, i_s = C1_blocks[c_name]
                for k in range(2):
                    for i in range(2):
                        W[k_s + k][i_s + i][col] = w_vec[k*2+i] * c_sign
            col += 1
            
    B2_blocks = {
        'B200': (0, 4), 'B201': (0, 6),
        'B210': (2, 4), 'B211': (2, 6)
    }
    
    C2_blocks = {
        'C200': (4, 0), 'C201': (4, 2),
        'C210': (6, 0), 'C211': (6, 2)
    }
    
    Ms2 = [
        ([('A00', 1), ('A11', 1)], [('B200', 1), ('B211', 1)], [('C200', 1), ('C211', 1)]),
        ([('A10', 1), ('A11', 1)], [('B200', 1)], [('C210', 1)]),
        ([('A00', 1)], [('B201', 1), ('B211', -1)], [('C201', 1)]),
        ([('A11', 1)], [('B210', 1), ('B200', -1)], [('C200', 1)]),
        ([('A00', 1), ('A01', 1)], [('B211', 1)], [('C200', -1), ('C201', 1)]),
        ([('A10', 1), ('A00', -1)], [('B200', 1), ('B201', 1)], [('C211', 1)]),
        ([('A01', 1), ('A11', -1)], [('B210', 1), ('B211', 1)], [('C200', 1)])
    ]
    
    for a_blocks, b_blocks, c_blocks in Ms2:
        for c in range(7):
            if time.time() > deadline - margin:
                return {"U": U, "V": V, "W": W}
                
            u_vec = [U2[i][c] for i in range(4)]
            v_vec = [V2[i][c] for i in range(4)]
            w_vec = [W2[i][c] for i in range(4)]
            
            for (a_name, a_sign) in a_blocks:
                i_s, j_s = A_blocks[a_name]
                for i in range(2):
                    for j in range(2):
                        U[i_s + i][j_s + j][col] = u_vec[i*2+j] * a_sign
            for (b_name, b_sign) in b_blocks:
                j_s, k_s = B2_blocks[b_name]
                for j in range(2):
                    for k in range(2):
                        V[j_s + j][k_s + k][col] = v_vec[j*2+k] * b_sign
            for (c_name, c_sign) in c_blocks:
                k_s, i_s = C2_blocks[c_name]
                for k in range(2):
                    for i in range(2):
                        W[k_s + k][i_s + i][col] = w_vec[k*2+i] * c_sign
            col += 1
            
    for i in range(4):
        for j in range(4):
            if time.time() > deadline - margin:
                return {"U": U, "V": V, "W": W}
            U[i][j][col] = 1
            V[j][8][col] = 1
            W[8][i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
