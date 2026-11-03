import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(k, i): return k * 4 + i
    
    def block_coeff(block_id):
        coeffs = {}
        if block_id == 0:
            coeffs = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        elif block_id == 1:
            coeffs = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}
        elif block_id == 2:
            coeffs = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}
        elif block_id == 3:
            coeffs = {(2,2): 1, (2,3): 1, (3,2): 1, (3,3): 1}
        return coeffs

    def add_blocks(b1, c1, b2, c2):
        res = {}
        for k, v in b1.items():
            res[k] = res.get(k, 0) + c1 * v
        for k, v in b2.items():
            res[k] = res.get(k, 0) + c2 * v
        return res

    steps = []
    
    X0 = [ [add_blocks(block_coeff(0), 1, block_coeff(3), 1)], [{}], [{}], [{}] ]
    Y0 = [ [add_blocks(block_coeff(0), 1, block_coeff(3), 1)], [{}], [{}], [{}] ]
    steps.append((X0, Y0))
    
    X1 = [ [add_blocks(block_coeff(2), 1, block_coeff(3), 1)], [{}], [{}], [{}] ]
    Y1 = [ [block_coeff(0)], [{}], [{}], [{}] ]
    steps.append((X1, Y1))
    
    X2 = [ [block_coeff(0)], [{}], [{}], [{}] ]
    Y2 = [ [{}], [add_blocks(block_coeff(1), 1, block_coeff(3), -1)], [{}], [{}] ]
    steps.append((X2, Y2))
    
    X3 = [ [block_coeff(3)], [{}], [{}], [{}] ]
    Y3 = [ [add_blocks(block_coeff(0), -1, block_coeff(2), 1)], [{}], [{}], [{}] ]
    steps.append((X3, Y3))
    
    X4 = [ [add_blocks(block_coeff(0), 1, block_coeff(1), 1)], [{}], [{}], [{}] ]
    Y4 = [ [block_coeff(3)], [{}], [{}], [{}] ]
    steps.append((X4, Y4))
    
    X5 = [ [add_blocks(block_coeff(0), -1, block_coeff(2), 1)], [{}], [{}], [{}] ]
    Y5 = [ [add_blocks(block_coeff(0), 1, block_coeff(1), 1)], [{}], [{}], [{}] ]
    steps.append((X5, Y5))
    
    X6 = [ [add_blocks(block_coeff(3), -1, block_coeff(1), 1)], [{}], [{}], [{}] ]
    Y6 = [ [{}], [{}], [block_coeff(2)], [block_coeff(3)] ]
    steps.append((X6, Y6))
    
    for l in range(7):
        if time.time() > deadline:
            break
        Xl, Yl = steps[l]
        x11, x12 = Xl[0][0], Xl[0][1]
        x21, x22 = Xl[1][0], Xl[1][1]
        y11, y12 = Yl[0][0], Yl[0][1]
        y21, y22 = Yl[1][0], Yl[1][1]
        
        s_A = []
        s_B = []
        
        s_A.append(add_blocks(x11, 1, x22, 1))
        s_B.append(add_blocks(y11, 1, y22, 1))
        
        s_A.append(add_blocks(x21, 1, x22, 1))
        s_B.append(y11)
        
        s_A.append(x11)
        s_B.append(add_blocks(y12, 1, y22, -1))
        
        s_A.append(x22)
        s_B.append(add_blocks(y21, 1, y11, -1))
        
        s_A.append(add_blocks(x11, 1, x12, 1))
        s_B.append(y22)
        
        s_A.append(add_blocks(x11, -1, x21, 1))
        s_B.append(add_blocks(y11, 1, y12, 1))
        
        s_A.append(add_blocks(x12, 1, x22, -1))
        s_B.append(add_blocks(y21, 1, y22, 1))
        
        contribs = [
            [(0,0, 1), (1,1, 1)],
            [(1,0, 1), (1,1, -1)],
            [(0,1, 1), (1,1, 1)],
            [(0,0, 1), (1,0, 1)],
            [(0,0, -1), (0,1, 1)],
            [(1,1, 1)],
            [(0,0, 1)]
        ]
        
        for t in range(7):
            r = l * 7 + t
            for (i, j), coeff in s_A[t].items():
                U[u_idx(i, j)][r] += coeff
            for (j, k), coeff in s_B[t].items():
                V[v_idx(j, k)][r] += coeff
            
            for (br, bc, coeff) in contribs[t]:
                for ri in range(2):
                    for ci in range(2):
                        k = 2 * bc + ci
                        i = 2 * br + ri
                        W[w_idx(k, i)][r] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
