import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 50s margin
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # U has 6 entries (2x3): indices i*3+j
    # V has 12 entries (3x4): indices j*4+k
    # W has 8 entries (4x2): indices k*2+i
    # W index mapping: 0=C[0,0], 1=C[1,0], 2=C[0,1], 3=C[1,1], 4=C[0,2], 5=C[1,2], 6=C[0,3], 7=C[1,3]
    
    def make_u(**kwargs):
        vec = [0] * 6
        for key, val in kwargs.items():
            vec[key] = val
        return vec
    
    def make_v(**kwargs):
        vec = [0] * 12
        for key, val in kwargs.items():
            vec[key] = val
        return vec
    
    def make_w(**kwargs):
        vec = [0] * 8
        for key, val in kwargs.items():
            vec[key] = val
        return vec
    
    def u_idx(i, j):
        return i * 3 + j
    
    def v_idx(j, k):
        return j * 4 + k
    
    # Part 1: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen rank-7
    # M1 = (a00+a11)(b00+b11): C00+=1, C10+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,0): 1, v_idx(1,1): 1}),
             make_w(**{0: 1, 2: 1}))
    # M2 = (a10+a11)(b00+b01): C10+=1, C11+=1
    add_term(make_u(**{u_idx(1,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,0): 1, v_idx(0,1): 1}),
             make_w(**{2: 1, 3: 1}))
    # M3 = a00*(b10-b11): C01+=1, C11-=1
    add_term(make_u(**{u_idx(0,0): 1}),
             make_v(**{v_idx(1,0): 1, v_idx(1,1): -1}),
             make_w(**{1: 1, 3: -1}))
    # M4 = a11*(b01-b00): C00-=1, C01+=1
    add_term(make_u(**{u_idx(1,1): 1}),
             make_v(**{v_idx(0,1): 1, v_idx(0,0): -1}),
             make_w(**{0: -1, 1: 1}))
    # M5 = (a00+a01)*b11: C00-=1, C11+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(0,1): 1}),
             make_v(**{v_idx(1,1): 1}),
             make_w(**{0: -1, 3: 1}))
    # M6 = (a10+a11)*b01: C11+=1
    add_term(make_u(**{u_idx(1,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,1): 1}),
             make_w(**{3: 1}))
    # M7 = (a00-a10)*(b00+b10): C00+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(1,0): -1}),
             make_v(**{v_idx(0,0): 1, v_idx(1,0): 1}),
             make_w(**{0: 1}))
    
    # Part 1b: A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
    # W indices: 4=C[0,2], 5=C[1,2], 6=C[0,3], 7=C[1,3]
    # M1 = (a00+a11)(b02+b13): C02+=1, C12+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,2): 1, v_idx(1,3): 1}),
             make_w(**{4: 1, 6: 1}))
    # M2 = (a10+a11)(b02+b03): C12+=1, C13+=1
    add_term(make_u(**{u_idx(1,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,2): 1, v_idx(0,3): 1}),
             make_w(**{6: 1, 7: 1}))
    # M3 = a00*(b12-b13): C03+=1, C13-=1
    add_term(make_u(**{u_idx(0,0): 1}),
             make_v(**{v_idx(1,2): 1, v_idx(1,3): -1}),
             make_w(**{5: 1, 7: -1}))
    # M4 = a11*(b03-b02): C02-=1, C03+=1
    add_term(make_u(**{u_idx(1,1): 1}),
             make_v(**{v_idx(0,3): 1, v_idx(0,2): -1}),
             make_w(**{4: -1, 5: 1}))
    # M5 = (a00+a01)*b13: C02-=1, C13+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(0,1): 1}),
             make_v(**{v_idx(1,3): 1}),
             make_w(**{4: -1, 7: 1}))
    # M6 = (a10+a11)*b03: C13+=1
    add_term(make_u(**{u_idx(1,0): 1, u_idx(1,1): 1}),
             make_v(**{v_idx(0,3): 1}),
             make_w(**{7: 1}))
    # M7 = (a00-a10)*(b02+b12): C02+=1
    add_term(make_u(**{u_idx(0,0): 1, u_idx(1,0): -1}),
             make_v(**{v_idx(0,2): 1, v_idx(1,2): 1}),
             make_w(**{4: 1}))
    
    # Part 2: A[0:2, 2:3] @ B[2:3, 0:4] -> C[0:2, 0:4]
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(4):
            w_k = k * 2 + i
            add_term(make_u(**{u_idx(i, 2): 1}),
                     make_v(**{v_idx(2, k): 1}),
                     make_w(**{w_k: 1}))
    
    rank = len(U)
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_rec = np.zeros((n, m, p))
    for r_idx in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i, j, k] += U_arr[i * m + j, r_idx] * V_arr[j * p + k, r_idx] * W_arr[k * n + i, r_idx]
    
    T_target = np.ones((n, m, p))
    error = np.linalg.norm(T_target - T_rec)
    
    if error < 1e-9:
        return {
            "U": U,
            "V": V,
            "W": W
        }
    
    # Fallback to schoolbook
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
