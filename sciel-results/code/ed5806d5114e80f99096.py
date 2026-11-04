import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds is plenty for this
    
    n, m, p = 4, 4, 8
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for a in range(2):
        for d in range(4):
            for c in range(2):
                if time.time() > deadline:
                    break
                
                def u_idx(r, c_local):
                    return 4 * (2 * a + r) + (2 * c + c_local)
                
                def v_idx(r, c_local):
                    return 8 * (2 * c + r) + (2 * d + c_local)
                
                def w_idx(r, c_local):
                    return 8 * (2 * d + c_local) + (2 * a + r)
                
                # M1
                U[u_idx(0,0)][col] += 1
                U[u_idx(1,1)][col] += 1
                V[v_idx(0,0)][col] += 1
                V[v_idx(1,1)][col] += 1
                W[w_idx(0,0)][col] += 1
                W[w_idx(1,1)][col] += 1
                col += 1
                
                # M2
                U[u_idx(1,0)][col] += 1
                U[u_idx(1,1)][col] += 1
                V[v_idx(0,0)][col] += 1
                W[w_idx(1,0)][col] += 1
                col += 1
                
                # M3
                U[u_idx(0,0)][col] += 1
                V[v_idx(0,1)][col] += 1
                V[v_idx(1,1)][col] -= 1
                W[w_idx(0,1)][col] += 1
                W[w_idx(1,1)][col] += 1
                col += 1
                
                # M4
                U[u_idx(1,1)][col] += 1
                V[v_idx(1,0)][col] += 1
                V[v_idx(0,0)][col] -= 1
                W[w_idx(0,0)][col] += 1
                W[w_idx(1,0)][col] += 1
                col += 1
                
                # M5
                U[u_idx(0,0)][col] += 1
                U[u_idx(0,1)][col] += 1
                V[v_idx(1,1)][col] += 1
                W[w_idx(0,0)][col] -= 1
                W[w_idx(0,1)][col] += 1
                col += 1
                
                # M6
                U[u_idx(1,0)][col] += 1
                U[u_idx(0,0)][col] -= 1
                V[v_idx(0,0)][col] += 1
                V[v_idx(0,1)][col] += 1
                W[w_idx(1,1)][col] += 1
                col += 1
                
                # M7
                U[u_idx(0,1)][col] += 1
                U[u_idx(1,1)][col] -= 1
                V[v_idx(1,0)][col] += 1
                V[v_idx(1,1)][col] += 1
                W[w_idx(0,0)][col] += 1
                col += 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
