import time

def construct():
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    block_size = 2
    
    # Strassen factors for 2x2x2 matrix multiplication
    # U, V are 4x7, W is 4x7
    # Indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  -> w = k*2 + i
    
    S_U = [
        [1, 0, 1, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    S_V = [
        [1, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for q in range(num_blocks):
        base_rank = q * 7
        for r in range(7):
            col = base_rank + r
            
            # Fill U
            # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global U index: i * m + j
            for i in range(2):
                for j in range(2):
                    local_u_idx = i * 2 + j
                    val = S_U[r][local_u_idx]
                    if val != 0:
                        global_u_idx = i * m + j
                        U[global_u_idx][col] = val
                        
            # Fill V
            # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local k' is 0,1. Global k = 2q + k'
            # Global V index: j * p + k
            for j in range(2):
                for kp in range(2):
                    local_v_idx = j * 2 + kp
                    val = S_V[r][local_v_idx]
                    if val != 0:
                        global_k = 2 * q + kp
                        global_v_idx = j * p + global_k
                        V[global_v_idx][col] = val
                        
            # Fill W
            # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
            # Local W index w_loc = kp * 2 + i
            # Global W index: k * n + i
            for kp in range(2):
                for i in range(2):
                    local_w_idx = kp * 2 + i
                    val = S_W[r][local_w_idx]
                    if val != 0:
                        global_k = 2 * q + kp
                        global_w_idx = global_k * n + i
                        W[global_w_idx][col] = val
                        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = 2, 2, 16
    import numpy as np
    
    # Verify decomposition
    # T_{u,v,w} = sum_r U[u,r] V[v,r] W[w,r]
    # Should be 1 if u=im+j, v=jp+k, w=kn+i, else 0
    
    # Pick a random valid (i,j,k)
    i, j, k = 1, 0, 5
    u = i * m + j
    v = j * p + k
    w = k * n + i
    
    total = 0
    for r in range(len(U[0])):
        total += U[u][r] * V[v][r] * W[w][r]
        
    if total != 1:
        print(f"Error at ({i},{j},{k}): {total}")
    else:
        print(f"Correct at ({i},{j},{k})")
        
    # Pick an invalid combination
    # e.g. i=0, j=1, k=5 -> u=1, v=1*16+5=21, w=5*2+0=10
    # But Strassen blocks are disjoint in k.
    # Let's check a case that should be 0.
    # In schoolbook, T is 0 if indices don't match the specific (i,j,k) tuple.
    # Here, we sum over blocks.
    # The sum is 1 if there is a block q such that the local indices match Strassen decomposition.
    # Since Strassen decomposition covers the full 2x2x2 tensor, it should be 1 for any valid (i,j,k).
    # And 0 otherwise?
    # The tensor T is defined by the schoolbook as having 1s ONLY at (im+j, jp+k, kn+i).
    # So for any other combination of u,v,w, it should be 0.
    
    u_test, v_test, w_test = 0, 1, 0
    total_test = 0
    for r in range(len(U[0])):
        total_test += U[u_test][r] * V[v_test][r] * W[w_test][r]
        
    if total_test != 0:
        print(f"Error at invalid ({u_test},{v_test},{w_test}): {total_test}")
    else:
        print(f"Correct at invalid ({u_test},{v_test},{w_test})")
        
    print(f"Rank: {len(U[0])}")
