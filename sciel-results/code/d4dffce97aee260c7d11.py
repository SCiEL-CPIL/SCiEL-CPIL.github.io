import time
import sys

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Target Rank 60
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: M_{1,4,8}
    # i=0, j=0..3, k=0..7
    # Rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # Block 2: M_{2,4,8}
    # i=1,2 (local 0,1), j=0..3, k=0..7
    # Split into 4 blocks of M_{2,2,2}
    
    # Helper to get Strassen tensors
    def get_strassen_tensors():
        tensors = []
        # 1
        tensors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # 2
        tensors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # 3
        tensors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # 4
        tensors.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
        # 5
        tensors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # 6
        tensors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # 7
        tensors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return tensors

    strassen_tensors = get_strassen_tensors()
    
    # Sub-blocks: (j_offset, k_offset)
    # j_off in {0, 2}, k_off in {0, 2}
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            # Local j' in 0..1, k' in 0..1
            # Global j = j' + j_off, k = k' + k_off
            # Global i = 1 + i' (i' in 0..1)
            
            for s in range(7):
                if time.time() > deadline:
                    # Return what we have so far? No, must be valid.
                    # If we run out of time, we might return an incomplete matrix which is invalid.
                    # However, the construction is fast (constant time).
                    pass
                    
                u_local, v_local, w_local = strassen_tensors[s]
                
                # u_local is indexed [i'*2 + j']
                # v_local is indexed [j'*2 + k']
                # w_local is indexed [k'*2 + i'] (Note: W[k*n+i] with n=2 locally)
                
                for ip in range(2):
                    for jp in range(2):
                        u_idx_local = ip * 2 + jp
                        u_val = u_local[u_idx_local]
                        if u_val != 0:
                            # Global U index: (1+ip)*4 + (jp + j_off)
                            u_global_idx = (1 + ip) * 4 + (jp + j_off)
                            U[u_global_idx][col] += u_val
                            
                for jp in range(2):
                    for kp in range(2):
                        v_idx_local = jp * 2 + kp
                        v_val = v_local[v_idx_local]
                        if v_val != 0:
                            # Global V index: (jp + j_off)*8 + (kp + k_off)
                            v_global_idx = (jp + j_off) * 8 + (kp + k_off)
                            V[v_global_idx][col] += v_val
                            
                for kp in range(2):
                    for ip in range(2):
                        w_idx_local = kp * 2 + ip
                        w_val = w_local[w_idx_local]
                        if w_val != 0:
                            # Global W index: (kp + k_off)*3 + (1 + ip)
                            w_global_idx = (kp + k_off) * 3 + (1 + ip)
                            W[w_global_idx][col] += w_val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

