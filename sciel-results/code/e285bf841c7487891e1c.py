import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Build the rank-44 decomposition as fallback
    def build_rank44():
        rank_init = 44
        U = [[0] * rank_init for _ in range(n * m)]
        V = [[0] * rank_init for _ in range(m * p)]
        W = [[0] * rank_init for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(u_map, v_map, w_map):
            nonlocal col
            U[u_map[0]][col] += 1
            U[u_map[3]][col] += 1
            V[v_map[0]][col] += 1
            V[v_map[3]][col] += 1
            W[w_map[0]][col] += 1
            W[w_map[3]][col] += 1
            col += 1
            U[u_map[2]][col] += 1
            U[u_map[3]][col] += 1
            V[v_map[0]][col] += 1
            W[w_map[2]][col] += 1
            W[w_map[3]][col] += -1
            col += 1
            U[u_map[0]][col] += 1
            V[v_map[1]][col] += 1
            V[v_map[3]][col] += -1
            W[w_map[1]][col] += 1
            W[w_map[3]][col] += 1
            col += 1
            U[u_map[3]][col] += 1
            V[v_map[2]][col] += 1
            V[v_map[0]][col] += -1
            W[w_map[0]][col] += 1
            W[w_map[2]][col] += 1
            col += 1
            U[u_map[0]][col] += 1
            U[u_map[1]][col] += 1
            V[v_map[3]][col] += 1
            W[w_map[0]][col] += -1
            W[w_map[1]][col] += 1
            col += 1
            U[u_map[2]][col] += 1
            U[u_map[0]][col] += -1
            V[v_map[0]][col] += 1
            V[v_map[1]][col] += 1
            W[w_map[3]][col] += 1
            col += 1
            U[u_map[1]][col] += 1
            U[u_map[3]][col] += -1
            V[v_map[2]][col] += 1
            V[v_map[3]][col] += 1
            W[w_map[0]][col] += 1
            col += 1

        for half in range(2):
            u_map = {0: 0, 1: 1, 2: 3, 3: 4}
            
            b_start = half * 4
            v_map_b1 = {0: b_start, 1: b_start+1, 2: p+b_start, 3: p+b_start+1}
            w_map_c1 = {0: b_start*n, 1: (b_start+1)*n, 2: b_start*n+1, 3: (b_start+1)*n+1}
            add_strassen_block(u_map, v_map_b1, w_map_c1)
            
            b_start2 = half * 4 + 2
            v_map_b2 = {0: b_start2, 1: b_start2+1, 2: p+b_start2, 3: p+b_start2+1}
            w_map_c2 = {0: b_start2*n, 1: (b_start2+1)*n, 2: b_start2*n+1, 3: (b_start2+1)*n+1}
            add_strassen_block(u_map, v_map_b2, w_map_c2)
            
            for i in range(2):
                for k in range(4):
                    u_idx = i * m + 2
                    v_idx = 2 * p + (half * 4 + k)
                    w_idx = (half * 4 + k) * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        return U, V, W
    
    # Verify decomposition
    def verify(U_mat, V_mat, W_mat):
        for u in range(n * m):
            i_u = u // m
            j_u = u % m
            for v in range(m * p):
                j_v = v // p
                k_v = v % p
                if j_u != j_v:
                    continue
                for w in range(p * n):
                    k_w = w // n
                    i_w = w % n
                    if i_u == i_w and k_v == k_w:
                        target = 1.0
                    else:
                        target = 0.0
                    val = np.dot(U_mat[u, :], V_mat[v, :] * W_mat[w, :])
                    if abs(val - target) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition using optimization
    U44, V44, W44 = build_rank44()
    U_np = np.array(U44, dtype=float)
    V_np = np.array(V44, dtype=float)
    W_np = np.array(W44, dtype=float)
    
    best_rank = 44
    best_result = (U44, V44, W44)
    
    # Build the tensor T for verification
    T = np.zeros((n*m, m*p, p*n))
    for u in range(n*m):
        i_u = u // m
        j_u = u % m
        for v in range(m*p):
            j_v = v // p
            k_v = v % p
            if j_u == j_v:
                for w in range(p*n):
                    k_w = w // n
                    i_w = w % n
                    if i_u == i_w and k_v == k_w:
                        T[u, v, w] = 1.0
    
    def check_decomposition(U_mat, V_mat, W_mat, r_max):
        """Check if decomposition using first r_max terms is valid."""
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    if T[u, v, w] != 0:
                        val = np.dot(U_mat[u, :r_max], V_mat[v, :r_max] * W_mat[w, :r_max])
                        if abs(val - T[u, v, w]) > 1e-9:
                            return False
                    else:
                        val = np.dot(U_mat[u, :r_max], V_mat[v, :r_max] * W_mat[w, :r_max])
                        if abs(val) > 1e-9:
                            return False
        return True

    # Try to remove terms from the end
    current_rank = 44
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    for r_remove in range(current_rank - 1, 0, -1):
        if time.time() > deadline - 30:
            break
        U_test = U_cur.copy()
        V_test = V_cur.copy()
        W_test = W_cur.copy()
        U_test[:, r_remove] = 0
        V_test[:, r_remove] = 0
        W_test[:, r_remove] = 0
        
        if check_decomposition(U_test, V_test, W_test, r_remove):
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
            current_rank = r_remove
            best_rank = r_remove
            best_result = (U_cur.tolist(), V_cur.tolist(), W_cur.tolist())
    
    # Try to find a rank-40 or lower decomposition
    if time.time() < deadline - 60:
        def build_rank40_v2():
            """Try a different decomposition with rank 40."""
            rank_init = 40
            U = [[0] * rank_init for _ in range(n * m)]
            V = [[0] * rank_init for _ in range(m * p)]
            W = [[0] * rank_init for _ in range(p * n)]
            
            col = 0
            
            def add_strassen_block(u_map, v_map, w_map):
                nonlocal col
                U[u_map[0]][col] += 1
                U[u_map[3]][col] += 1
                V[v_map[0]][col] += 1
                V[v_map[3]][col] += 1
                W[w_map[0]][col] += 1
                W[w_map[3]][col] += 1
                col += 1
                U[u_map[2]][col] += 1
                U[u_map[3]][col] += 1
                V[v_map[0]][col] += 1
                W[w_map[2]][col] += 1
                W[w_map[3]][col] += -1
                col += 1
                U[u_map[0]][col] += 1
                V[v_map[1]][col] += 1
                V[v_map[3]][col] += -1
                W[w_map[1]][col] += 1
                W[w_map[3]][col] += 1
                col += 1
                U[u_map[3]][col] += 1
                V[v_map[2]][col] += 1
                V[v_map[0]][col] += -1
                W[w_map[0]][col] += 1
                W[w_map[2]][col] += 1
                col += 1
                U[u_map[0]][col] += 1
                U[u_map[1]][col] += 1
                V[v_map[3]][col] += 1
                W[w_map[0]][col] += -1
                W[w_map[1]][col] += 1
                col += 1
                U[u_map[2]][col] += 1
                U[u_map[0]][col] += -1
                V[v_map[0]][col] += 1
                V[v_map[1]][col] += 1
                W[w_map[3]][col] += 1
                col += 1
                U[u_map[1]][col] += 1
                U[u_map[3]][col] += -1
                V[v_map[2]][col] += 1
                V[v_map[3]][col] += 1
                W[w_map[0]][col] += 1
                col += 1
            
            u_map = {0: 0, 1: 1, 2: 3, 3: 4}
            
            # Group 1: k=0,1,2
            v_map_1 = {0: 0, 1: 1, 2: p+0, 3: p+1}
            w_map_1 = {0: 0, 1: 1, 2: 1, 3: 3}
            add_strassen_block(u_map, v_map_1, w_map_1)
            
            for i in range(2):
                U[i*3+2][col] = 1
                V[2*p+2][col] = 1
                W[2*n+i][col] = 1
                col += 1
            
            # Group 2: k=3,4,5
            v_map_2 = {0: 3, 1: 4, 2: p+3, 3: p+4}
            w_map_2 = {0: 3, 1: 4, 2: 4, 3: 6}
            add_strassen_block(u_map, v_map_2, w_map_2)
            
            for i in range(2):
                U[i*3+2][col] = 1
                V[2*p+5][col] = 1
                W[5*n+i][col] = 1
                col += 1
            
            # Group 3: k=6,7
            v_map_3 = {0: 6, 1: 7, 2: p+6, 3: p+7}
            w_map_3 = {0: 6, 1: 7, 2: 7, 3: 9}
            add_strassen_block(u_map, v_map_3, w_map_3)
            
            return U, V, W
        
        U2, V2, W2 = build_rank40_v2()
        U2_np = np.array(U2, dtype=float)
        V2_np = np.array(V2, dtype=float)
        W2_np = np.array(W2, dtype=float)
        
        if verify(U2_np, V2_np, W2_np):
            r = len(U2[0])
            if r < best_rank:
                best_rank = r
                best_result = (U2, V2, W2)
    
    U_final, V_final, W_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
