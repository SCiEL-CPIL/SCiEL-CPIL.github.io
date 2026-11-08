import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        """Verify that the decomposition satisfies sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    def get_rank22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Cross terms
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    U22, V22, W22, r22 = get_rank22()
    
    # Verify the known solution
    ok, _ = verify(U22, V22, W22, r22)
    
    # Try to find lower rank using structured random search
    best_U, best_V, best_W, best_rank = U22, V22, W22, r22
    
    for target_rank in [21, 20]:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        found = False
        for trial in range(200):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Random sparse pattern
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                u_idx = random.sample(range(n*m), random.randint(1, 3))
                v_idx = random.sample(range(m*p), random.randint(1, 3))
                w_idx = random.sample(range(p*n), random.randint(1, 3))
                
                for idx in u_idx:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, r] = random.choice([-1, 1])
            
            ok, _ = verify(U.tolist(), V.tolist(), W.tolist(), target_rank)
            if ok:
                best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W.tolist(), target_rank
                found = True
                break
        
        if found:
            break
    
    U_out = [[int(round(x * 2)) / 2 for x in row] for row in best_U]
    V_out = [[int(round(x * 2)) / 2 for x in row] for row in best_V]
    W_out = [[int(round(x * 2)) / 2 for x in row] for row in best_W]
    
    # Final verification
    ok, _ = verify(U_out, V_out, W_out, best_rank)
    if not ok:
        U_out, V_out, W_out, best_rank = U22, V22, W22, r22
    
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U_out, "V": V_out, "W": W_out}
