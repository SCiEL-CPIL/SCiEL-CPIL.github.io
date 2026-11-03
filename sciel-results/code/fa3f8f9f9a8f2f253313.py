import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0
    
    n, m, p = DIMENSIONS
    
    def make_rank_55():
        """
        Rank 55 decomposition for 2x3 x 3x10.
        """
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
            
        # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1
            W[w_11][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11][col] += 1
            V[v_12][col] += 1
            V[v_22][col] += -1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22][col] += 1
            V[v_21][col] += 1
            V[v_11][col] += -1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11][col] += 1
            U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1
            W[w_01][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21][col] += 1
            U[u_11][col] += -1
            V[v_11][col] += 1
            V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12][col] += 1
            U[u_22][col] += -1
            V[v_21][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1
            
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(decomp, tol=1e-9):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def search_2x2x4_rank_13(time_limit):
        """
        Search for a rank 13 decomposition of the 2x2x4 tensor.
        Returns (U, V, W) of shapes (4,13), (8,13), (8,13) or None.
        """
        import random
        
        n2, m2, p2 = 2, 2, 4
        U_size = 4
        V_size = 8
        W_size = 8
        target_rank = 13
        
        def verify_2x2x4(U, V, W, tol=1e-9):
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            r = Ua.shape[1]
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        total = 0.0
                        for c in range(r):
                            total += Ua[i * m2 + j, c] * Va[j * p2 + k, c] * Wa[k * n2 + i, c]
                        if abs(total - 1.0) > tol:
                            return False
            return True
        
        # Initialize with 2 Strassen blocks (rank 14)
        def strassen_block_2x2x2(k0):
            terms = []
            v00 = k0
            v01 = k0 + 1
            v10 = 4 + k0
            v11 = 4 + k0 + 1
            w00 = 2 * k0
            w01 = 2 * k0 + 1
            w10 = 2 * k0 + 2
            w11 = 2 * k0 + 3
            
            def make_vec(size, indices):
                vec = [0] * size
                for idx, val in indices:
                    vec[idx] = val
                return vec
            
            u1 = make_vec(4, [(0,1),(3,1)])
            v1 = make_vec(8, [(v00,1),(v11,1)])
            w1 = make_vec(8, [(w00,1),(w11,1)])
            terms.append((u1, v1, w1))
            
            u2 = make_vec(4, [(2,1),(3,1)])
            v2 = make_vec(8, [(v00,1)])
            w2 = make_vec(8, [(w10,1),(w11,-1)])
            terms.append((u2, v2, w2))
            
            u3 = make_vec(4, [(0,1)])
            v3 = make_vec(8, [(v01,1),(v11,-1)])
            w3 = make_vec(8, [(w01,1),(w11,1)])
            terms.append((u3, v3, w3))
            
            u4 = make_vec(4, [(3,1)])
            v4 = make_vec(8, [(v10,1),(v00,-1)])
            w4 = make_vec(8, [(w00,1),(w10,1)])
            terms.append((u4, v4, w4))
            
            u5 = make_vec(4, [(0,1),(1,1)])
            v5 = make_vec(8, [(v11,1)])
            w5 = make_vec(8, [(w00,-1),(w01,1)])
            terms.append((u5, v5, w5))
            
            u6 = make_vec(4, [(2,1),(0,-1)])
            v6 = make_vec(8, [(v00,1),(v01,1)])
            w6 = make_vec(8, [(w11,1)])
            terms.append((u6, v6, w6))
            
            u7 = make_vec(4, [(1,1),(3,-1)])
            v7 = make_vec(8, [(v10,1),(v11,1)])
            w7 = make_vec(8, [(w00,1)])
            terms.append((u7, v7, w7))
            
            return terms
        
        U_init = []
        V_init = []
        W_init = []
        for k0 in [0, 2]:
            terms = strassen_block_2x2x2(k0)
            for u, v, w in terms:
                U_init.append(u[:])
                V_init.append(v[:])
                W_init.append(w[:])
        
        # Now we have 14 terms. We want to reduce to 13.
        # We can try to remove one term and adjust the others.
        # Or use random search.
        
        start_time = time.time()
        deadline = start_time + time_limit
        
        for trial in range(100):
            if time.time() > deadline:
                break
            
            # Try to remove one term from the 14-term solution and see if we can fix it
            # This is a heuristic approach. A proper search would be better.
            # For now, we'll just return None if we don't find a quick solution.
            # The probability of finding a rank 13 solution via random search is low.
            # We'll rely on the rank 55 solution.
            
        return None

    # Try to find a lower rank decomposition
    result_55 = make_rank_55()
    if result_55 is None:
        return make_schoolbook()
    
    if not verify(result_55):
        return make_schoolbook()
    
    # Attempt to find rank 54 solution
    # This requires a rank 13 solution for 2x2x4
    remaining_time = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining_time > 5:
        sol_13 = search_2x2x4_rank_13(remaining_time - 2)
        if sol_13 is not None:
            U_13, V_13, W_13 = sol_13
            # Construct full rank 54 solution
            rank = 54
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            # Part A: j=0
            for i in range(n):
                for k in range(p):
                    U[i * m + 0][col] = 1
                    V[0 * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            
            # Part B: j=1,2
            # First 4 columns (k=0..3) use rank 13 solution
            # Map U_13 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> U indices 3,4,5,6
            # Map V_13 indices: 0..7 -> V indices 10..17
            # Map W_13 indices: 0..7 -> W indices 0..7
            
            for c in range(13):
                for u_idx in range(4):
                    U[3 + u_idx][col] = U_13[u_idx][c]
                for v_idx in range(8):
                    V[10 + v_idx][col] = V_13[v_idx][c]
                for w_idx in range(8):
                    W[w_idx][col] = W_13[w_idx][c]
                col += 1
            
            # Remaining 6 columns (k=4..9) use Strassen blocks
            for b in range(3):
                k0 = 4 + 2 * b
                k1 = k0 + 1
                
                u_11 = 0 * m + 1
                u_12 = 0 * m + 2
                u_21 = 1 * m + 1
                u_22 = 1 * m + 2
                
                v_11 = 1 * p + k0
                v_12 = 1 * p + k1
                v_21 = 2 * p + k0
                v_22 = 2 * p + k1
                
                w_00 = k0 * n + 0
                w_01 = k1 * n + 0
                w_10 = k0 * n + 1
                w_11 = k1 * n + 1
                
                # M1
                U[u_11][col] += 1
                U[u_22][col] += 1
                V[v_11][col] += 1
                V[v_22][col] += 1
                W[w_00][col] += 1
                W[w_11][col] += 1
                col += 1
                
                # M2
                U[u_21][col] += 1
                U[u_22][col] += 1
                V[v_11][col] += 1
                W[w_10][col] += 1
                W[w_11][col] += -1
                col += 1
                
                # M3
                U[u_11][col] += 1
                V[v_12][col] += 1
                V[v_22][col] += -1
                W[w_01][col] += 1
                W[w_11][col] += 1
                col += 1
                
                # M4
                U[u_22][col] += 1
                V[v_21][col] += 1
                V[v_11][col] += -1
                W[w_00][col] += 1
                W[w_10][col] += 1
                col += 1
                
                # M5
                U[u_11][col] += 1
                U[u_12][col] += 1
                V[v_22][col] += 1
                W[w_00][col] += -1
                W[w_01][col] += 1
                col += 1
                
                # M6
                U[u_21][col] += 1
                U[u_11][col] += -1
                V[v_11][col] += 1
                V[v_12][col] += 1
                W[w_11][col] += 1
                col += 1
                
                # M7
                U[u_12][col] += 1
                U[u_22][col] += -1
                V[v_21][col] += 1
                V[v_22][col] += 1
                W[w_00][col] += 1
                col += 1
            
            if col == rank and verify({"U": U, "V": V, "W": W}):
                return {"U": U, "V": V, "W": W}
    
    return result_55

# EVOLVE-BLOCK-END
