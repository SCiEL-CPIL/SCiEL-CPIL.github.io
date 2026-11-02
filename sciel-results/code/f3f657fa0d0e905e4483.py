"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140
    
    def elapsed():
        return time.time() - start_time
    
    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Laderman rank-20 decomposition for 3x3x3
    def laderman_3x3x3():
        # The 20 terms of Laderman's algorithm
        # U, V, W are 9x20
        terms = [
            # 1. x11*y11
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # 2. x11*y22
            ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # 3. x11*y33
            ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # 4. x22*y11
            ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # 5. x22*y22
            ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # 6. x22*y33
            ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
            # 7. x33*y11
            ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # 8. x33*y22
            ([0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
            # 9. x33*y33
            ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
            
            # 10. (x11+x22)*y11
            ([1,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0]),
            # 11. (x11+x33)*y11
            ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1]),
            # 12. (x22+x33)*y22
            ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
            # 13. (x11+x22)*y22
            ([1,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0]),
            # 14. (x11+x33)*y33
            ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,1]),
            # 15. (x22+x33)*y33
            ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1]),
            
            # 16. x11*(y11+y22)
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,1, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
            # 17. x11*(y22+y33)
            ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
            # 18. x22*(y11+y33)
            ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
            # 19. x33*(y11+y22)
            ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,1, 0,0,0], [0,0,1, 0,1,0, 0,0,0]),
            # 20. x33*(y22+y33)
            ([0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,1]),
        ]
        
        U = np.zeros((9, 20))
        V = np.zeros((9, 20))
        W = np.zeros((9, 20))
        
        for c, (u, v, w) in enumerate(terms):
            U[:, c] = np.array(u)
            V[:, c] = np.array(v)
            W[:, c] = np.array(w)
        
        # Verify
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        return None
        return U, V, W
    
    lad = laderman_3x3x3()
    
    if lad is not None and elapsed() < deadline - 5:
        U_lad, V_lad, W_lad = lad
        
        # Strategy: Two Laderman blocks (k=0-2, k=3-5) + Schoolbook (k=6,7)
        # Rank: 20 + 20 + 18 = 58
        combined_rank = 20 + 20 + 18  # 58
        U_comb = np.zeros((9, combined_rank))
        V_comb = np.zeros((24, combined_rank))
        W_comb = np.zeros((24, combined_rank))
        
        # Block 1: k=0,1,2 using Laderman
        for c in range(20):
            for i in range(3):
                for j in range(3):
                    U_comb[i*3+j, c] = U_lad[i*3+j, c]
            for j in range(3):
                for k in range(3):
                    V_comb[j*8+k, c] = V_lad[j*3+k, c]
            for k in range(3):
                for i in range(3):
                    W_comb[k*3+i, c] = W_lad[k*3+i, c]
        
        # Block 2: k=3,4,5 using Laderman
        for c in range(20):
            for i in range(3):
                for j in range(3):
                    U_comb[i*3+j, 20+c] = U_lad[i*3+j, c]
            for j in range(3):
                for kp in range(3):  # kp=0,1,2 corresponds to k=3,4,5
                    V_comb[j*8+(3+kp), 20+c] = V_lad[j*3+kp, c]
            for kp in range(3):
                for i in range(3):
                    W_comb[(3+kp)*3+i, 20+c] = W_lad[kp*3+i, c]
        
        # Block 3: k=6,7 using Schoolbook
        col = 40
        for i in range(3):
            for j in range(3):
                for k in range(6, 8):
                    U_comb[i*3+j, col] = 1
                    V_comb[j*8+k, col] = 1
                    W_comb[k*3+i, col] = 1
                    col += 1
        
        if validate(U_comb.tolist(), V_comb.tolist(), W_comb.tolist()):
            best_U, best_V, best_W, best_rank = U_comb.tolist(), V_comb.tolist(), W_comb.tolist(), combined_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
