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
    
    # Laderman rank-23 decomposition for 3x3x3
    def laderman_3x3x3():
        terms = [
            ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
            ([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
            ([0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
            ([0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
            ([0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
            ([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0]),
            ([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]),
            ([0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1]),
            ([0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0]),
            ([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0]),
            ([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0]),
            ([1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0]),
            ([0,0,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0]),
            ([1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]),
            ([0,1,1,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0]),
            ([0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0,1]),
            ([1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1]),
        ]
        
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
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
        
        # Strategy 1: Two Laderman blocks (k=0-2, k=3-5) + schoolbook (k=6,7)
        # Rank: 23 + 23 + 18 = 64
        combined_rank = 23 + 23 + 18  # 64
        U_comb = np.zeros((9, combined_rank))
        V_comb = np.zeros((24, combined_rank))
        W_comb = np.zeros((24, combined_rank))
        
        # Block 1: k=0,1,2 using Laderman
        for c in range(23):
            for i in range(3):
                for j in range(3):
                    U_comb[i*3+j, c] = U_lad[i*3+j, c]
            for j in range(3):
                for k in range(3):
                    V_comb[j*8+k, c] = V_lad[j*3+k, c]
            for k in range(3):
                for i in range(3):
                    W_comb[k*3+i, c] = W_lad[k*3+i, c]
        
        # Block 2: k=3,4,5 using Laderman (same U, shifted V and W)
        for c in range(23):
            for i in range(3):
                for j in range(3):
                    U_comb[i*3+j, 23+c] = U_lad[i*3+j, c]
            for j in range(3):
                for kp in range(3):  # kp=0,1,2 corresponds to k=3,4,5
                    V_comb[j*8+(3+kp), 23+c] = V_lad[j*3+kp, c]
            for kp in range(3):
                for i in range(3):
                    W_comb[(3+kp)*3+i, 23+c] = W_lad[kp*3+i, c]
        
        # Block 3: k=6,7 using schoolbook
        col = 46
        for i in range(3):
            for j in range(3):
                for k in range(6, 8):
                    U_comb[i*3+j, col] = 1
                    V_comb[j*8+k, col] = 1
                    W_comb[k*3+i, col] = 1
                    col += 1
        
        if validate(U_comb.tolist(), V_comb.tolist(), W_comb.tolist()):
            best_U, best_V, best_W, best_rank = U_comb.tolist(), V_comb.tolist(), W_comb.tolist(), combined_rank
        else:
            # Fallback: One Laderman block + schoolbook
            combined_rank_fb = 23 + 45  # 68
            U_comb_fb = np.zeros((9, combined_rank_fb))
            V_comb_fb = np.zeros((24, combined_rank_fb))
            W_comb_fb = np.zeros((24, combined_rank_fb))
            
            for c in range(23):
                for i in range(3):
                    for j in range(3):
                        U_comb_fb[i*3+j, c] = U_lad[i*3+j, c]
                for j in range(3):
                    for k in range(3):
                        V_comb_fb[j*8+k, c] = V_lad[j*3+k, c]
                for k in range(3):
                    for i in range(3):
                        W_comb_fb[k*3+i, c] = W_lad[k*3+i, c]
            
            col = 23
            for i in range(3):
                for j in range(3):
                    for k in range(3, 8):
                        U_comb_fb[i*3+j, col] = 1
                        V_comb_fb[j*8+k, col] = 1
                        W_comb_fb[k*3+i, col] = 1
                        col += 1
            
            if validate(U_comb_fb.tolist(), V_comb_fb.tolist(), W_comb_fb.tolist()):
                best_U, best_V, best_W, best_rank = U_comb_fb.tolist(), V_comb_fb.tolist(), W_comb_fb.tolist(), combined_rank_fb
        
        # Strategy 2: Try to reduce rank by merging schoolbook terms with Laderman
        # Check if we can eliminate redundant terms from the rank-64 solution
        if elapsed() < deadline - 10 and best_rank == 64:
            U_arr = np.array(best_U, dtype=float)
            V_arr = np.array(best_V, dtype=float)
            W_arr = np.array(best_W, dtype=float)
            rank = 64
            
            # Try removing each term and see if the decomposition still works
            # This is a greedy sparsification approach
            for trial in range(200):
                if elapsed() > deadline - 10:
                    break
                
                # Try to remove a random term
                col_to_try = random.randint(0, rank - 1)
                
                # Check if this term is redundant
                U_test = U_arr.copy()
                V_test = V_arr.copy()
                W_test = W_arr.copy()
                U_test[:, col_to_try] = 0
                V_test[:, col_to_try] = 0
                W_test[:, col_to_try] = 0
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_test[i*m+j], V_test[j*p+k] * W_test[k*n+i])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Remove this term
                    mask = np.ones(rank, dtype=bool)
                    mask[col_to_try] = False
                    U_arr = U_arr[:, mask]
                    V_arr = V_arr[:, mask]
                    W_arr = W_arr[:, mask]
                    rank -= 1
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
