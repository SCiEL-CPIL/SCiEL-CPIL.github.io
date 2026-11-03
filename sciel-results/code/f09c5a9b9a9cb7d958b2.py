import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110
    
    def make_schoolbook(rank):
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
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        for row in mat:
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) > 1e-6:
                    return False
        return True

    def find_3x3x3_strassen():
        """
        Find a rank-7 decomposition for 3x3x3 matrix multiplication.
        Returns (U, V, W) as lists or None.
        """
        n3, m3, p3 = 3, 3, 3
        target_rank = 7
        
        # Try random search
        for attempt in range(500):
            if time.time() > deadline - 20:
                break
                
            U = np.zeros((n3 * m3, target_rank))
            V = np.zeros((m3 * p3, target_rank))
            
            # Use sparse random initialization with +/- 1
            for r in range(target_rank):
                # Randomly select rows for U and V
                # Strassen structures are somewhat sparse
                u_rows = random.sample(range(n3 * m3), 3)
                v_rows = random.sample(range(m3 * p3), 3)
                
                for row in u_rows:
                    U[row, r] = random.choice([-1, 1])
                for row in v_rows:
                    V[row, r] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p3 * n3, target_rank))
            valid = True
            
            for i in range(n3):
                for k in range(p3):
                    A_ik = np.zeros(target_rank)
                    for j in range(m3):
                        A_ik += U[i * m3 + j] * V[j * p3 + k]
                    
                    denom = np.dot(A_ik, A_ik)
                    if abs(denom) < 1e-10:
                        valid = False
                        break
                    
                    w = A_ik / denom
                    W[k * n3 + i] = w
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify
            is_valid = True
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_list[i * m3 + j][r] * V_list[j * p3 + k][r] * W_list[k * n3 + i][r]
                        if abs(val - 1.0) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Check half-integer
                if check_half_integer(U_list) and check_half_integer(V_list) and check_half_integer(W_list):
                    return U_list, V_list, W_list
                else:
                    # Try rounding
                    U_rounded = [[round(val * 2) / 2 for val in row] for row in U_list]
                    V_rounded = [[round(val * 2) / 2 for val in row] for row in V_list]
                    W_rounded = [[round(val * 2) / 2 for val in row] for row in W_list]
                    
                    # Re-verify rounded
                    is_valid_rounded = True
                    for i in range(n3):
                        for j in range(m3):
                            for k in range(p3):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U_rounded[i * m3 + j][r] * V_rounded[j * p3 + k][r] * W_rounded[k * n3 + i][r]
                                if abs(val - 1.0) > 1e-9:
                                    is_valid_rounded = False
                                    break
                            if not is_valid_rounded:
                                break
                        if not is_valid_rounded:
                            break
                    
                    if is_valid_rounded:
                        return U_rounded, V_rounded, W_rounded
        
        return None

    # Try to build the composite decomposition
    strassen_decomp = find_3x3x3_strassen()
    
    if strassen_decomp:
        U3, V3, W3 = strassen_decomp
        rank_333 = 7
        rank_332 = 18
        total_rank = rank_333 + rank_332
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Part 1: 3x3x3 (Strassen)
        # U: 9x7
        for i in range(9):
            for r in range(rank_333):
                U[i][r] = U3[i][r]
        
        # V: 15x7
        # Rows 0-8 correspond to B[0:3, 0:3]
        for i in range(9):
            for r in range(rank_333):
                V[i][r] = V3[i][r]
        # Rows 9-14 are 0
        
        # W: 15x7
        # Rows 0-8 correspond to C[0:3, 0:3]
        for i in range(9):
            for r in range(rank_333):
                W[i][r] = W3[i][r]
        # Rows 9-14 are 0
        
        # Part 2: 3x3x2 (Schoolbook)
        col_offset = rank_333
        for i in range(n):
            for j in range(m):
                for k in range(2): # k is local index for the 2 columns
                    global_k = 3 + k
                    U[i * m + j][col_offset] = 1
                    V[j * p + global_k][col_offset] = 1
                    W[global_k * n + i][col_offset] = 1
                    col_offset += 1
        
        # Verify the full decomposition
        if verify_decomposition(U, V, W, total_rank):
            if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
    
    return best_result
