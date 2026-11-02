import numpy as np
import time
import itertools

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy:
    1. Recognize 3x3x6 multiplication as two 3x3x3 multiplications (splitting the 6 columns of B into two sets of 3).
    2. Search for a rank-minimal decomposition for 3x3x3 with integer/half-integer entries.
    3. Construct the 3x3x6 decomposition by duplicating the 3x3x3 solution for each block.
    4. If no low-rank 3x3x3 solution is found, return schoolbook 54.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    # Build the target tensor T for 3x3x6
    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    # Search for 3x3x3 decomposition
    # Dimensions for 3x3x3
    n3, m3, p3 = 3, 3, 3
    dim_A3 = n3 * m3 # 9
    dim_B3 = m3 * p3 # 9
    dim_C3 = p3 * n3 # 9

    T3_nonzero = {}
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                a = i * m3 + j
                b = j * p3 + k
                c = k * n3 + i
                T3_nonzero[(a, b, c)] = 1.0

    def find_3x3x3_decomp(R, attempts=5000):
        """
        Searches for a rank R decomposition of 3x3x3 tensor.
        Returns (U, V, W) as numpy arrays or None.
        """
        for _ in range(attempts):
            if time.time() > deadline:
                return None
            
            # Random integer U, V
            U = np.random.randint(-1, 2, size=(dim_A3, R))
            V = np.random.randint(-1, 2, size=(dim_B3, R))
            
            W = np.zeros((dim_C3, R))
            valid = True
            
            for c in range(dim_C3):
                M_c = np.zeros((dim_A3 * dim_B3, R))
                b_c = np.zeros(dim_A3 * dim_B3)
                
                idx = 0
                for a in range(dim_A3):
                    for b in range(dim_B3):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in T3_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                # Check half-integer
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                # Verify
                is_valid = True
                for (a, b, c), val in T3_nonzero.items():
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check some zeros
                    for _ in range(20):
                        a = np.random.randint(0, dim_A3)
                        b = np.random.randint(0, dim_B3)
                        c = np.random.randint(0, dim_C3)
                        if (a, b, c) not in T3_nonzero:
                            s = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                    
                    if is_valid:
                        return U, V, W

        return None

    # Try ranks for 3x3x3
    # Known upper bound is 23? Let's try 21 to 27.
    # If we find R333, total rank is 2 * R333.
    # We want 2 * R333 < 54 => R333 < 27.
    
    found_333 = None
    best_R333 = 27 # Fallback to schoolbook equivalent
    
    for R333 in range(21, 28):
        if time.time() > deadline:
            break
            
        # Allocate more attempts for higher ranks if lower ones fail quickly?
        # Or just fixed attempts.
        # 5000 attempts might be too slow if lstsq is slow.
        # 162xR system. R~25. 162x25. Very fast.
        # 5000 * 18 * (162x25 lstsq) should be fast.
        
        res = find_3x3x3_decomp(R333, attempts=2000)
        if res:
            found_333 = res
            best_R333 = R333
            break # Found the minimal rank in this range
    
    if found_333:
        U3, V3, W3 = found_333
        R_total = 2 * best_R333
        
        # Construct 3x3x6 decomposition
        # U_total is U3 duplicated? No, U is for A.
        # In 3x3x6, A is 3x3. In 3x3x3, A is 3x3.
        # So U_total should be the same for both blocks?
        # T_total = T1 + T2.
        # T1 = sum U_r v_{1,r} w_{1,r}
        # T2 = sum U_r v_{2,r} w_{2,r}
        # We can use the same U_r for both.
        # So U_total = U3 (shape 9 x R333)
        # But wait, the rank is 2*R333.
        # We need to expand the rank dimension.
        
        U_total = np.zeros((dim_A, R_total))
        V_total = np.zeros((dim_B, R_total))
        W_total = np.zeros((dim_C, R_total))
        
        # Block 1: k in 0,1,2
        # U_total[:, 0:R333] = U3
        # V_total[:, 0:R333] corresponds to B1 (cols 0-2 of B)
        # W_total[:, 0:R333] corresponds to C1 (rows 0-8 of C? No, C1 is first 3 rows of each block? No.)
        
        # Let's map indices carefully.
        # 3x3x3 indices: a, b, c in 0..8.
        # 3x3x6 indices:
        # a = i*3 + j
        # b = j*6 + k
        # c = k*3 + i
        
        # For Block 1 (k=0,1,2):
        # b3 = j*3 + k (where k is 0,1,2). This is the first 3 columns of B.
        # c3 = k*3 + i. This is the first 9 rows of C?
        # c = k*3 + i.
        # If k=0, c=i (0,1,2).
        # If k=1, c=3+i (3,4,5).
        # If k=2, c=6+i (6,7,8).
        # So C1 occupies rows 0..8 of C.
        
        # For Block 2 (k=3,4,5):
        # Let k' = k-3 (0,1,2).
        # b = j*6 + (k'+3) = j*6 + 3 + k'.
        # This is the second 3 columns of B.
        # c = (k'+3)*3 + i = 9 + k'*3 + i.
        # If k'=0, c=9+i (9,10,11).
        # If k'=1, c=12+i (12,13,14).
        # If k'=2, c=15+i (15,16,17).
        # So C2 occupies rows 9..17 of C.
        
        # Mapping:
        # U_total[:, 0:R333] = U3
        # U_total[:, R333:2*R333] = U3
        
        # V_total:
        # V3 has indices b3 in 0..8.
        # b3 = j*3 + k3.
        # In Block 1, b = b3.
        # So V_total[b3, 0:R333] = V3[b3, :]
        
        # In Block 2, b = j*6 + 3 + k3.
        # Let b2 = j*6 + 3 + k3.
        # We need V_total[b2, R333:2*R333] = V3[b3, :]
        # where b3 = j*3 + k3.
        
        for j in range(3):
            for k3 in range(3):
                b3 = j*3 + k3
                b1 = j*6 + k3
                b2 = j*6 + 3 + k3
                V_total[b1, 0:R333] = V3[b3, :]
                V_total[b2, R333:2*R333] = V3[b3, :]
                
        # W_total:
        # W3 has indices c3 in 0..8.
        # c3 = k3*3 + i.
        # In Block 1, c = c3.
        # So W_total[c3, 0:R333] = W3[c3, :]
        
        # In Block 2, c = 9 + c3.
        # So W_total[9+c3, R333:2*R333] = W3[c3, :]
        
        for k3 in range(3):
            for i in range(3):
                c3 = k3*3 + i
                c1 = c3
                c2 = 9 + c3
                W_total[c1, 0:R333] = W3[c3, :]
                W_total[c2, R333:2*R333] = W3[c3, :]
        
        # Verify the full 3x3x6 decomposition
        is_valid = True
        for (a, b, c), val in T_nonzero.items():
            s = np.sum(U_total[a, :] * V_total[b, :] * W_total[c, :])
            if abs(s - val) > 1e-9:
                is_valid = False
                break
        
        if is_valid:
            # Check zeros
            for _ in range(100):
                a = np.random.randint(0, dim_A)
                b = np.random.randint(0, dim_B)
                c = np.random.randint(0, dim_C)
                if (a, b, c) not in T_nonzero:
                    s = np.sum(U_total[a, :] * V_total[b, :] * W_total[c, :])
                    if abs(s) > 1e-9:
                        is_valid = False
                        break
                        
            if is_valid:
                U_list = [[int(x) for x in row] for row in U_total.tolist()]
                V_list = [[int(x) for x in row] for row in V_total.tolist()]
                W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W_total.tolist()]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R_total
                return best_result

    return best_result

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    n3, m3, p3 = 3, 3, 3
    dim_A3 = n3 * m3 # 9
    dim_B3 = m3 * p3 # 9
    dim_C3 = p3 * n3 # 9

    T3_nonzero = {}
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                a = i * m3 + j
                b = j * p3 + k
                c = k * n3 + i
                T3_nonzero[(a, b, c)] = 1.0

    def find_3x3x3_decomp(R, attempts=2000):
        for _ in range(attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.randint(-1, 2, size=(dim_A3, R))
            V = np.random.randint(-1, 2, size=(dim_B3, R))
            
            W = np.zeros((dim_C3, R))
            valid = True
            
            for c in range(dim_C3):
                M_c = np.zeros((dim_A3 * dim_B3, R))
                b_c = np.zeros(dim_A3 * dim_B3)
                
                idx = 0
                for a in range(dim_A3):
                    for b in range(dim_B3):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in T3_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                is_valid = True
                for (a, b, c), val in T3_nonzero.items():
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    for _ in range(20):
                        a = np.random.randint(0, dim_A3)
                        b = np.random.randint(0, dim_B3)
                        c = np.random.randint(0, dim_C3)
                        if (a, b, c) not in T3_nonzero:
                            s = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                    
                    if is_valid:
                        return U, V, W

        return None

    found_333 = None
    best_R333 = 27

    for R333 in range(21, 28):
        if time.time() > deadline:
            break
        
        res = find_3x3x3_decomp(R333, attempts=2000)
        if res:
            found_333 = res
            best_R333 = R333
            break
    
    if found_333:
        U3, V3, W3 = found_333
        R_total = 2 * best_R333
        
        U_total = np.zeros((dim_A, R_total))
        V_total = np.zeros((dim_B, R_total))
        W_total = np.zeros((dim_C, R_total))
        
        U_total[:, 0:R333] = U3
        U_total[:, R333:2*R333] = U3
        
        for j in range(3):
            for k3 in range(3):
                b3 = j*3 + k3
                b1 = j*6 + k3
                b2 = j*6 + 3 + k3
                V_total[b1, 0:R333] = V3[b3, :]
                V_total[b2, R333:2*R333] = V3[b3, :]
                
        for k3 in range(3):
            for i in range(3):
                c3 = k3*3 + i
                c1 = c3
                c2 = 9 + c3
                W_total[c1, 0:R333] = W3[c3, :]
                W_total[c2, R333:2*R333] = W3[c3, :]
        
        is_valid = True
        for (a, b, c), val in T_nonzero.items():
            s = np.sum(U_total[a, :] * V_total[b, :] * W_total[c, :])
            if abs(s - val) > 1e-9:
                is_valid = False
                break
        
        if is_valid:
            for _ in range(100):
                a = np.random.randint(0, dim_A)
                b = np.random.randint(0, dim_B)
                c = np.random.randint(0, dim_C)
                if (a, b, c) not in T_nonzero:
                    s = np.sum(U_total[a, :] * V_total[b, :] * W_total[c, :])
                    if abs(s) > 1e-9:
                        is_valid = False
                        break
                        
            if is_valid:
                U_list = [[int(x) for x in row] for row in U_total.tolist()]
                V_list = [[int(x) for x in row] for row in V_total.tolist()]
                W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W_total.tolist()]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R_total
                return best_result

    return best_result
# EVOLVE-BLOCK-END
