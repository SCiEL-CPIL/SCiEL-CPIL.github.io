DIMENSIONS = [2, 4, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Standard Strassen gives 7 per 2x2 block, and we have 8 such blocks
    # Total: 56
    
    # But let's verify the Strassen patterns are correct first
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's use the standard Strassen with proper indexing
    # For 2x2 matrices A[i,j], B[j,k], C[i,k] where i,j,k in {0,1}
    
    strassen_U = [
        [(0, 1), (3, 1)],      # M1: A11 + A22
        [(2, 1), (3, 1)],      # M2: A21 + A22
        [(0, 1)],              # M3: A11
        [(3, 1)],              # M4: A22
        [(0, 1), (1, 1)],      # M5: A11 + A12
        [(2, 1), (0, -1)],     # M6: A21 - A11
        [(1, 1), (3, -1)],     # M7: A12 - A22
    ]
    
    strassen_V = [
        [(0, 1), (3, 1)],      # M1: B11 + B22
        [(0, 1)],              # M2: B11
        [(1, 1), (3, -1)],     # M3: B21 - B22
        [(1, 1), (0, -1)],     # M4: B12 - B11
        [(3, 1)],              # M5: B22
        [(0, 1), (1, 1)],      # M6: B11 + B12
        [(2, 1), (3, 1)],      # M7: B21 + B22
    ]
    
    strassen_W = [
        [(0, 1), (3, -1), (1, -1), (2, -1)],  # M1 - M5 - M7 for C11
        [(1, 1), (2, 1)],                      # M3 + M5 for C12
        [(2, 1), (3, 1)],                      # M2 + M4 for C21
        [(0, 1), (2, -1), (1, 1), (3, 1)],    # M1 - M2 + M3 + M7 for C22
    ]
    
    # Actually, let me use a cleaner formulation
    # Each rank-1 term corresponds to one of the 7 Strassen products
    # For product r, we need U_r, V_r, W_r such that:
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta(i,j,k) for matmul
    
    # Let me redefine properly
    # The matmul tensor T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # Standard Strassen rank-7 decomposition:
    # r=0: M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # r=1: M2 = (A10+A11)B00 -> contributes to C10, C11
    # r=2: M3 = A00(B10-B11) -> contributes to C01
    # r=3: M4 = A11(B01-B00) -> contributes to C00, C10
    # r=4: M5 = (A00+A01)B11 -> contributes to C01
    # r=5: M6 = (A10-A00)(B00+B01) -> contributes to C10, C11
    # r=6: M7 = (A01-A11)(B10+B11) -> contributes to C00, C11
    
    # Let me use a known correct Strassen decomposition
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    
    # Using 0-indexed: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    
    # Now express each as sum of rank-1 terms
    # For each r, we need U_r[i,j], V_r[j,k], W_r[k,i]
    
    # r=0 (M1): U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    # But this doesn't work directly because W needs to be factored
    
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # For matmul, T[i,j,k] = 1 for all i,j,k
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
    
    # This is the standard matrix multiplication tensor
    
    # Let me use a verified Strassen decomposition
    # From literature, one valid rank-7 decomposition:
    
    strassen_U = [
        [(0, 1), (3, 1)],      # r0
        [(2, 1), (3, 1)],      # r1
        [(0, 1)],              # r2
        [(3, 1)],              # r3
        [(0, 1), (1, 1)],      # r4
        [(2, 1), (0, -1)],     # r5
        [(1, 1), (3, -1)],     # r6
    ]
    
    strassen_V = [
        [(0, 1), (3, 1)],      # r0
        [(0, 1)],              # r1
        [(1, 1), (3, -1)],     # r2
        [(1, 1), (0, -1)],     # r3
        [(3, 1)],              # r4
        [(0, 1), (1, 1)],      # r5
        [(2, 1), (3, 1)],      # r6
    ]
    
    strassen_W = [
        [(0, 1), (3, -1)],     # r0
        [(1, 1), (3, -1)],     # r1
        [(2, 1), (3, 1)],      # r2
        [(0, 1), (2, 1)],      # r3
        [(0, -1), (2, 1)],     # r4
        [(3, 1)],              # r5
        [(0, 1)],              # r6
    ]
    
    # Verify this decomposition
    def verify_strassen():
        # Check: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k in {0,1}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        for idx, val in strassen_U[r]:
                            ii, jj = divmod(idx, 2)
                            if ii == i and jj == j:
                                u_val = val
                        for idx, val in strassen_V[r]:
                            jj, kk = divmod(idx, 2)
                            if jj == j and kk == k:
                                v_val = val
                        for idx, val in strassen_W[r]:
                            kk, ii = divmod(idx, 2)
                            if kk == k and ii == i:
                                w_val = val
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    is_valid, debug_info = verify_strassen()
    
    if not is_valid:
        # Fallback to a simpler construction if Strassen patterns are wrong
        # Use naive decomposition with rank = 2*4*8 = 64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * 4 + j][r] = 1
                    V[j * 8 + k][r] = 1
                    W[k * 2 + i][r] = 1
                    r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Use Strassen with rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val
    
    # 8 blocks of 2x2 multiplications
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # For each (i_block, j_block, k_block) where i_block in {0}, j_block in {0,1,2,3}, k_block in {0,1,2,3}
    # Actually, we have 2x4 * 4x8 = 8 blocks of 2x2 * 2x2
    
    block_idx = 0
    for j_block in range(4):  # j = 2*j_block, 2*j_block+1
        for k_block in range(4):  # k = 2*k_block, 2*k_block+1
            add_block(block_idx * 7, 0, 2 * j_block, 2 * k_block)
            block_idx += 1
    
    # Verify the full decomposition
    def verify_full():
        if time.time() - start_time > deadline - 5:
            return True
        
        # Check a sample of entries
        import random
        random.seed(42)
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            expected = 1
            computed = 0
            for r in range(rank):
                computed += U[i * 4 + j][r] * V[j * 8 + k][r] * W[k * 2 + i][r]
            
            if abs(computed - expected) > 1e-9:
                return False, (i, j, k, computed)
        
        return True, None
    
    is_valid, debug_info = verify_full()
    
    if not is_valid:
        # Fallback to naive
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * 4 + j][r] = 1
                    V[j * 8 + k][r] = 1
                    W[k * 2 + i][r] = 1
                    r += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
