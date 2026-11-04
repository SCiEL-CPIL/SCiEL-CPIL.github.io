"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen():
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: i=0, schoolbook (32 terms)
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
        
        # Part 2: i=1,2 using Strassen on 2x4x8 blocks
        # Split into 4 sub-blocks of 2x2x2 (j_block in {0,1}, k_block in {0,1,2,3})
        # Each 2x2x2 uses rank 7, total 4*7=28, plus 32 schoolbook = 60? No, 4*2*7=56+32=88
        
        # Standard Strassen 2x2x2 matrices (7 rank terms, 4 positions each)
        # Position l = i_prime * 2 + k_prime for W, i_prime*2+j_prime for U, j_prime*2+k_prime for V
        strassen_U = [
            [1, 0, 0, 1],    # M1: (A00+A11)
            [0, 0, 1, 1],    # M2: (A10+A11)
            [1, 0, 0, 0],    # M3: A00
            [0, 0, 0, 1],    # M4: A11
            [1, 1, 0, 0],    # M5: (A00+A01)
            [-1, 0, 1, 0],   # M6: (A10-A00)
            [0, 1, 0, -1]    # M7: (A01-A11)
        ]
        strassen_V = [
            [1, 0, 0, 1],    # M1: (B00+B11)
            [1, 0, 0, 0],    # M2: B00
            [0, 1, 0, -1],   # M3: (B01-B11)
            [-1, 0, 1, 0],   # M4: (B10-B00)
            [0, 0, 0, 1],    # M5: B11
            [1, 1, 0, 0],    # M6: (B00+B01)
            [0, 0, 1, 1]     # M7: (B10+B11)
        ]
        strassen_W = [
            [1, 0, 0, 1],    # M1: C00, C11
            [0, 0, 1, -1],   # M2: C10, -C11
            [0, 1, 0, 1],    # M3: C01, C11
            [1, 0, 1, 0],    # M4: C00, C10
            [-1, 1, 0, 0],   # M5: -C00, C01
            [0, 0, 0, 1],    # M6: C11
            [1, 0, 0, 0]     # M7: C00
        ]
        
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                for t in range(7):
                    # U: positions (i_prime, j_prime) where l = i_prime*2 + j_prime
                    for l in range(4):
                        if strassen_U[t][l] != 0:
                            local_i_prime = l // 2
                            local_j_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_j = j_start + local_j_prime
                            U[global_i * m + global_j][col] += strassen_U[t][l]
                    
                    # V: positions (j_prime, k_prime) where l = j_prime*2 + k_prime
                    for l in range(4):
                        if strassen_V[t][l] != 0:
                            local_j_prime = l // 2
                            local_k_prime = l % 2
                            global_j = j_start + local_j_prime
                            global_k = k_start + local_k_prime
                            V[global_j * p + global_k][col] += strassen_V[t][l]
                    
                    # W: positions (i_prime, k_prime) where l = i_prime*2 + k_prime
                    for l in range(4):
                        if strassen_W[t][l] != 0:
                            local_i_prime = l // 2
                            local_k_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_k = k_start + local_k_prime
                            W[global_k * n + global_i][col] += strassen_W[t][l]
                    
                    col += 1
        
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"], dtype=np.float64)
        V = np.array(decomp["V"], dtype=np.float64)
        W = np.array(decomp["W"], dtype=np.float64)
        rank = U.shape[1]
        
        # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # The matrix multiplication tensor should be all 1s
        # T[i,j,k] = 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-9):
            # Find where it fails
            diff = np.abs(T - 1.0)
            max_diff = np.max(diff)
            if max_diff > 1e-9:
                idx = np.unravel_index(np.argmax(diff), T.shape)
                return False
        return True

    if time.time() < deadline - 10:
        res = build_strassen()
        if res and verify(res):
            return res
        # If Strassen fails, try to debug by checking which terms are wrong
        # For now, fall back to schoolbook
        if time.time() < deadline - 5:
            # Try a different partition: 2 rows schoolbook + 1 row Strassen
            # 2*4*8 = 64 schoolbook + 1*4*8/4*7... no, let's just use schoolbook
            pass
    
    return build_schoolbook()
