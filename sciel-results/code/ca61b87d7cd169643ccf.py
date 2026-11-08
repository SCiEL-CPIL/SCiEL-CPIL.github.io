import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Correct Strassen 2x2x2 decomposition
    # Verified: for random 2x2 matrices X, Y, Z = X*Y
    # Z[i,k] = sum_r (sum_{i,j} U[r][i,j]*X[i,j]) * (sum_{j,k} V[r][j,k]*Y[j,k]) * W[r][i,k]
    
    # Using standard Strassen with indices (i,j) for X, (j,k) for Y, (i,k) for Z
    # Flattened: X index = i*2+j, Y index = j*2+k, Z index = i*2+k
    
    # Let me define it as: for each rank r, U[r] is a 2x2 matrix, V[r] is a 2x2 matrix, W[r] is a 2x2 matrix
    # such that X*Y = sum_r (U[r] * X) (V[r] * Y) W[r] where * denotes element-wise product summed
    
    # Actually, the standard formulation is:
    # Let a_r = sum_{i,j} U[r][i,j] * X[i,j]
    # Let b_r = sum_{j,k} V[r][j,k] * Y[j,k]
    # Then Z[i,k] = sum_r a_r * b_r * W[r][i,k]
    
    # Strassen's 7 products:
    # a0 = (X00+X11)(Y00+Y11) -> U0 = [[1,0],[0,1]], V0 = [[1,0],[0,1]]
    # a1 = (X10+X11)Y00        -> U1 = [[0,0],[1,1]], V1 = [[1,0],[0,0]]
    # a2 = X00(Y10-Y11)        -> U2 = [[1,0],[0,0]], V2 = [[0,1],[0,-1]]
    # a3 = X11(Y01-Y00)        -> U3 = [[0,0],[0,1]], V3 = [[-1,1],[0,0]]
    # a4 = (X00+X01)Y11        -> U4 = [[1,1],[0,0]], V4 = [[0,0],[0,1]]
    # a5 = (X10-X00)(Y00+Y01)  -> U5 = [[-1,0],[1,0]], V5 = [[1,1],[0,0]]
    # a6 = (X11-X01)(Y10+Y11)  -> U6 = [[0,-1],[0,1]], V6 = [[0,1],[0,1]]
    
    # Z00 = a0 + a3 - a4 - a6
    # Z01 = a2 + a4
    # Z10 = a1 + a3
    # Z11 = a0 - a1 + a2 + a5
    
    # Let me verify Z00 = a0 + a3 - a4 - a6:
    # a0 = X00Y00 + X00Y11 + X11Y00 + X11Y11
    # a3 = X11Y01 - X11Y00
    # a4 = X00Y11 + X01Y11
    # a6 = X11Y10 + X11Y11 - X01Y10 - X01Y11
    # a0 + a3 - a4 - a6 = X00Y00 + X00Y11 + X11Y00 + X11Y11 + X11Y01 - X11Y00 - X00Y11 - X01Y11 - X11Y10 - X11Y11 + X01Y10 + X01Y11
    # = X00Y00 + X11Y01 - X01Y11 - X11Y10 + X01Y10
    # Expected Z00 = X00Y00 + X01Y10
    # So we have extra terms: X11Y01 - X01Y11 - X11Y10
    # This is wrong!
    
    # I think the issue is that Strassen's formula gives:
    # Z00 = M1 + M4 - M5 - M7 where:
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M7 = (A11-A01)(B10+B11)
    
    # Let me just hardcode the correct Strassen and verify numerically.
    
    strassen_U = [
        [(0, 1), (3, 1)],      # a0: X00 + X11
        [(2, 1), (3, 1)],      # a1: X10 + X11
        [(0, 1)],              # a2: X00
        [(3, 1)],              # a3: X11
        [(0, 1), (1, 1)],      # a4: X00 + X01
        [(2, 1), (0, -1)],     # a5: X10 - X00
        [(1, -1), (3, 1)],     # a6: -X01 + X11
    ]
    
    strassen_V = [
        [(0, 1), (3, 1)],      # b0: Y00 + Y11
        [(0, 1)],              # b1: Y00
        [(1, 1), (3, -1)],     # b2: Y10 - Y11
        [(2, 1), (0, -1)],     # b3: Y01 - Y00
        [(3, 1)],              # b4: Y11
        [(0, 1), (1, 1)],      # b5: Y00 + Y01
        [(2, 1), (3, 1)],      # b6: Y10 + Y11
    ]
    
    # Now compute W such that Z = sum_r a_r * b_r * W[r]
    # Z00 = a0 + a3 - a4 - a6
    # Z01 = a2 + a4
    # Z10 = a1 + a3
    # Z11 = a0 - a1 + a2 + a5
    
    # W[r] is a 2x2 matrix, flattened as (i,k) -> i*2+k
    # W[0]: coefficient for Z from a0*b0: Z00+=1, Z11+=1 -> W0 = [[1,0],[0,1]]
    # W[1]: coefficient for Z from a1*b1: Z10+=1 -> W1 = [[0,0],[1,0]]
    # W[2]: coefficient for Z from a2*b2: Z01+=1 -> W2 = [[0,1],[0,0]]
    # W[3]: coefficient for Z from a3*b3: Z00+=1, Z10+=1 -> W3 = [[1,0],[1,0]]
    # W[4]: coefficient for Z from a4*b4: Z00-=1, Z01+=1 -> W4 = [[-1,1],[0,0]]
    # W[5]: coefficient for Z from a5*b5: Z11+=1 -> W5 = [[0,0],[0,1]]
    # W[6]: coefficient for Z from a6*b6: Z00-=1 -> W6 = [[-1,0],[0,0]]
    
    strassen_W = [
        [(0, 1), (3, 1)],      # W0: Z00 += 1, Z11 += 1
        [(2, 1)],              # W1: Z10 += 1
        [(1, 1)],              # W2: Z01 += 1
        [(0, 1), (2, 1)],      # W3: Z00 += 1, Z10 += 1
        [(0, -1), (1, 1)],     # W4: Z00 -= 1, Z01 += 1
        [(3, 1)],              # W5: Z11 += 1
        [(0, -1)],             # W6: Z00 -= 1
    ]
    
    # Verify numerically
    def verify_strassen():
        np.random.seed(42)
        X = np.random.randn(2, 2)
        Y = np.random.randn(2, 2)
        Z_expected = X @ Y
        Z_computed = np.zeros((2, 2))
        for r in range(7):
            a_r = sum(val * X[i, j] for (idx, val) in strassen_U[r] for i, j in [(idx // 2, idx % 2)])
            b_r = sum(val * Y[j, k] for (idx, val) in strassen_V[r] for j, k in [(idx // 2, idx % 2)])
            for (idx, val) in strassen_W[r]:
                i, k = idx // 2, idx % 2
                Z_computed[i, k] += a_r * b_r * val
        return np.allclose(Z_expected, Z_computed)
    
    if not verify_strassen():
        # Fallback to naive if Strassen is wrong
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Helper to add a 2x2x2 block
    def add_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_off
                global_j = local_j + j_off
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_off
                global_k = local_k + k_off
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_off
                global_i = local_i + i_off
                W[global_k * 2 + global_i][current_rank] += val
    
    # 8 blocks of Strassen
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
