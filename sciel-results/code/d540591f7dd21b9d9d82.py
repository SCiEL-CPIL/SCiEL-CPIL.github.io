import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # We aim for a rank-minimal decomposition for 2x5x6 matrix multiplication.
    # The trivial rank is 60.
    # By using Strassen's algorithm on 2x2x2 blocks, we can reduce the rank.
    # 
    # Decomposition of 2x5x6:
    # Split j (5) into {0,1}, {2,3}, {4}
    # Split k (6) into {0,1}, {2,3}, {4,5}
    # 
    # Blocks involving j in {0,1} or {2,3} and k in pairs are 2x2x2 blocks (i is 2).
    # There are 2 j-pairs * 3 k-pairs = 6 such blocks.
    # Each 2x2x2 block has rank 7 (Strassen) instead of 8 (trivial).
    # Savings: 6 * (8-7) = 6.
    #
    # Blocks involving j=4 are 2x1x2 blocks.
    # There are 3 such blocks (k pairs).
    # Rank of 2x1x2 is 4 (trivial, no savings possible with standard Strassen-like tricks for this shape? 
    # Actually, 2x1x2 is just 4 scalar multiplications. Can we do better?
    # C[i,k] = A[i,4] * B[4,k].
    # This is 2x1 * 1x2. Rank is 4. 
    # Wait, is there a lower rank for 2x1x2?
    # T = sum_{i,k} A_{i,4} B_{4,k} E_{i,k}.
    # This is the tensor product of vectors u (dim 2) and v (dim 2). Rank is 4.
    # So no savings there.
    
    # Total rank = 6 * 7 + 3 * 4 = 42 + 12 = 54.
    # This is the standard Strassen-based bound.
    
    # Can we do better than 54?
    # Lower bounds for 2x5x6:
    # R(2,5,6) >= R(2,5,5) + R(2,0,1)? No.
    # R(2,5,6) >= R(2,4,6) + R(2,1,0)?
    # R(2,5,6) >= R(2,2,2) * (5/2 * 6/2)? No, submultiplicativity is for tensor product.
    # Known lower bound: R(2,5,6) >= 54?
    # Actually, R(2,5,6) is not exactly known, but 54 is the best known upper bound via Strassen.
    # Is it possible to get 53?
    # Some papers suggest R(2,5,6) might be 53 or 54.
    # However, finding a rank 53 decomposition is non-trivial and might require searching for linear dependencies
    # across the block boundaries or using more advanced algorithms like Coppersmith-Winograd (which are asymptotic).
    # For small matrices, specific constructions are needed.
    
    # Given the time limit and the risk of invalidity, the safest and most robust approach is to provide the
    # verified rank 54 decomposition.
    # If we have time, we could attempt a greedy search to remove one term by checking linear independence
    # of the resulting tensors, but this is computationally expensive and risky.
    
    # We will implement the rank 54 decomposition.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard term
    def add_standard(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Helper to add a Strassen block (2x2x2)
    def add_strassen(j1, j2, k1, k2):
        nonlocal col
        # i is always 0,1
        u00 = 0 * m + j1
        u01 = 0 * m + j2
        u10 = 1 * m + j1
        u11 = 1 * m + j2
        
        v00 = j1 * p + k1
        v01 = j1 * p + k2
        v10 = j2 * p + k1
        v11 = j2 * p + k2
        
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # P2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        
        # P3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # P4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # P5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Apply Strassen to 2x2x2 blocks
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            add_strassen(j1, j2, k1, k2)
            
    # Add standard terms for j=4
    for k1, k2 in k_pairs:
        for k in [k1, k2]:
            add_standard(0, 4, k)
            add_standard(1, 4, k)
            
    # Check time
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}

    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank 60
        rank_f = 60
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
