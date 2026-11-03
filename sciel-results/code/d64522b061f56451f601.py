import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 8
    dim_U = n * m  # 8
    dim_V = m * p  # 32
    dim_W = p * n  # 16
    
    # The tensor T is defined by T[i, j, k] = 1 if the output index (k, i) depends on A[i,j] and B[j,k].
    # In the decomposition C = A B, C[k, i] = sum_j A[i, j] B[j, k].
    # So the term A[i, j] B[j, k] contributes to C[k, i].
    # The tensor T has indices (u_idx, v_idx, w_idx) where:
    # u_idx corresponds to A[i, j] -> i * m + j
    # v_idx corresponds to B[j, k] -> j * p + k
    # w_idx corresponds to C[k, i] -> k * n + i
    # T[u_idx, v_idx, w_idx] = 1 if u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i.
    # Note: For a fixed (i, j, k), u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i.
    # There is a unique correspondence.
    
    # We want to find rank r such that T = sum_{t=1}^r U[:,t] V[:,t] W[:,t]^T
    # i.e., T[u, v, w] = sum_t U[u, t] V[v, t] W[w, t]
    
    # We know the schoolbook rank is 64.
    # We suspect the rank might be lower.
    # Let's try to find a decomposition of rank 56, 52, etc.
    
    # A simple approach: Random search with integer/half-integer constraints is hard.
    # Let's try a structured approach based on block decomposition.
    
    # The matrix multiplication 2x4 by 4x8 can be seen as:
    # A is 2x4, B is 4x8.
    # We can split B into two 4x4 blocks: B = [B1, B2].
    # Then C = [A B1, A B2].
    # So we need to compute two 2x4 by 4x4 multiplications.
    # If we can do 2x4 by 4x4 with rank R, then total rank is 2R.
    # Schoolbook 2x4x4 is 32. So 2*32 = 64.
    # Can we do 2x4x4 with rank < 32?
    # 2x4x4 can be split into two 2x2x4 multiplications?
    # A = [A1, A2] where A1, A2 are 2x2.
    # B is 4x4. Split B into 2x2 blocks:
    # B = [[B11, B12], [B21, B22]]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # This is not independent.
    
    # Alternative: Use the fact that 2x2x2 has rank 7 (Winograd).
    # 2x2x4 can be done with rank 14?
    # Let A be 2x2, B be 2x4.
    # B = [B1, B2] where B1, B2 are 2x2.
    # C = [A B1, A B2].
    # If we use the same linear combinations for A in both products, we can save rank.
    # Let the Winograd factors for A be U1, U2, ..., U7.
    # For B1, we need V1, V2, ..., V7.
    # For B2, we need V8, V9, ..., V14.
    # Then C1 = sum U_t V_t, C2 = sum U_t V_{t+7}.
    # The W factors are the same for C1 and C2?
    # C is 2x4. C1 is 2x2, C2 is 2x2.
    # The W factors for C1 are W1..W7, for C2 are W8..W14.
    # So total rank is 14.
    # This gives 2x2x4 rank 14.
    
    # Now 2x4x4.
    # A is 2x4. Split into A1, A2 (2x2).
    # B is 4x4. Split into 4 2x2 blocks: B11, B12, B21, B22.
    # C = [[A1 B11 + A2 B21, A1 B12 + A2 B22]]? No.
    # A = [A1, A2] (2x4). B = [[B11, B12], [B21, B22]] (4x4).
    # C = A B = [A1 B11 + A2 B21, A1 B12 + A2 B22] (2x4).
    # Let C1 = A1 B11 + A2 B21, C2 = A1 B12 + A2 B22.
    # This is a sum of two products.
    # We can compute A1 B11, A2 B21, A1 B12, A2 B22.
    # Each is 2x2x2.
    # If we do them independently, rank is 4 * 7 = 28.
    # Then C1 = C11 + C21, C2 = C12 + C22.
    # The W factors for C11 and C21 add up.
    # So we can share the W factors?
    # No, the W factors for C11 and C21 are for the same output positions.
    # So we can just add the W vectors.
    # But the U and V factors are different.
    # So we need 4 sets of 7 U's and 4 sets of 7 V's.
    # Total rank 28.
    # Then 2x4x8 is two 2x4x4's, so rank 56.
    
    # So rank 56 is achievable.
    # Let's construct it.
    
    # Helper to add a 2x2x2 rank 7 block.
    # A indices: a0, a1, a2, a3 (2x2)
    # B indices: b0, b1, b2, b3 (2x2)
    # C indices: c0, c1, c2, c3 (2x2)
    # offset: current column in U, V, W
    
    def add_2x2x2_rank7(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, offset, U, V, W):
        # Winograd 7-term scheme for 2x2x2.
        # P1 = (a0+a1+a2+a3)(b0+b1+b2+b3)
        # P2 = (a0-a1+a2-a3)(b0-b1+b2-b3)
        # P3 = (a0+a1-a2-a3)(b0+b1-b2-b3)
        # P4 = (a0-a1-a2+a3)(b0-b1-b2+b3)
        # P5 = (a0-a1)(b0-b1)
        # P6 = (a2-a3)(b2-b3)
        # P7 = (a1-a3)(b1-b3)
        
        # C00 = (P1+P2+P3+P4)/4 + (P5+P7)/2
        # C01 = (P1-P2+P3-P4)/4 + (P5-P7)/2
        # C10 = (P1+P2-P3-P4)/4 + (P6+P7)/2
        # C11 = (P1-P2-P3+P4)/4 + (P6-P7)/2
        
        # Let's verify this.
        # This is a standard Winograd scheme.
        
        # We will store the coefficients in U, V, W.
        # Since we need integer/half-integer entries, we can multiply everything by 4 to avoid fractions?
        # No, the task allows half-integers.
        # So we can use 1/4, 1/2, etc.
        
        # Term 1
        U[a0][offset] += 1; U[a1][offset] += 1; U[a2][offset] += 1; U[a3][offset] += 1
        V[b0][offset] += 1; V[b1][offset] += 1; V[b2][offset] += 1; V[b3][offset] += 1
        W[c0][offset] += 0.25; W[c1][offset] += 0.25; W[c2][offset] += 0.25; W[c3][offset] += 0.25
        offset += 1
        
        # Term 2
        U[a0][offset] += 1; U[a1][offset] -= 1; U[a2][offset] += 1; U[a3][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] -= 1; V[b2][offset] += 1; V[b3][offset] -= 1
        W[c0][offset] += 0.25; W[c1][offset] -= 0.25; W[c2][offset] += 0.25; W[c3][offset] -= 0.25
        offset += 1
        
        # Term 3
        U[a0][offset] += 1; U[a1][offset] += 1; U[a2][offset] -= 1; U[a3][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] += 1; V[b2][offset] -= 1; V[b3][offset] -= 1
        W[c0][offset] += 0.25; W[c1][offset] += 0.25; W[c2][offset] -= 0.25; W[c3][offset] -= 0.25
        offset += 1
        
        # Term 4
        U[a0][offset] += 1; U[a1][offset] -= 1; U[a2][offset] -= 1; U[a3][offset] += 1
        V[b0][offset] += 1; V[b1][offset] -= 1; V[b2][offset] -= 1; V[b3][offset] += 1
        W[c0][offset] += 0.25; W[c1][offset] -= 0.25; W[c2][offset] -= 0.25; W[c3][offset] += 0.25
        offset += 1
        
        # Term 5
        U[a0][offset] += 1; U[a1][offset] -= 1
        V[b0][offset] += 1; V[b1][offset] -= 1
        W[c0][offset] += 0.5; W[c1][offset] += 0.5
        offset += 1
        
        # Term 6
        U[a2][offset] += 1; U[a3][offset] -= 1
        V[b2][offset] += 1; V[b3][offset] -= 1
        W[c2][offset] += 0.5; W[c3][offset] += 0.5
        offset += 1
        
        # Term 7
        U[a1][offset] += 1; U[a3][offset] -= 1
        V[b1][offset] += 1; V[b3][offset] -= 1
        W[c0][offset] += 0.5; W[c1][offset] -= 0.5; W[c2][offset] += 0.5; W[c3][offset] -= 0.5
        offset += 1
        
        return offset

    # Let's test this 2x2x2 rank 7 decomposition.
    # A = [[1, 2], [3, 4]]
    # B = [[5, 6], [7, 8]]
    # C = [[19, 22], [43, 50]]
    
    # U_a0 = 1,1,1,1,1,0,1
    # U_a1 = 1,-1,1,-1,-1,0,-1
    # U_a2 = 1,1,-1,-1,0,1,0
    # U_a3 = 1,-1,-1,1,0,-1,-1
    
    # V_b0 = 1,1,1,1,1,0,0
    # V_b1 = 1,-1,1,-1,-1,0,1
    # V_b2 = 1,1,-1,-1,0,1,0
    # V_b3 = 1,-1,-1,1,0,-1,-1
    
    # W_c0 = 0.25,0.25,0.25,0.25,0.5,0,0.5
    # W_c1 = 0.25,-0.25,0.25,-0.25,0.5,0,-0.5
    # W_c2 = 0.25,0.25,-0.25,-0.25,0,0.5,0.5
    # W_c3 = 0.25,-0.25,-0.25,0.25,0,0.5,-0.5
    
    # Let's check C00.
    # Sum U_a0 V_b0 W_c0 + U_a1 V_b1 W_c0 + ...
    # This is too tedious to do by hand. I will trust the Winograd scheme.
    # But I should verify it in code.
    
    # Let's create the full 2x4x8 decomposition with rank 56.
    rank = 56
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    offset = 0
    
    # We have 4 2x2x2 blocks for the first 4 columns of B (B1), and 4 for the second 4 columns (B2).
    # Actually, we split A into A1, A2 (2x2 each).
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # B1 = [[B11, B12], [B21, B22]] where each is 2x2.
    # B2 = [[B31, B32], [B41, B42]] where each is 2x2.
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # C3 = A1 B31 + A2 B41
    # C4 = A1 B32 + A2 B42
    
    # Indices for A:
    # A1: 0, 1, 2, 3
    # A2: 4, 5, 6, 7
    
    # Indices for B:
    # B11: 0, 1, 2, 3
    # B12: 4, 5, 6, 7
    # B21: 8, 9, 10, 11
    # B22: 12, 13, 14, 15
    # B31: 16, 17, 18, 19
    # B32: 20, 21, 22, 23
    # B41: 24, 25, 26, 27
    # B42: 28, 29, 30, 31
    
    # Indices for C:
    # C1: 0, 1, 2, 3
    # C2: 4, 5, 6, 7
    # C3: 8, 9, 10, 11
    # C4: 12, 13, 14, 15
    
    # Block 1: A1 B11 -> C1
    offset = add_2x2x2_rank7(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, offset, U, V, W)
    
    # Block 2: A2 B21 -> C1
    offset = add_2x2x2_rank7(4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, offset, U, V, W)
    
    # Block 3: A1 B12 -> C2
    offset = add_2x2x2_rank7(0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, offset, U, V, W)
    
    # Block 4: A2 B22 -> C2
    offset = add_2x2x2_rank7(4, 5, 6, 7, 12, 13, 14, 15, 4, 5, 6, 7, offset, U, V, W)
    
    # Block 5: A1 B31 -> C3
    offset = add_2x2x2_rank7(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, offset, U, V, W)
    
    # Block 6: A2 B41 -> C3
    offset = add_2x2x2_rank7(4, 5, 6, 7, 24, 25, 26, 27, 8, 9, 10, 11, offset, U, V, W)
    
    # Block 7: A1 B32 -> C4
    offset = add_2x2x2_rank7(0, 1, 2, 3, 20, 21, 22, 23, 12, 13, 14, 15, offset, U, V, W)
    
    # Block 8: A2 B42 -> C4
    offset = add_2x2x2_rank7(4, 5, 6, 7, 28, 29, 30, 31, 12, 13, 14, 15, offset, U, V, W)
    
    # Verify the decomposition
    # T[u, v, w] = sum_t U[u, t] V[v, t] W[w, t]
    # We can check a few entries.
    
    # Let's check if the rank is indeed 56.
    # offset should be 56.
    if offset != rank:
        # Fallback to schoolbook if something went wrong
        rank = 64
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Optional: Verify with random matrices
    if time.time() < deadline - 10:
        # Generate random A, B
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C_decomposed
        # C[k, i] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] ? No.
        # The decomposition is:
        # C = sum_t U[:,t] V[:,t] W[:,t]^T
        # Wait, the indices are:
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # The tensor T has indices (u, v, w).
        # T[u, v, w] = sum_t U[u, t] V[v, t] W[w, t].
        # u = i*m+j, v = j*p+k, w = k*n+i.
        # So C[k, i] = T[i*m+j, j*p+k, k*n+i] = sum_j sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
        
        C_decomposed = np.zeros((p, n))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for t in range(rank):
                        s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                C_decomposed[k, i] = s
        
        # Check if C_decomposed matches C_expected
        if not np.allclose(C_decomposed, C_expected, atol=1e-5):
            # Fallback to schoolbook
            rank = 64
            U = [[0] * rank for _ in range(dim_U)]
            V = [[0] * rank for _ in range(dim_V)]
            W = [[0] * rank for _ in range(dim_W)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
