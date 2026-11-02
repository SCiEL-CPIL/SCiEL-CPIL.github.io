"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is (u, v, w) where:
      u: vector of length 4 (2x2 input A flattened row-major: A[0,0], A[0,1], A[1,0], A[1,1])
      v: vector of length 4 (2x2 input B flattened row-major: B[0,0], B[0,1], B[1,0], B[1,1])
      w: vector of length 4 (2x2 output C flattened row-major: C[0,0], C[0,1], C[1,0], C[1,1])
    """
    # Strassen's 7 products for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    terms = []
    # m1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, -1]  # C11 += 1, C22 += -1
    terms.append((u1, v1, w1))
    
    # m2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]  # C21 += 1
    terms.append((u2, v2, w2))
    
    # m3
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]  # C12 += 1, C22 += 1
    terms.append((u3, v3, w3))
    
    # m4
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]  # C11 += 1, C21 += 1
    terms.append((u4, v4, w4))
    
    # m5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]  # C11 += -1, C12 += 1
    terms.append((u5, v5, w5))
    
    # m6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]  # C22 += 1
    terms.append((u6, v6, w6))
    
    # m7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 1]  # C11 += 1, C22 += 1
    terms.append((u7, v7, w7))
    
    return terms

def _rank11_2x3x2():
    """
    Return 11 rank-1 terms for 2x3x2 matrix multiplication.
    A is 2x3 (flattened: 6 elements), B is 3x2 (flattened: 6 elements), C is 2x2 (flattened: 4 elements).
    C = A * B where A[i,j] * B[j,k] summed over j.
    """
    # Split the 3 middle dimension into 2+1:
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x2); B2 (1x2)]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2x2, rank 7 (Strassen)
    # A2*B2: 2x1x2, rank 2 (each of the 2 outputs is a dot product of 2 vectors, but actually rank 2)
    # Wait: 2x1x2: A is 2x1 (2 elements), B is 1x2 (2 elements), C is 2x2 (4 elements)
    # C[i,k] = A[i,0] * B[0,k]
    # This is outer product, rank 2: terms (A[i,0], B[0,k], e_i ⊗ e_k)
    # So rank = 2
    # Total: 7 + 2 = 9? But the known rank is 11...
    
    # Let me reconsider. Actually R(2,3,2) might be 11.
    # Let me just use schoolbook for the 2x3x2 part: rank 12
    # Then total for 2x3x6 = 3 * 12 = 36, which is not an improvement.
    
    # Alternative: use the known rank-11 construction for 2x3x2.
    # Let me implement it carefully.
    
    # For 2x3x2: A(2x3) * B(3x2) = C(2x2)
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Split middle dim: j=0,1 vs j=2
    # Part 1: j in {0,1}: A1(2x2) * B1(2x2) -> C_part1(2x2), rank 7 (Strassen)
    # Part 2: j=2: A2(2x1) * B2(1x2) -> C_part2(2x2), rank 2
    # Total: 9
    
    # But wait, is R(2,3,2) = 9 or 11? Let me verify rank 9 is valid.
    # If valid, then 2x3x6 = 3 * 9 = 27.
    
    # Let me verify the rank-9 construction is correct.
    # Strassen for A1*B1 gives 7 terms.
    # Outer product for A2*B2 gives 2 terms.
    # Total 9 terms. This should be valid.
    
    terms = []
    
    # Strassen terms for the 2x2x2 part (j=0,1)
    strassen_terms = _strassen_2x2x2()
    for u, v, w in strassen_terms:
        # u is length 4: [A[0,0], A[0,1], A[1,0], A[1,1]] (for the 2x2 submatrix)
        # v is length 4: [B[0,0], B[0,1], B[1,0], B[1,1]] (for the 2x2 submatrix)
        # w is length 4: [C[0,0], C[0,1], C[1,0], C[1,1]]
        
        # Map to full 2x3x2:
        # A flattened: [A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]] (row-major)
        # B flattened: [B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1]] (row-major)
        # C flattened: [C[0,0], C[0,1], C[1,0], C[1,1]] (row-major)
        
        u_full = [0.0] * 6
        v_full = [0.0] * 6
        w_full = [0.0] * 4
        
        # u[0]=A[0,0], u[1]=A[0,1], u[2]=A[1,0], u[3]=A[1,1]
        u_full[0] = u[0]  # A[0,0]
        u_full[1] = u[1]  # A[0,1]
        u_full[4] = u[2]  # A[1,0]
        u_full[5] = u[3]  # A[1,1]
        
        # v[0]=B[0,0], v[1]=B[0,1], v[2]=B[1,0], v[3]=B[1,1]
        v_full[0] = v[0]  # B[0,0]
        v_full[1] = v[1]  # B[0,1]
        v_full[2] = v[2]  # B[1,0]
        v_full[3] = v[3]  # B[1,1]
        
        # w[0]=C[0,0], w[1]=C[0,1], w[2]=C[1,0], w[3]=C[1,1]
        w_full[0] = w[0]
        w_full[1] = w[1]
        w_full[2] = w[2]
        w_full[3] = w[3]
        
        terms.append((u_full, v_full, w_full))
    
    # Outer product terms for A2*B2 (j=2)
    # A2 = [A[0,2], A[1,2]], B2 = [B[2,0], B[2,1]]
    # C[i,k] += A[i,2] * B[2,k]
    # Term 1: i=0, k=0: A[0,2]*B[2,0] -> C[0,0]
    u_t1 = [0.0] * 6
    v_t1 = [0.0] * 6
    w_t1 = [0.0] * 4
    u_t1[2] = 1.0  # A[0,2]
    v_t1[4] = 1.0  # B[2,0]
    w_t1[0] = 1.0  # C[0,0]
    terms.append((u_t1, v_t1, w_t1))
    
    # Term 2: i=0, k=1: A[0,2]*B[2,1] -> C[0,1]
    u_t2 = [0.0] * 6
    v_t2 = [0.0] * 6
    w_t2 = [0.0] * 4
    u_t2[2] = 1.0  # A[0,2]
    v_t2[5] = 1.0  # B[2,1]
    w_t2[1] = 1.0  # C[0,1]
    terms.append((u_t2, v_t2, w_t2))
    
    # Term 3: i=1, k=0: A[1,2]*B[2,0] -> C[1,0]
    u_t3 = [0.0] * 6
    v_t3 = [0.0] * 6
    w_t3 = [0.0] * 4
    u_t3[5] = 1.0  # A[1,2]
    v_t3[4] = 1.0  # B[2,0]
    w_t3[2] = 1.0  # C[1,0]
    terms.append((u_t3, v_t3, w_t3))
    
    # Term 4: i=1, k=1: A[1,2]*B[2,1] -> C[1,1]
    u_t4 = [0.0] * 6
    v_t4 = [0.0] * 6
    w_t4 = [0.0] * 4
    u_t4[5] = 1.0  # A[1,2]
    v_t4[5] = 1.0  # B[2,1]
    w_t4[3] = 1.0  # C[1,1]
    terms.append((u_t4, v_t4, w_t4))
    
    return terms  # 7 + 4 = 11 terms

def construct():
    """
    Construct an exact tensor decomposition for M_{2,3,6} using block decomposition.
    Split the 6 columns into three 2-column blocks.
    Each 2x3x2 block has rank 11 (7 Strassen + 4 outer product).
    Total rank = 3 * 11 = 33.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Get the 11 rank-1 terms for 2x3x2
    base_terms = _rank11_2x3x2()
    
    rank = 3 * len(base_terms)  # 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    for block in range(3):
        for t_idx, (u, v, w) in enumerate(base_terms):
            col = block * len(base_terms) + t_idx
            
            # u is length 6: A flattened [A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]]
            # Index in U: i*m + j where i in {0,1}, j in {0,1,2}
            for idx in range(6):
                U[idx][col] = u[idx]
            
            # v is length 6: B flattened for this block
            # Block 0: columns 0,1 of B -> B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1]
            # Block 1: columns 2,3 of B -> B[0,2], B[0,3], B[1,2], B[1,3], B[2,2], B[2,3]
            # Block 2: columns 4,5 of B -> B[0,4], B[0,5], B[1,4], B[1,5], B[2,4], B[2,5]
            # v[idx] corresponds to B[j, k] where j=idx//2, k=idx%2 within the block
            # In full B (3x6), B[j, block*2 + k]
            # Index in V: j*p + k where j in {0,1,2}, k in {0,...,5}
            for idx in range(6):
                j = idx // 2
                k_local = idx % 2
                k_full = block * 2 + k_local
                V[j * p + k_full][col] = v[idx]
            
            # w is length 4: C flattened [C[0,0], C[0,1], C[1,0], C[1,1]]
            # Block 0: C columns 0,1 -> C[0,0], C[0,1], C[1,0], C[1,1]
            # Block 1: C columns 2,3 -> C[0,2], C[0,3], C[1,2], C[1,3]
            # Block 2: C columns 4,5 -> C[0,4], C[0,5], C[1,4], C[1,5]
            # w[idx] corresponds to C[i, block*2 + k] where i=idx//2, k=idx%2
            # Index in W: k*n + i where k in {0,...,5}, i in {0,1}
            for idx in range(4):
                i = idx // 2
                k_local = idx % 2
                k_full = block * 2 + k_local
                W[k_full * n + i][col] = w[idx]
    
    # Verify correctness
    if time.time() < deadline - 10:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Test with random matrices
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute C from decomposition: C[k*n+i, :] = sum_r W[k*n+i, r] * (A_flat * U[:,r]) * (B_flat * V[:,r])
            # Actually: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j terms] * V[j*p+k terms] * W[k*n+i]
            # More precisely: C[i,k] = sum_r (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r]) * W[k*n+i, r]
            
            A_flat = A_test.flatten()  # length 6
            B_flat = B_test.flatten()  # length 18
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                a_part = np.dot(A_flat, U_np[:, r])
                b_part = np.dot(B_flat, V_np[:, r])
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += a_part * b_part * W_np[k * n + i, r]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fall back to schoolbook
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][column] = 1
                            V[j * p + k][column] = 1
                            W[k * n + i][column] = 1
                            column += 1
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
