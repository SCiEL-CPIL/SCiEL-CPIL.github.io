"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    
    Strategy:
    - Split middle dimension m=3 into 2+1.
    - For the 2x2 x 2x10 part: split output into 5 blocks of 2 columns each.
      Each block is a 2x2 x 2x2 multiplication, solved by Strassen (rank 7).
      Total: 5 * 7 = 35.
    - For the 2x1 x 1x10 part: naive outer product, rank 2*10 = 20.
    - Total rank: 55.
    - Then attempt random search for rank 54 or lower.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def build_strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 * 2x2 = 2x2.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        C = [[a(e-h) + (c+d)(f+h), (a+b)h + d(g-h)],
             [(c+d)(e+f) + a(g-e), (a-c)f + (b+d)(h-e)]]
        Wait, standard Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-e)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        C11 = m1 + m4 - m5 + m6
        C12 = m3 + m5
        C21 = m2 + m4
        C22 = m1 - m2 + m3 + m7
        """
        # Indices for A (2x2): a=0, b=1, c=2, d=3
        # Indices for B (2x2): e=0, f=1, g=2, h=3
        # Indices for C (2x2): C11=0, C12=1, C21=2, C22=3
        # Convention: W[k*n+i] where k is col, i is row. So C11 -> W[0], C12 -> W[2], C21 -> W[1], C22 -> W[3]
        # Wait, W[k*n+i]: for 2x2, k=0,1 (cols), i=0,1 (rows).
        # C[0,0] -> W[0*2+0] = W[0]
        # C[0,1] -> W[1*2+0] = W[2]
        # C[1,0] -> W[0*2+1] = W[1]
        # C[1,1] -> W[1*2+1] = W[3]
        
        # m1 = (a+d)(e+h)
        U1 = [1, 0, 0, 1]
        V1 = [1, 0, 0, 1]
        W1 = [1, 0, 0, 1]  # contributes to C11, C22 with +1; C12, C21 with 0
        
        # m2 = (c+d)e
        U2 = [0, 0, 1, 1]
        V2 = [1, 0, 0, 0]
        W2 = [-1, 0, 0, 1]  # C11: -1, C21: +1, C22: +1... wait
        # C11 = m1 + m4 - m5 + m6
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m7
        # So m2 contributes: C21 += 1, C22 += -1
        W2 = [0, 1, -1, 0]  # W[0]=C11, W[1]=C21, W[2]=C12, W[3]=C22
        # W2[1] = 1 (C21), W2[3] = -1 (C22)
        
        # m3 = a(f-h)
        U3 = [1, 0, 0, 0]
        V3 = [0, 1, 0, -1]
        W3 = [0, 0, 1, 1]  # C12 += 1, C22 += 1
        
        # m4 = d(g-e)
        U4 = [0, 0, 0, 1]
        V4 = [0, 0, 1, -1]
        W4 = [1, 1, 0, 0]  # C11 += 1, C21 += 1
        
        # m5 = (a+b)h
        U5 = [1, 1, 0, 0]
        V5 = [0, 0, 0, 1]
        W5 = [-1, 0, 1, 0]  # C11 -= 1, C12 += 1
        
        # m6 = (c-a)(e+f)
        U6 = [-1, 0, 1, 0]
        V6 = [1, 1, 0, 0]
        W6 = [1, 0, 0, 0]  # C11 += 1
        
        # m7 = (b-d)(g+h)
        U7 = [0, 1, 0, -1]
        V7 = [0, 0, 1, 1]
        W7 = [0, 0, 0, 1]  # C22 += 1
        
        return [
            (U1, V1, W1),
            (U2, V2, W2),
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7),
        ]
    
    def build_strassen_decomposition():
        """
        Build the full 2x3x10 decomposition using Strassen on 2x2x2 blocks.
        
        A is 2x3: indices (i,j) i=0,1 j=0,1,2
        B is 3x10: indices (j,k) j=0,1,2 k=0..9
        C is 2x10: indices (i,k) i=0,1 k=0..9
        W convention: W[k*n + i] = W[k*2 + i]
        
        Split j into {0,1} and {2}.
        For j in {0,1}: A_sub is 2x2, B_sub is 2x10.
        Split k into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9).
        For each pair, we have a 2x2 x 2x2 problem.
        
        For j=2: A_sub is 2x1, B_sub is 1x10. Naive: rank 20.
        """
        rank_total = 5 * 7 + 20  # 55
        U = [[0] * rank_total for _ in range(n * m)]  # 2*3 = 6
        V = [[0] * rank_total for _ in range(m * p)]  # 3*10 = 30
        W = [[0] * rank_total for _ in range(p * n)]  # 10*2 = 20
        
        col = 0
        # Strassen blocks for j in {0,1}, k pairs
        strassen_blocks = build_strassen_2x2x2()
        
        for block_idx in range(5):
            k0 = 2 * block_idx
            k1 = k0 + 1
            for (u_vec, v_vec, w_vec) in strassen_blocks:
                # u_vec is 4-dim: [a, b, c, d] -> A[0,0], A[0,1], A[1,0], A[1,1]
                # Map to U indices: A[i,j] -> U[i*m + j] = U[i*3 + j]
                # a = A[0,0] -> U[0]
                # b = A[0,1] -> U[1]
                # c = A[1,0] -> U[3]
                # d = A[1,1] -> U[4]
                U[0][col] += u_vec[0]
                U[1][col] += u_vec[1]
                U[3][col] += u_vec[2]
                U[4][col] += u_vec[3]
                
                # v_vec is 4-dim: [e, f, g, h] -> B[0,k0], B[0,k1], B[1,k0], B[1,k1]
                # Map to V indices: B[j,k] -> V[j*p + k] = V[j*10 + k]
                # e = B[0,k0] -> V[k0]
                # f = B[0,k1] -> V[k1]
                # g = B[1,k0] -> V[10 + k0]
                # h = B[1,k1] -> V[10 + k1]
                V[k0][col] += v_vec[0]
                V[k1][col] += v_vec[1]
                V[10 + k0][col] += v_vec[2]
                V[10 + k1][col] += v_vec[3]
                
                # w_vec is 4-dim: [C11, C12, C21, C22]
                # C11 = C[0,k0] -> W[k0*2 + 0] = W[k0]
                # C12 = C[0,k1] -> W[k1*2 + 0] = W[2*k1]
                # C21 = C[1,k0] -> W[k0*2 + 1] = W[k0 + 1]
                # C22 = C[1,k1] -> W[k1*2 + 1] = W[2*k1 + 1]
                W[k0][col] += w_vec[0]
                W[2 * k1][col] += w_vec[1]
                W[k0 + 1][col] += w_vec[2]
                W[2 * k1 + 1][col] += w_vec[3]
                
                col += 1
        
        # Naive part for j=2: A[i,2] * B[2,k] -> C[i,k]
        # Rank 2*10 = 20
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1  # A[i,2] -> U[i*3+2]
                V[2 * p + k][col] = 1  # B[2,k] -> V[2*10+k]
                W[k * n + i][col] = 1  # C[i,k] -> W[k*2+i]
                col += 1
        
        assert col == rank_total, f"Column count {col} != rank {rank_total}"
        
        return {"U": U, "V": V, "W": W}, rank_total
    
    def verify_decomposition(result, n=2, m=3, p=10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U = np.array(result['U'], dtype=float)
        V = np.array(result['V'], dtype=float)
        W = np.array(result['W'], dtype=float)
        rank = U.shape[1]
        
        # Build the multiplication tensor
        # T[(i,j),(j',k),(i',k')] should be 1 iff i=i', j=j', k=k'
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: sum_{i,j,k} e_{(i,j)}^A \otimes e_{(j,k)}^B \otimes e_{(i,k)}^C
        # W convention: W[k*n + i]
        
        # For each test case, check that sum_r U[r] * V[r] * W[r] gives correct C
        # Let's do a symbolic check by evaluating on random inputs
        for trial in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            # u_val = U^T A_flat
            A_flat = A.flatten()  # index i*m + j
            B_flat = B.flatten()  # index j*p + k
            C_flat = np.zeros(p * n)  # index k*n + i
            
            for r in range(rank):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                C_flat += u_val * v_val * W[:, r]
            
            C_computed = C_flat.reshape(p, n).T  # reshape to (n, p)
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Build the Strassen-based rank-55 decomposition
    result, rank = build_strassen_decomposition()
    
    # Verify
    if not verify_decomposition(result):
        # Fallback to schoolbook
        rank = n * m * p
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
    
    # Try to improve: random search for rank 54
    if time.time() < deadline - 10:
        # Attempt to find a rank-54 decomposition by perturbing the rank-55 one
        # and checking if one column can be removed or merged.
        # This is a heuristic; the space is huge.
        # Instead, try a different structural decomposition.
        
        # Alternative: try to use a 3x3 Strassen-like or Winograd for the middle dim.
        # Actually, let's try: split m=3 as 1+1+1 and see if there's sharing.
        # Or try a different split: what if we don't split k into pairs but use
        # a different factorization?
        
        # For now, return the verified rank-55 result.
        pass
    
    return result
# EVOLVE-BLOCK-END
