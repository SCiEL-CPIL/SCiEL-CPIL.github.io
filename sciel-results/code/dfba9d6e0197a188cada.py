import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Verify current decomposition is valid before proceeding
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the tensor decomposition correctly computes matrix multiplication."""
        # For each output element C[i,j] = sum_k A[i,k] * B[k,j]
        # In our indexing: A is n x m, B is m x p, C is n x p
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # W[k*n+i, r] corresponds to output C[i,k]
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # Check that the coefficient of A[i,k]*B[k,j] in C[i,j] is 1
                    # A[i,k] is at row i*m + k in U
                    # B[k,j] is at row k*p + j in V
                    # C[i,j] is at row j*n + i in W
                    u_idx = i * m + k
                    v_idx = k * p + j
                    w_idx = j * n + i
                    
                    coeff = 0.0
                    for r in range(rank):
                        coeff += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(coeff - 1.0) > 1e-10:
                        return False
        
        # Also verify that no cross terms exist (coefficients should be 0 for wrong combinations)
        # This is more expensive, so we'll do a spot check
        import random
        random.seed(42)
        for _ in range(100):
            i1, i2 = random.randint(0, n-1), random.randint(0, n-1)
            j1, j2 = random.randint(0, p-1), random.randint(0, p-1)
            k1, k2 = random.randint(0, m-1), random.randint(0, m-1)
            
            # Check that A[i1,k1]*B[k2,j2] doesn't contribute to C[i2,j1] unless (i1,k1,j1) == (i2,k2,j2)
            if i1 != i2 or k1 != k2 or j1 != j2:
                u_idx = i1 * m + k1
                v_idx = k2 * p + j2
                w_idx = j1 * n + i2
                
                coeff = 0.0
                for r in range(rank):
                    coeff += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(coeff) > 1e-10:
                    return False
        
        return True
    
    # We'll verify after building the decomposition
    
    # Part 1: Strassen for 2x2 * 2x10 (5 blocks of 2x2 * 2x2)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        v_b0k0 = k0
        v_b0k1 = k1
        v_b1k0 = 10 + k0
        v_b1k1 = 10 + k1
        
        w_c0k0 = 2 * k0
        w_c0k1 = 2 * k1
        w_c1k0 = 2 * k0 + 1
        w_c1k1 = 2 * k1 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U[u_a00][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U[u_a10][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        W[w_c1k0][idx] += 1
        W[w_c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U[u_a00][idx] += 1
        V[v_b0k1][idx] += 1
        V[v_b1k1][idx] -= 1
        W[w_c0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U[u_a11][idx] += 1
        V[v_b1k0][idx] += 1
        V[v_b0k0][idx] -= 1
        W[w_c0k0][idx] += 1
        W[w_c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U[u_a00][idx] += 1
        U[u_a01][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] -= 1
        W[w_c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U[u_a10][idx] += 1
        U[u_a00][idx] -= 1
        V[v_b0k0][idx] += 1
        V[v_b0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U[u_a01][idx] += 1
        U[u_a11][idx] -= 1
        V[v_b1k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        idx += 1

    # Part 2: Schoolbook for 2x1 * 1x10
    u_a02 = 2
    u_a12 = 5
    
    for k in range(10):
        v_b2k = 20 + k
        w_c0k = 2 * k
        w_c1k = 2 * k + 1
        
        # A[0,2] * B[2,k]
        U[u_a02][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c0k][idx] += 1
        idx += 1
        
        # A[1,2] * B[2,k]
        U[u_a12][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c1k][idx] += 1
        idx += 1
    
    # Verify the decomposition is correct
    if not verify_decomposition(U, V, W, n, m, p):
        # If verification fails, fall back to schoolbook (rank 60)
        rank_schoolbook = n * m * p
        U_sb = [[0.0] * rank_schoolbook for _ in range(n * m)]
        V_sb = [[0.0] * rank_schoolbook for _ in range(m * p)]
        W_sb = [[0.0] * rank_schoolbook for _ in range(p * n)]
        
        idx_sb = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_idx = i * m + k
                    v_idx = k * p + j
                    w_idx = j * n + i
                    
                    U_sb[u_idx][idx_sb] = 1.0
                    V_sb[v_idx][idx_sb] = 1.0
                    W_sb[w_idx][idx_sb] = 1.0
                    idx_sb += 1
        
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}
    
    return {'U': U, 'V': V, 'W': W}
