import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start time to guard against timeouts
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Recursive Strassen decomposition
    # Returns U, V, W for size n x m x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    def strassen_decompose(n, m, p):
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # We assume n == m == p for Strassen recursion
        half_n = n // 2
        half_m = m // 2
        half_p = p // 2
        
        # Recursive call
        U_sub, V_sub, W_sub = strassen_decompose(half_n, half_m, half_p)
        R_sub = len(U_sub[0]) if U_sub else 0
        R_total = 7 * R_sub
        
        # Initialize global matrices
        U = [[0] * R_total for _ in range(n * m)]
        V = [[0] * R_total for _ in range(m * p)]
        W = [[0] * R_total for _ in range(p * n)]
        
        # Coefficients for A_L
        # L, I, J, coeff
        alpha = [
            { (0,0): 1, (1,1): 1 }, # M1
            { (1,0): 1, (1,1): 1 }, # M2
            { (0,0): 1 },           # M3
            { (1,1): 1 },           # M4
            { (0,0): 1, (0,1): 1 }, # M5
            { (1,0): 1, (0,0): -1 },# M6
            { (0,1): 1, (1,1): -1 } # M7
        ]
        
        # Coefficients for B_L
        # L, J, K, coeff
        beta = [
            { (0,0): 1, (1,1): 1 }, # M1
            { (0,0): 1 },           # M2
            { (0,1): 1, (1,1): -1 },# M3
            { (1,0): 1, (0,0): -1 },# M4
            { (1,1): 1 },           # M5
            { (0,0): 1, (0,1): 1 }, # M6
            { (1,0): 1, (1,1): 1 }  # M7
        ]
        
        # Coefficients for C_IK from M_L
        # I, K, [coeffs for L=0..6]
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        gamma = [
            [1, 0, 0, -1, 1, 0, -1], # C00
            [0, 0, 1, 0, 1, 0, 0],   # C01
            [0, 1, 0, 1, 0, 0, 0],   # C10
            [1, -1, 1, 0, 0, 1, 0]   # C11
        ]
        
        for L in range(7):
            base_col = L * R_sub
            for r_sub in range(R_sub):
                col = base_col + r_sub
                
                # Iterate over sub-block indices
                for i_prime in range(half_n):
                    for j_prime in range(half_m):
                        # U contribution
                        # Global i = 2*I + i_prime
                        # Global j = 2*J + j_prime
                        # U[global_i * m + global_j, col] += alpha[L, I, J] * U_sub[i_prime * half_m + j_prime, r_sub]
                        
                        # This is inefficient to loop over all i_prime, j_prime for each L.
                        # But n=4, so half_n=2, half_m=2. Small.
                        
                        pass # Logic handled below
        
        # More efficient construction:
        # We need to map (I, J) to (i_prime, j_prime) in the sub-problem.
        # The sub-problem U_sub has indices (i_prime, j_prime) corresponding to blocks A_IJ?
        # No, U_sub is the decomposition of the multiplication of the blocks.
        # The index in U_sub is (i_sub, j_sub) where i_sub, j_sub are local indices within the block size.
        # The block A_IJ corresponds to local indices (i_prime, j_prime) where global i = 2I + i_prime, global j = 2J + j_prime.
        
        # So, for a fixed L:
        # A_L is a sum of blocks A_IJ with coeff alpha[L, I, J].
        # The "matrix" A_L of size n x m is formed by placing the sub-matrices.
        # But we are building the tensor factors.
        
        # The factor U for the global problem at column 'col' should be:
        # U[2I + i_prime, 2J + j_prime, col] = alpha[L, I, J] * U_sub[i_prime, j_prime, r_sub]
        
        # So:
        for L in range(7):
            base_col = L * R_sub
            for r_sub in range(R_sub):
                col = base_col + r_sub
                
                # Get sub-factors
                # U_sub[i_prime * half_m + j_prime, r_sub]
                # V_sub[j_prime * half_p + k_prime, r_sub]
                # W_sub[k_prime * half_n + i_prime, r_sub]
                
                for I in range(2):
                    for J in range(2):
                        a_coeff = alpha[L].get((I, J), 0)
                        if a_coeff == 0: continue
                        
                        for K in range(2):
                            b_coeff = beta[L].get((J, K), 0)
                            if b_coeff == 0: continue
                            
                            # Note: The product is A_L * B_L.
                            # The sub-tensor handles the multiplication of the "values" of the blocks.
                            # The coefficients a_coeff and b_coeff scale U and V.
                            # Does the sub-rank r_sub correspond to the same "path"?
                            # Yes, the decomposition is linear.
                            
                            # However, U_sub and V_sub are fixed for the sub-problem.
                            # We just scale them.
                            
                            # But wait, U_sub depends on i_prime, j_prime.
                            # We need to iterate over i_prime, j_prime, k_prime.
                            
                            for i_prime in range(half_n):
                                for j_prime in range(half_m):
                                    u_val = a_coeff * U_sub[i_prime * half_m + j_prime][r_sub]
                                    if u_val == 0: continue
                                    
                                    global_i = 2 * I + i_prime
                                    global_j = 2 * J + j_prime
                                    u_idx = global_i * m + global_j
                                    U[u_idx][col] += u_val
                                    
                            for j_prime in range(half_m):
                                for k_prime in range(half_p):
                                    v_val = b_coeff * V_sub[j_prime * half_p + k_prime][r_sub]
                                    if v_val == 0: continue
                                    
                                    global_j = 2 * J + j_prime
                                    global_k = 2 * K + k_prime
                                    v_idx = global_j * p + global_k
                                    V[v_idx][col] += v_val
                                    
                            for i_prime in range(half_n):
                                for k_prime in range(half_p):
                                    # W depends on I, K (output blocks)
                                    # gamma[I, K, L]
                                    g_coeff = gamma[I][K][L]
                                    if g_coeff == 0: continue
                                    
                                    w_val = g_coeff * W_sub[k_prime * half_n + i_prime][r_sub]
                                    if w_val == 0: continue
                                    
                                    global_k = 2 * K + k_prime
                                    global_i = 2 * I + i_prime
                                    w_idx = global_k * n + global_i
                                    W[w_idx][col] += w_val
                                    
        return U, V, W

    U, V, W = strassen_decompose(4, 4, 4)
    
    # Verification
    # Construct random A, B
    np.random.seed(42)
    A = np.random.randint(-1, 2, (4, 4)).astype(float)
    B = np.random.randint(-1, 2, (4, 4)).astype(float)
    C_expected = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # C_computed[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
    # No, the sum is over r.
    # The term is U[?, r] V[?, r] W[?, r].
    # The indices in U, V, W are flattened.
    # U[i, r] corresponds to A_{i, j}? No, U is (n*m, R).
    # The seed code implies:
    # C[i, k] = sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] is NOT summing over j.
    # The seed code has ONE term per r.
    # So C[i, k] = sum_r ( sum_j U[i*m+j, r] V[j*p+k, r] ) W[k*n+i, r]?
    # No, that's not bilinear in A and B.
    
    # Correct check:
    # The decomposition is of the tensor T where T[i, j, k] = delta.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # This is equivalent to:
    # C[i, k] = sum_r ( sum_j U[i*m+j, r] A[i, j] ) ( sum_j V[j*p+k, r] B[j, k] ) W[k*n+i, r]
    # This requires W to be scalar per r? No, W is a vector.
    
    # Actually, the standard tensor decomposition form is:
    # C[i, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] is not right.
    
    # Let's just check the tensor product.
    # Tensor T has indices (i, j, k).
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i, j, k] to be 1 if the term A[i, j]B[j, k] contributes to C[i, k].
    # It should be 1 for all i, j, k? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # So T[i, j, k] should be 1.
    
    T_computed = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(U_np.shape[1]):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*4+i, r]
                T_computed[i, j, k] = s
                
    # T should be all ones
    if not np.allclose(T_computed, 1.0):
        # Fallback to schoolbook
        U_s = [[0] * 64 for _ in range(16)]
        V_s = [[0] * 64 for _ in range(16)]
        W_s = [[0] * 64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_s[i*4+j][col] = 1
                    V_s[j*4+k][col] = 1
                    W_s[k*4+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
