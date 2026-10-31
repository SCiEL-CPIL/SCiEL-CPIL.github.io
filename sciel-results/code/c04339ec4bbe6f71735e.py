import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Start with rank 49
    rank_init = 49
    
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    # Define the 7x7 Strassen coefficients
    beta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    delta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0 (M1)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    delta[0][0][0] = 1; delta[0][1][1] = 1
    
    # s=1 (M2)
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    delta[1][1][0] = 1; delta[1][1][1] = -1
    
    # s=2 (M3)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    delta[2][0][1] = 1; delta[2][1][1] = 1
    
    # s=3 (M4)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    delta[3][0][0] = 1; delta[3][1][0] = 1
    
    # s=4 (M5)
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    delta[4][0][0] = -1; delta[4][0][1] = 1
    
    # s=5 (M6)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    delta[5][1][1] = 1
    
    # s=6 (M7)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    delta[6][0][0] = 1
    
    # Define U2, V2, W2 for 2x2 Strassen (rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # t=0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # t=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # t=2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # t=3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # t=4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # t=5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # t=6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            for i in range(4):
                for j in range(4):
                    I = i // 2; i_prime = i % 2
                    J = j // 2; j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * U2[i_prime * 2 + j_prime][t]
                    if coeff != 0:
                        U[row_u][c] = coeff
                        
            for j in range(4):
                for k in range(4):
                    J = j // 2; j_prime = j % 2
                    K = k // 2; k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * V2[j_prime * 2 + k_prime][t]
                    if coeff != 0:
                        V[row_v][c] = coeff
                        
            for k in range(4):
                for i in range(4):
                    K = k // 2; k_prime = k % 2
                    I = i // 2; i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * W2[k_prime * 2 + i_prime][t]
                    if coeff != 0:
                        W[row_w][c] = coeff
                        
    # Attempt to reduce rank to 48 by finding a linear dependency
    if time.time() < deadline - 10:
        # Flatten the 49 terms into a matrix of shape (4096, 49)
        # Term t is U[:, t] (16), V[:, t] (16), W[:, t] (16)
        # Outer product U_t (V_t (W_t^T))?
        # The tensor contribution is U[:,t] * V[:,t] * W[:,t] (element-wise product of outer products?)
        # No, the tensor T is sum of u_t v_t w_t.
        # In flat form, it's the vectorized Kronecker product.
        # np.kron(np.kron(U[:,t], V[:,t]), W[:,t])
        
        flat_terms = np.zeros((16 * 16 * 16, 49), dtype=float)
        for t in range(49):
            u_t = U[:, t]
            v_t = V[:, t]
            w_t = W[:, t]
            # Flatten u_t (v_t (w_t^T))?
            # The index in C is determined by W.
            # Standard tensor rank decomposition: T = sum u_t (v_t^T w_t) ?
            # Actually, C = (U A) * (V B) * W^T is not right.
            # C_ik = sum_t U_{i,*,t} ...
            # The term is u_t \otimes v_t \otimes w_t.
            # Vectorized: np.kron(np.kron(u_t, v_t), w_t)
            flat_terms[:, t] = np.kron(np.kron(u_t, v_t), w_t)
            
        # Find null space
        # SVD of flat_terms (4096 x 49)
        # We want vector lambda such that flat_terms @ lambda = 0
        try:
            s, vh = np.linalg.svd(flat_terms, full_matrices=False)
            # vh has shape (49, 49)
            # The last row of vh corresponds to the smallest singular value
            # If rank is 48, the 49th singular value should be 0 (or very small)
            if s.shape[0] > 0 and s[-1] < 1e-9:
                lambda_vec = vh[-1, :]
                # Find index k with max |lambda_k| to minimize numerical error in division
                k = np.argmax(np.abs(lambda_vec))
                if abs(lambda_vec[k]) > 1e-9:
                    alpha = -1.0 / lambda_vec[k]
                    # New coefficients
                    new_coeffs = 1.0 + alpha * lambda_vec
                    new_coeffs[k] = 0.0
                    
                    # Construct new U, V, W with 48 columns
                    U_new = np.zeros((16, 48), dtype=float)
                    V_new = np.zeros((16, 48), dtype=float)
                    W_new = np.zeros((16, 48), dtype=float)
                    
                    col_idx = 0
                    for t in range(49):
                        if t == k:
                            continue
                        coeff = new_coeffs[t]
                        # It is possible that coeff is very small?
                        # If we drop the term with largest lambda, others should be fine.
                        U_new[:, col_idx] = U[:, t] * coeff
                        V_new[:, col_idx] = V[:, t] * coeff
                        W_new[:, col_idx] = W[:, t] * coeff
                        col_idx += 1
                    
                    # Verify the new decomposition
                    # Check if U_new A V_new B W_new^T approximates C
                    # We can check the tensor equality.
                    flat_new = np.zeros((4096, 48), dtype=float)
                    for t in range(48):
                        flat_new[:, t] = np.kron(np.kron(U_new[:, t], V_new[:, t]), W_new[:, t])
                    
                    # Reconstruct original tensor T from original 49 terms
                    T_orig = np.zeros(4096, dtype=float)
                    for t in range(49):
                        T_orig += flat_terms[:, t]
                        
                    T_new = np.zeros(4096, dtype=float)
                    for t in range(48):
                        T_new += flat_new[:, t]
                        
                    if np.allclose(T_orig, T_new, atol=1e-9):
                        # Success! Return rank 48
                        return {
                            "U": U_new.tolist(),
                            "V": V_new.tolist(),
                            "W": W_new.tolist()
                        }
        except Exception as e:
            pass
            
    # Fallback to rank 49
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
