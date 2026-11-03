import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T is 1 if j_a == j_b and i_a == i_c and k_b == k_c
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    # To allow for half-integer solutions, we scale the target tensor by 8.
    # If U, V, W are integers decomposing 8T, then U/2, V/2, W/2 decompose T.
    # The problem allows half-integers.
    T_scaled = 8.0 * T

    # Try to find a decomposition with rank R
    # We try R from 26 to 30
    for R in range(26, 31):
        # Try a few random initializations
        for trial in range(100):
            # Sparse random integer initialization for U and V
            # Using small integers
            U = np.zeros((dim_u, R), dtype=int)
            V = np.zeros((dim_v, R), dtype=int)
            
            # Randomly place 1s and -1s in U and V
            # Density ~ 0.3
            for r in range(R):
                # Ensure each column has some non-zeros
                num_non_zero_u = max(1, int(np.random.uniform(0.2, 0.5) * dim_u))
                indices_u = np.random.choice(dim_u, num_non_zero_u, replace=False)
                for idx in indices_u:
                    U[idx, r] = np.random.choice([-1, 1])
                
                num_non_zero_v = max(1, int(np.random.uniform(0.2, 0.5) * dim_v))
                indices_v = np.random.choice(dim_v, num_non_zero_v, replace=False)
                for idx in indices_v:
                    V[idx, r] = np.random.choice([-1, 1])
            
            valid = True
            W = np.zeros((dim_w, R), dtype=int)
            
            for c in range(dim_w):
                # Extract T_scaled[:, :, c]
                T_c = T_scaled[:, :, c]
                
                # Form matrix M of size (dim_u * dim_v) x R
                # M[eq_idx, r] = U[a, r] * V[b, r]
                M = np.zeros((dim_u * dim_v, R), dtype=float)
                for a in range(dim_u):
                    for b in range(dim_v):
                        eq_idx = a * dim_v + b
                        M[eq_idx, :] = U[a, :] * V[b, :]
                
                T_c_flat = T_c.flatten()
                
                # Solve M * W_c = T_c_flat
                try:
                    W_c, residuals, rank, s = np.linalg.lstsq(M, T_c_flat, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is integer (close to integer)
                W_c_int = np.round(W_c)
                if not np.allclose(W_c, W_c_int, atol=1e-9):
                    valid = False
                    break
                
                W[c, :] = W_c_int
            
            if valid:
                # Verify the solution
                T_prime = np.zeros((dim_u, dim_v, dim_w))
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            T_prime[a, b, c] = np.dot(U[a, :], V[b, :] * W[c, :])
                
                if np.allclose(T_prime, T_scaled, atol=1e-9):
                    # Convert to half-integers (divide by 2)
                    U_list = (U / 2.0).tolist()
                    V_list = (V / 2.0).tolist()
                    W_list = (W / 2.0).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook (rank 30)
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
    return {"U": U, "V": V, "W": W}
