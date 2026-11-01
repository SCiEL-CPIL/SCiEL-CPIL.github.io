import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 8
    
    # Current best known: rank 56 via Strassen block decomposition
    rank = 56
    
    U = [[0]*rank for _ in range(n*m)]  # 8 x 56
    V = [[0]*rank for _ in range(m*p)]  # 32 x 56
    W = [[0]*rank for _ in range(p*n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # For each term t, define:
    # x_coeffs: dict mapping (i', j') to coefficient
    # y_coeffs: dict mapping (j', k') to coefficient
    # z_coeffs: dict mapping (i', k') to coefficient
    
    strassen_terms = [
        # t=0: (x00+x11)(y00+y11)
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1: (x10+x11)y00
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2: x00(y01-y11)
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3: x11(y10-y00)
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4: (x00+x01)y11
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5: (x10-x00)(y00+y01)
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6: (x01-x11)(y10+y11)
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col = 0
    for a in range(2):  # A block column: 2a, 2a+1
        for b in range(4):  # B column block: 2b, 2b+1
            for t in range(7):
                term = strassen_terms[t]
                # X part: A[i', 2a+j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                
                # Y part: B[2a+j', 2b+k']
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                
                # Z part: C[i', 2b+k'] -> W[2b+k', i']
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                
                col += 1
    
    # Quick verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify: sum_t U[:,t] * V[:,t] * W[:,t] should equal the multiplication tensor
    # The tensor T[i,j,k] = 1 if the (i,j) element of A and (j,k) element of B contribute to C[i,k]
    # T[i, j, k] where i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # i = i_A * m + j_A, j = j_B * p + k_B, k = k_C * n + i_C
    # Non-zero when j_A == j_B and i_A == i_C
    
    # Check a few random cases
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        u_vec = U_np[:, t]
        v_vec = V_np[:, t]
        w_vec = W_np[:, t]
        
        # Reconstruct A from u_vec: u_vec[i*m + j] gives coefficient for A[i,j]
        A_t = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_t[i, j] = u_vec[i*m + j]
        
        # Reconstruct B from v_vec: v_vec[j*p + k] gives coefficient for B[j,k]
        B_t = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_t[j, k] = v_vec[j*p + k]
        
        # C_t = A_t @ B_t, then w_vec[k*n + i] should be the coefficient
        C_t = A_t @ B_t
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += w_vec[k*n + i] * C_t[i, k]
    
    # Actually the decomposition works differently. Let me verify properly.
    # The tensor decomposition says: T = sum_t U_t ⊗ V_t ⊗ W_t
    # where T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    # And T[i,j,k] = 1 if the multiplication of A[i_A, j_A] and B[j_B, k_B] 
    # contributes to C[i_C, k_C] where i_A==i_C, j_A==j_B
    
    # Let's verify the tensor directly
    T_computed = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U_np[:,t], V_np[:,t], W_np[:,t])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i_A in range(n):
        for j_A in range(m):
            for j_B in range(m):
                for k_B in range(p):
                    if j_A == j_B:
                        i = i_A * m + j_A
                        j = j_B * p + k_B
                        for i_C in range(n):
                            if i_A == i_C:
                                k = k_B * n + i_C
                                T_expected[i, j, k] = 1
    
    diff = np.max(np.abs(T_computed - T_expected))
    if diff > 1e-10:
        # Fallback: return schoolbook decomposition (rank 64)
        rank_fb = n * m * p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j][idx] = 1
                    V_fb[j*p + k][idx] = 1
                    W_fb[k*n + i][idx] = 1
                    idx += 1
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
