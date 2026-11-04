import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Laderman's 23-term algorithm for 3x3*3x3
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,1]),
        ([0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,1,0, 0,0,1]),
    ]

    def verify_laderman():
        n3, m3, p3 = 3, 3, 3
        T_target = np.zeros((n3*m3, m3*p3, p3*n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T_target[i*m3+j, j*p3+k, k*n3+i] = 1.0
        
        T_comp = np.zeros((n3*m3, m3*p3, p3*n3))
        for u, v, w in laderman_terms:
            u_np = np.array(u)
            v_np = np.array(v)
            w_np = np.array(w)
            T_comp += np.einsum('a,b,c->abc', u_np, v_np, w_np)
            
        return np.allclose(T_comp, T_target, atol=1e-9)

    if verify_laderman():
        R = 46
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Block 1: k in 0..2
        for r, (u, v, w) in enumerate(laderman_terms):
            u_np = np.array(u)
            v_np = np.array(v)
            w_np = np.array(w)
            
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = u_np[i*m+j]
                    
            for j in range(m):
                for k in range(3):
                    V[j*p+k][r] = v_np[j*3+k]
                    
            for k in range(3):
                for i in range(n):
                    W[k*n+i][r] = w_np[k*3+i]

        # Block 2: k in 3..5
        for r, (u, v, w) in enumerate(laderman_terms):
            idx = 23 + r
            u_np = np.array(u)
            v_np = np.array(v)
            w_np = np.array(w)
            
            for i in range(n):
                for j in range(m):
                    U[i*m+j][idx] = u_np[i*m+j]
                    
            for j in range(m):
                for k in range(3, 6):
                    # local k' = k - 3
                    V[j*p+k][idx] = v_np[j*3 + (k-3)]
                    
            for k in range(3, 6):
                for i in range(n):
                    # local k' = k - 3
                    W[k*n+i][idx] = w_np[(k-3)*3 + i]

        # Final verification of the full 3x3x6 decomposition
        T_target_full = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target_full[i*m+j, j*p+k, k*n+i] = 1.0
                    
        T_comp_full = np.zeros((n*m, m*p, p*n))
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for c in range(R):
            T_comp_full += np.einsum('a,b,c->abc', U_np[:,c], V_np[:,c], W_np[:,c])
            
        if np.allclose(T_comp_full, T_target_full, atol=1e-9):
            return {"U": U, "V": V, "W": W}

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
