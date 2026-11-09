import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Trivial rank
    R_trivial = n * m * p
    U_triv = np.zeros((n * m, R_trivial))
    V_triv = np.zeros((m * p, R_trivial))
    W_triv = np.zeros((p * n, R_trivial))
    
    # Fill trivial solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1

    # Identify blocks to apply Strassen
    # Disjoint 2x2x2 blocks
    blocks = [
        (0, 0, 0), # i_s, j_s, k_s
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # Check if blocks fit
    valid_blocks = []
    for i_s, j_s, k_s in blocks:
        if i_s + 2 <= n and j_s + 2 <= m and k_s + 2 <= p:
            valid_blocks.append((i_s, j_s, k_s))
            
    if not valid_blocks:
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    # Create a set of blocked (i,j,k) tuples to exclude from trivial set
    blocked = set()
    for i_s, j_s, k_s in valid_blocks:
        for i in range(i_s, i_s+2):
            for j in range(j_s, j_s+2):
                for k in range(k_s, k_s+2):
                    blocked.add((i,j,k))
                    
    U_new = []
    V_new = []
    W_new = []
    
    # Add trivial terms not in blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in blocked:
                    col_U = np.zeros(n*m)
                    col_V = np.zeros(m*p)
                    col_W = np.zeros(p*n)
                    col_U[i*m+j] = 1
                    col_V[j*p+k] = 1
                    col_W[k*n+i] = 1
                    U_new.append(col_U)
                    V_new.append(col_V)
                    W_new.append(col_W)
                    
    # Add Strassen terms for each block
    # Strassen vectors (local)
    # u, v, w are length 4
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> (k,i)
    
    strassen_terms = [
        # u, v, w
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, -1, -1, 1])), # M1
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])), # M2
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])), # M3
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])), # M4
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])), # M5
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])), # M6
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))  # M7
    ]
    
    for i_s, j_s, k_s in valid_blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Map local A indices to global U
            # 0: (0,0) -> i_s, j_s
            # 1: (0,1) -> i_s, j_s+1
            # 2: (1,0) -> i_s+1, j_s
            # 3: (1,1) -> i_s+1, j_s+1
            u[i_s*m + j_s] = u_loc[0]
            u[i_s*m + (j_s+1)] = u_loc[1]
            u[(i_s+1)*m + j_s] = u_loc[2]
            u[(i_s+1)*m + (j_s+1)] = u_loc[3]
            
            # Map local B indices to global V
            # 0: (0,0) -> j_s, k_s
            # 1: (0,1) -> j_s, k_s+1
            # 2: (1,0) -> j_s+1, k_s
            # 3: (1,1) -> j_s+1, k_s+1
            v[j_s*p + k_s] = v_loc[0]
            v[j_s*p + (k_s+1)] = v_loc[1]
            v[(j_s+1)*p + k_s] = v_loc[2]
            v[(j_s+1)*p + (k_s+1)] = v_loc[3]
            
            # Map local C indices to global W
            # 0: (0,0) -> k_s, i_s
            # 1: (0,1) -> k_s, i_s+1
            # 2: (1,0) -> k_s+1, i_s
            # 3: (1,1) -> k_s+1, i_s+1
            w[k_s*n + i_s] = w_loc[0]
            w[k_s*n + (i_s+1)] = w_loc[1]
            w[(k_s+1)*n + i_s] = w_loc[2]
            w[(k_s+1)*n + (i_s+1)] = w_loc[3]
            
            U_new.append(u)
            V_new.append(v)
            W_new.append(w)
            
    if time.time() > deadline - 10:
        # If close to deadline, return trivial to be safe? 
        # Or return current if it seems valid.
        # We haven't verified yet.
        pass

    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
                
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(U_final.shape[1]):
        T_calc += np.einsum('a,b,c->abc', U_final[:,r], V_final[:,r], W_final[:,r])
        
    if not np.allclose(T_calc, T_expected):
        # Fallback to trivial
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
