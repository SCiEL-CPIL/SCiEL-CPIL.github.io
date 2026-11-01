import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 12
    
    # Initialize U and V based on the 12 forms
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    
    # Forms:
    # 0: (a00+a11)(b00+b10)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[4,0] = 1
    
    # 1: (a00-a11)(b00-b10)
    U[0,1] = 1; U[3,1] = -1
    V[0,1] = 1; V[4,1] = -1
    
    # 2: (a00+a10)(b01+b11)
    U[0,2] = 1; U[2,2] = 1
    V[1,2] = 1; V[5,2] = 1
    
    # 3: (a00-a10)(b01-b11)
    U[0,3] = 1; U[2,3] = -1
    V[1,3] = 1; V[5,3] = -1
    
    # 4: (a01+a11)(b02+b12)
    U[1,4] = 1; U[3,4] = 1
    V[2,4] = 1; V[6,4] = 1
    
    # 5: (a01-a11)(b02-b12)
    U[1,5] = 1; U[3,5] = -1
    V[2,5] = 1; V[6,5] = -1
    
    # 6: (a01+a10)(b03+b13)
    U[1,6] = 1; U[2,6] = 1
    V[3,6] = 1; V[7,6] = 1
    
    # 7: (a01-a10)(b03-b13)
    U[1,7] = 1; U[2,7] = -1
    V[3,7] = 1; V[7,7] = -1
    
    # 8: (a00+a01+a10+a11)(b00+b01+b02+b03)
    U[0,8] = 1; U[1,8] = 1; U[2,8] = 1; U[3,8] = 1
    V[0,8] = 1; V[1,8] = 1; V[2,8] = 1; V[3,8] = 1
    
    # 9: (a00+a01-a10-a11)(b00+b01-b02-b03)
    U[0,9] = 1; U[1,9] = 1; U[2,9] = -1; U[3,9] = -1
    V[0,9] = 1; V[1,9] = 1; V[2,9] = -1; V[3,9] = -1
    
    # 10: (a00-a01+a10-a11)(b00-b01+b02-b03)
    U[0,10] = 1; U[1,10] = -1; U[2,10] = 1; U[3,10] = -1
    V[0,10] = 1; V[1,10] = -1; V[2,10] = 1; V[3,10] = -1
    
    # 11: (a00-a01-a10+a11)(b00-b01-b02+b03)
    U[0,11] = 1; U[1,11] = -1; U[2,11] = -1; U[3,11] = 1
    V[0,11] = 1; V[1,11] = -1; V[2,11] = -1; V[3,11] = 1
    
    # Solve for W
    W = np.zeros((p*n, rank))
    
    # For each output index w (k*n + i)
    for i in range(n):
        for k_prime in range(p):
            w_idx = k_prime * n + i
            # We need to find W[w_idx, :] such that:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = delta(k, k_prime)
            # for all j in 0..m-1 and k in 0..p-1
            
            # Construct the linear system A * x = b
            # x is W[w_idx, :] (size rank)
            # Each (j, k) pair gives one equation.
            num_eqs = m * p
            A_mat = np.zeros((num_eqs, rank))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for j in range(m):
                for k in range(p):
                    # u_idx = i*m + j
                    # v_idx = j*p + k
                    u_idx = i * m + j
                    v_idx = j * p + k
                    
                    # The equation is sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = delta
                    # So the row of A is U[u_idx, :] * V[v_idx, :] (element-wise)
                    A_mat[eq_idx, :] = U[u_idx, :] * V[v_idx, :]
                    b_vec[eq_idx] = 1.0 if k == k_prime else 0.0
                    eq_idx += 1
            
            # Solve A_mat * x = b_vec
            # Since the system is underdetermined (12 unknowns, 8 equations), 
            # we use least squares or pseudo-inverse. 
            # However, for exact decomposition, a solution exists.
            # We can use np.linalg.lstsq
            sol, residuals, rank_sol, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W[w_idx, :] = sol
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
