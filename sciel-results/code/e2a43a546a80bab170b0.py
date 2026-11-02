import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    min_rank = 30
    max_rank = n * m * p
    
    # Pre-compute indices
    # U rows: i*m + j
    # V rows: j*p + k
    # W rows: k*n + i
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        num_trials = 500 if r == 30 else 200
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            U = np.random.choice([-1, 1], size=(n * m, r))
            V = np.random.choice([-1, 1], size=(m * p, r))
            
            W = np.zeros((p * n, r))
            valid = True
            
            # ones for the identity matrix flattened
            ones_m2 = np.ones(m * m)
            
            for i in range(n):
                for k in range(p):
                    # Construct A_ik of shape (m*m, r)
                    # Row index jj = j*m + j'
                    # Col index t
                    # Entry: U[i*m + j, t] * V[j*p + k, t]
                    
                    # Vectorize construction
                    # U_part: (m, r)
                    U_part = U[i*m : (i+1)*m, :]
                    # V_part: (m, r) - Note V index is j*p+k, so for fixed k, it's a column slice?
                    # V rows are j*p+k. For fixed k, rows are k, p+k, 2p+k...
                    # So V_part should be V[k::p, :]
                    V_part = V[k::p, :]
                    
                    # Outer product of U_part and V_part?
                    # We need matrix A where A[jj, t] = U_part[j, t] * V_part[j', t]
                    # jj = j*m + j'
                    # This is the element-wise product of U_part[:, t] and V_part[:, t] for each t,
                    # then stacked?
                    # A[:, t] = (U_part * V_part.T).flatten()? No.
                    # A[jj, t] = U_part[j, t] * V_part[j', t].
                    # This is the element-wise product of U_part (m x r) and V_part (m x r) along t?
                    # No, for a fixed t, A[:, t] is the vector of products U_part[:, t] * V_part[:, t].
                    # Wait, U_part[j, t] * V_part[j', t].
                    # This is the outer product of U_part[:, t] and V_part[:, t]?
                    # No, it's the element-wise product of the column t of U_part and column t of V_part?
                    # No, j and j' are independent.
                    # A[j*m + j', t] = U_part[j, t] * V_part[j', t].
                    # This is the Kronecker product? No.
                    # This is just:
                    # A = np.zeros((m*m, r))
                    # for t in range(r):
                    #   A[:, t] = np.kron(U_part[:, t], V_part[:, t])? No.
                    #   A[:, t] = (U_part[:, t] * V_part[:, t]).flatten()? No.
                    #   A[j*m+j', t] = U_part[j, t] * V_part[j', t].
                    #   This is the element-wise product of U_part[:, t] (repeated m times) and V_part[:, t] (repeated m times)?
                    #   Actually, A[:, t] = np.repeat(U_part[:, t], m) * np.tile(V_part[:, t], m)?
                    #   Let's check:
                    #   j=0, j'=0 -> U[0,t] V[0,t]
                    #   j=0, j'=1 -> U[0,t] V[1,t]
                    #   j=1, j'=0 -> U[1,t] V[0,t]
                    #   So A[:, t] is the outer product of U_part[:, t] and V_part[:, t] flattened?
                    #   Yes! A[:, t] = (U_part[:, t] * V_part[:, t].T).flatten()
                    
                    # Vectorized:
                    # U_part: (m, r), V_part: (m, r)
                    # We want A: (m*m, r)
                    # A[jj, t] = U_part[j, t] * V_part[j', t]
                    # This is equivalent to:
                    # A = np.einsum('jt,jt->jt', U_part, V_part)? No.
                    # A = np.zeros((m*m, r))
                    # for t in range(r):
                    #    A[:, t] = np.outer(U_part[:, t], V_part[:, t]).flatten()
                    
                    # Efficient way:
                    # A = np.einsum('jt,kt->jkt', U_part, V_part).reshape(m*m, r)
                    A_ik = np.einsum('jt,kt->jkt', U_part, V_part).reshape(m * m, r)
                    
                    try:
                        w, residuals, rank_a, s = np.linalg.lstsq(A_ik, ones_m2, rcond=None)
                    except:
                        valid = False
                        break
                    
                    two_w = 2 * w
                    rounded_two_w = np.round(two_w)
                    if not np.allclose(two_w, rounded_two_w, atol=1e-8):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w
            
            if valid:
                # Verify
                A_rand = np.random.randint(0, 10, size=(n, m))
                B_rand = np.random.randint(0, 10, size=(m, p))
                C_true = A_rand @ B_rand
                
                C_calc = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        for t in range(r):
                            s = 0
                            for j in range(m):
                                s += U[i*m+j][t] * V[j*p+k][t]
                            # Wait, the verification in the code was also wrong?
                            # C_calc[i, k] += s * W[k*n+i][t]
                            # This calculates sum_t (sum_j U V) W.
                            # But the condition is sum_t w_t U_{ij t} V_{j'k t} = delta.
                            # The correct calculation for C is:
                            # C_ik = sum_j A_ij B_jk
                            # Using decomposition:
                            # C_ik = sum_t (sum_j U_ij t A_ij) (sum_j V_jk t B_jk) W_ki t ?
                            # No, the decomposition is T_ijk = sum U V W.
                            # C_ik = sum_j T_ijk A_ij B_jk ?
                            # No, T_ijk is the structure constant.
                            # C_ik = sum_{j, j'} T_{i, j, j', k} A_{ij} B_{j'k}?
                            # No, the standard is C_ik = sum_j A_{ij} B_{jk}.
                            # The tensor T is defined by T_{i, j, k} = 1 if j is the link.
                            # So C_ik = sum_{j, k'} T_{i, j, k'} A_{ij} B_{j'k'}?
                            # No, the indices in T are i, j, k.
                            # The product is A_{ij} B_{jk}.
                            # So C_ik = sum_j T_{i, j, k} A_{ij} B_{jk}.
                            # With T_{i, j, k} = 1.
                            # So C_ik = sum_j A_{ij} B_{jk}.
                            # The decomposition says T_{i, j, k} = sum_t U_{i, j, t} V_{j, k, t} W_{k, i, t}.
                            # So C_ik = sum_j (sum_t U_{i, j, t} V_{j, k, t} W_{k, i, t}) A_{ij} B_{jk}.
                            # = sum_t W_{k, i, t} (sum_j U_{i, j, t} A_{ij}) (sum_j V_{j, k, t} B_{jk}).
                            
                            u_val = np.dot(U[i*m:(i+1)*m, t], A_rand[i, :])
                            v_val = np.dot(V[::p, t], B_rand[:, k]) # V rows are j*p+k
                            C_calc[i, k] += W[k*n+i, t] * u_val * v_val
                            
                if not np.allclose(C_true, C_calc, atol=1e-8):
                    valid = False
                    continue
                
                if valid:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
    # Fallback
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
