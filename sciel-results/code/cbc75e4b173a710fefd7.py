import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    schoolbook_rank = n * m * p
    
    # Build schoolbook
    U0 = np.zeros((dim_u, schoolbook_rank))
    V0 = np.zeros((dim_v, schoolbook_rank))
    W0 = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    def verify(U, V, W):
        r = U.shape[1]
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Non-zero entry
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero entries (sample or all)
        # Total entries: 12*20*15 = 3600
        # Non-zero: 60
        # Zero: 3540
        # We can check all for safety, it's fast enough (3540 dot products of length r)
        # 3540 * 60 * 2 ops ~ 400k ops. Very fast.
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    # Check if this is a non-zero index
                    # a = i*m+j, b = j*p+k, c = k*n+i
                    # i = a//m, j = a%m
                    # j2 = b//p, k = b%p
                    # k2 = c//n, i2 = c%n
                    # If i==i2 and j==j2 and k==k2, it's non-zero
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue # Already checked
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 0.0) > 1e-9:
                        return False
        return True

    # Search
    start_time = time.time()
    timeout = 100.0
    
    # Try ranks lower than schoolbook
    # Start from 59 down to 50
    for r in range(schoolbook_rank - 1, 49, -1):
        if time.time() - start_time > timeout:
            break
            
        # Try several random initializations
        for trial in range(50):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + trial)
            U = np.random.randint(-1, 2, size=(dim_u, r))
            V = np.random.randint(-1, 2, size=(dim_v, r))
            W = np.random.randint(-1, 2, size=(dim_w, r))
            
            # Alternating Least Squares with integer snapping
            for it in range(20):
                if time.time() - start_time > timeout:
                    break
                    
                # Solve U
                # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # For each a, we have equations for all (b,c).
                # Vectorize:
                # Let M_b_c be a matrix where M_b_c[t] = V[b,t] * W[c,t]
                # We want U[a, :] such that U[a, :] @ M_b_c = T[a,b,c] for all b,c.
                # This is a linear system: A_U @ U[a].T = T_a
                # Where A_U is of size (dim_v * dim_w) x r
                
                # Construct A_U
                # We can do this for each 'a' or vectorize over 'a'?
                # V and W don't depend on a. So A_U is the same for all a!
                # A_U = np.kron(V, W) ? No.
                # Row index is (b,c). Col index is t.
                # A_U[b*dim_w + c, t] = V[b, t] * W[c, t]
                
                # Construct A_U once per iteration?
                # V and W change, so yes.
                # Size: (20*15) x r = 300 x r.
                
                A_U = np.zeros((dim_v * dim_w, r))
                for b in range(dim_v):
                    for c in range(dim_w):
                        A_U[b*dim_w + c, :] = V[b, :] * W[c, :]
                        
                # Target T_flat
                # T_flat[b*dim_w + c] = T[a, b, c]
                # We need to solve for each a.
                # U_sol, _, _, _ = np.linalg.lstsq(A_U, T_flat_a, rcond=None)
                
                # We can stack all 'a' targets into a matrix T_all (dim_u x dim_v*dim_w)
                # Then U_all = lstsq(A_U, T_all.T).T
                
                T_all = np.zeros((dim_u, dim_v * dim_w))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            T_all[a, b*dim_w + c] = 1.0
                            
                # Solve U
                # A_U is (300, r). T_all.T is (300, 12).
                # U_sol is (r, 12).
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(A_U, T_all.T, rcond=None)
                    U = U_sol.T
                    U = np.round(U)
                except:
                    pass
                    
                # Solve V
                # A_V[b, t] = U[a, t] * W[c, t] for fixed b?
                # No, for fixed b, equations are for all (a,c).
                # A_V is (dim_u * dim_w) x r.
                # A_V[a*dim_w + c, t] = U[a, t] * W[c, t]
                
                A_V = np.zeros((dim_u * dim_w, r))
                for a in range(dim_u):
                    for c in range(dim_w):
                        A_V[a*dim_w + c, :] = U[a, :] * W[c, :]
                        
                # T_all_V: (dim_v, dim_u*dim_w)
                T_all_V = np.zeros((dim_v, dim_u * dim_w))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            b = j*p + k
                            a = i*m + j
                            c = k*n + i
                            T_all_V[b, a*dim_w + c] = 1.0
                            
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(A_V, T_all_V.T, rcond=None)
                    V = V_sol.T
                    V = np.round(V)
                except:
                    pass
                    
                # Solve W
                # A_W[c, t] = U[a, t] * V[b, t]
                A_W = np.zeros((dim_u * dim_v, r))
                for a in range(dim_u):
                    for b in range(dim_v):
                        A_W[a*dim_v + b, :] = U[a, :] * V[b, :]
                        
                T_all_W = np.zeros((dim_w, dim_u * dim_v))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = k*n + i
                            a = i*m + j
                            b = j*p + k
                            T_all_W[c, a*dim_v + b] = 1.0
                            
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(A_W, T_all_W.T, rcond=None)
                    W = W_sol.T
                    W = np.round(W)
                except:
                    pass
                
                # Check if converged
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
