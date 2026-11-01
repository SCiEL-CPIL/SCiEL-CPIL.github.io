import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Precompute tensor slices
    # T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i
    # We need T_a (vector of length mp*pn), T_b (vector of length nm*pn), T_d (vector of length nm*mp)
    
    # Initialize slices with zeros
    T_slices_a = [np.zeros(mp * pn) for _ in range(nm)]
    T_slices_b = [np.zeros(nm * pn) for _ in range(mp)]
    T_slices_d = [np.zeros(nm * mp) for _ in range(pn)]
    
    # Fill slices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                d = k*n+i
                
                # T[a, b, d] = 1
                # For T_a: index is b*pn + d
                T_slices_a[a][b*pn + d] = 1.0
                # For T_b: index is a*pn + d
                T_slices_b[b][a*pn + d] = 1.0
                # For T_d: index is a*mp + b
                T_slices_d[d][a*mp + b] = 1.0

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_rank = nm * mp * pn
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook initialization
    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank
    # Start from a lower bound estimate. 28 is a good guess.
    for r in range(28, 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Alternating Minimization
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for U
                U_new = np.zeros((nm, r))
                for a in range(nm):
                    # Matrix A_a: (mp*pn) x r
                    # A_a[idx, t] = V[b, t] * W[d, t] where idx = b*pn+d
                    # We can construct this efficiently
                    # V is mp x r, W is pn x r
                    # We need the Kronecker product structure?
                    # For each t, the vector is V[:,t] * W[:,t] (element-wise? No, outer product flattened)
                    # Actually, A_a[:, t] is the vector formed by V[:,t] * W[:,t] (element-wise product of columns? No)
                    # The term is V[b,t] * W[d,t].
                    # So column t of A_a is the element-wise product of V[:,t] and W[:,t] flattened?
                    # No, V[b,t] is a scalar. W[d,t] is a scalar.
                    # The vector for column t is [V[0,t]*W[0,t], V[0,t]*W[1,t], ..., V[mp-1,t]*W[pn-1,t]]
                    # This is the element-wise product of V[:,t] (repeated pn times) and W[:,t] (repeated mp times)?
                    # No.
                    # Let's just construct it.
                    # V_col = V[:, t] (mp,)
                    # W_col = W[:, t] (pn,)
                    # A_col = np.repeat(V_col, pn) * np.tile(W_col, mp)
                    
                    # This is O(mp*pn*r) per a. 6 * 24*16*28 ~ 64k ops. Fast.
                    A_a = np.zeros((mp*pn, r))
                    for t in range(r):
                        V_col = V[:, t]
                        W_col = W[:, t]
                        A_a[:, t] = np.repeat(V_col, pn) * np.tile(W_col, mp)
                    
                    x, residuals, rank_m, sv = np.linalg.lstsq(A_a, T_slices_a[a], rcond=None)
                    U_new[a, :] = x
                U = U_new
                
                # Solve for V
                V_new = np.zeros((mp, r))
                for b in range(mp):
                    A_b = np.zeros((nm*pn, r))
                    for t in range(r):
                        U_col = U[:, t]
                        W_col = W[:, t]
                        A_b[:, t] = np.repeat(U_col, pn) * np.tile(W_col, nm)
                    
                    x, residuals, rank_m, sv = np.linalg.lstsq(A_b, T_slices_b[b], rcond=None)
                    V_new[b, :] = x
                V = V_new
                
                # Solve for W
                W_new = np.zeros((pn, r))
                for d in range(pn):
                    A_d = np.zeros((nm*mp, r))
                    for t in range(r):
                        U_col = U[:, t]
                        V_col = V[:, t]
                        A_d[:, t] = np.repeat(U_col, mp) * np.tile(V_col, nm)
                    
                    x, residuals, rank_m, sv = np.linalg.lstsq(A_d, T_slices_d[d], rcond=None)
                    W_new[d, :] = x
                W = W_new
                
                # Check convergence
                # Compute residual
                # We can check a few random entries or the norm of the residual
                # For speed, check only non-zeros and a few zeros
                residual_norm = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            d = k*n+i
                            val = np.sum(U[a,:] * V[b,:] * W[d,:])
                            if abs(val - 1) > 1e-8:
                                residual_norm += (val - 1)**2
                
                if residual_norm < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Check if we can round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            d = k*n+i
                            val = np.sum(U_rounded[a,:] * V_rounded[b,:] * W_rounded[d,:])
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
                
                # Check zeros
                for _ in range(100):
                    a = np.random.randint(0, nm)
                    b = np.random.randint(0, mp)
                    d = np.random.randint(0, pn)
                    # Check if (a,b,d) is non-zero
                    is_nonzero = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i*m+j and b == j*p+k and d == k*n+i:
                                    is_nonzero = True
                    if not is_nonzero:
                        val = np.sum(U_rounded[a,:] * V_rounded[b,:] * W_rounded[d,:])
                        if abs(val) > 1e-9:
                            valid = False
                            break
                if not valid:
                    continue # Try next restart
                
                found = True
                best_rank = r
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break
        
        if found:
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
