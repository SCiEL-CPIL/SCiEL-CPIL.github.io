import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for valid i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_rank = nm * mp  # 36, the naive rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find lower rank decompositions
    # For rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that sum_t U[:,t] * V[:,t] * W[:,t] = T (as a tensor)
    
    # The condition is: for all (a,b,c): T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # This is equivalent to: for each t, U[:,t], V[:,t], W[:,t] form a rank-1 tensor
    # and the sum equals T.
    
    # We can reformulate: fix U and V, then W is determined (if the system is solvable)
    # Or use a more direct approach.
    
    # Alternative: use the fact that the tensor can be seen as a matrix
    # Reshape T to a matrix of shape (nm*mp, pn) or (nm, mp*pn) etc.
    # The rank of the tensor is related to the matrix flattening ranks.
    
    # Let's try a direct search for small improvements.
    # Start from rank 36 and try to reduce.
    
    for r in range(36, 20, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a rank-r decomposition
        # Use randomized approach: pick random U, V, then solve for W
        # Actually, let's use a more systematic approach.
        
        # The tensor equation is:
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # 
        # For fixed U and V, this is linear in W:
        # For each (a,b): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all c
        # 
        # This gives a system: for each (a,b), we have a vector equation in W
        # sum_t (U[a,t]*V[b,t]) * W[:,t] = T[a,b,:]
        # 
        # Let M_{ab,t} = U[a,t]*V[b,t], then sum_t M_{ab,t} * W[:,t] = T[a,b,:]
        # 
        # This is a linear system: A * W = T_reshaped
        # where A has shape (nm*mp, r*pn) if we vectorize W
        # Actually, W has shape (pn, r), so let's vectorize it.
        
        # Let's try a different approach: use the fact that we can
        # parameterize and use least squares.
        
        # Try several random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            # Random U and V
            U_rand = np.random.randn(nm, r)
            V_rand = np.random.randn(mp, r)
            
            # Build the linear system for W
            # For each (a,b,c): T[a,b,c] = sum_t U_rand[a,t] * V_rand[b,t] * W[c,t]
            # Let M[a,b,t] = U_rand[a,t] * V_rand[b,t]
            # Then: sum_t M[a,b,t] * W[c,t] = T[a,b,c]
            
            # Vectorize: let W_vec be W flattened (pn*r elements)
            # The equation for each (a,b,c) is:
            # sum_t M[a,b,t] * W[c,t] = T[a,b,c]
            
            # This is a linear system. Let's build it.
            # Number of equations: nm * mp * pn = 9 * 12 * 12 = 1296
            # Number of unknowns: pn * r = 12 * r
            
            num_eqs = nm * mp * pn
            num_unk = pn * r
            
            if num_unk >= num_eqs:
                # Underdetermined or square, might work
                A = np.zeros((num_eqs, num_unk))
                b_vec = np.zeros(num_eqs)
                
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            for t in range(r):
                                A[idx, c*r + t] = U_rand[a, t] * V_rand[b, t]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
                
                # Solve least squares
                try:
                    W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                    W_rand = W_vec.reshape(pn, r)
                    
                    # Verify
                    error = 0.0
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                val = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                                error += (val - T[a, b, c])**2
                    
                    if error < 1e-6:
                        # Check if we can clean up to integers/half-integers
                        # Try to round
                        U_clean = np.round(U_rand * 2) / 2
                        V_clean = np.round(V_rand * 2) / 2
                        W_clean = np.round(W_rand * 2) / 2
                        
                        error2 = 0.0
                        for a in range(nm):
                            for b in range(mp):
                                for c in range(pn):
                                    val = np.sum(U_clean[a, :] * V_clean[b, :] * W_clean[c, :])
                                    error2 += (val - T[a, b, c])**2
                        
                        if error2 < 1e-6:
                            best_U = U_clean
                            best_V = V_clean
                            best_W = W_clean
                            best_rank = r
                            found = True
                            break
                        else:
                            # Use the unrounded version if it works
                            best_U = U_rand
                            best_V = V_rand
                            best_W = W_rand
                            best_rank = r
                            found = True
                            break
                except:
                    pass
        
        if found:
            # Try to go lower
            continue
        else:
            # Can't find rank r, try r+1
            break
    
    # If we didn't find anything better, use the naive decomposition
    if best_U is None:
        r = nm * mp  # 36
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        
        best_U = U
        best_V = V
        best_W = W
        best_rank = r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
