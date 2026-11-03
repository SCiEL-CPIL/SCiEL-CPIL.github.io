import time
import numpy as np
import itertools

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 110
    
    # Build the target tensor
    # T[i, k, j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # Indexing: U index = i*m + k, V index = k*p + j, W index = j*n + i
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i*m+k, k*p+j, j*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Initial schoolbook decomposition (rank 54)
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = [0.0]*(n*m)
                v = [0.0]*(m*p)
                w = [0.0]*(p*n)
                u[i*m+k] = 1.0
                v[k*p+j] = 1.0
                w[j*n+i] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    
    def verify_decomp(U_cols, V_cols, W_cols, T, tol=1e-8):
        r = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i*m+k
                    b = k*p+j
                    c = j*n+i
                    val = np.sum(U[a,:r] * V[b,:r] * W[c,:r])
                    if abs(val - T[a,b,c]) > tol:
                        return False
        return True
    
    def try_reduce_rank(U_cols, V_cols, W_cols, T, deadline):
        """Try to reduce rank by finding linear dependencies."""
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        
        # Try removing each term one at a time
        for s in range(rank):
            if time.time() > deadline - 20:
                break
            remaining = [x for x in range(rank) if x != s]
            r_new = len(remaining)
            
            # Build system: for each tensor entry, the sum of remaining terms should equal T
            # U[a,rem] * V[b,rem] * W[c,rem] should sum to T[a,b,c]
            # This is nonlinear in the remaining terms' values
            
            # Instead, try: can term s be expressed as a linear combination of others?
            # U[:,s] = sum_{t in remaining} alpha_t * U[:,t]
            # V[:,s] = sum_{t in remaining} beta_t * V[:,t]  
            # W[:,s] = sum_{t in remaining} gamma_t * W[:,t]
            # And alpha_t * beta_t * gamma_t should work out...
            
            # This is hard. Let me try a different approach: 
            # Check if the s-th term is "redundant" by seeing if removing it 
            # and adjusting others can still give the right tensor.
            
            # Simpler: just check if all entries of U[:,s], V[:,s], W[:,s] are zero
            if np.allclose(U[:,s], 0) or np.allclose(V[:,s], 0) or np.allclose(W[:,s], 0):
                U_cols.pop(s)
                V_cols.pop(s)
                W_cols.pop(s)
                rank -= 1
                continue
        
        return U_cols, V_cols, W_cols
    
    # Apply Strassen-like reductions
    # For 2x2 blocks, use 7 multiplications instead of 4
    # This is complex to implement generally, so let me try a different approach.
    
    # Use iterative local search: try to replace pairs of terms with fewer terms
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        rank = len(U_cols)
        
        # Try to remove each term
        for s in range(rank):
            if time.time() > deadline - 25:
                break
            
            # Try removing term s
            test_U = [U_cols[t] for t in range(rank) if t != s]
            test_V = [V_cols[t] for t in range(rank) if t != s]
            test_W = [W_cols[t] for t in range(rank) if t != s]
            
            if verify_decomp(test_U, test_V, test_W, T):
                U_cols = test_U
                V_cols = test_V
                W_cols = test_W
                improved = True
                break
        
        # Try to merge pairs of terms into one (if possible)
        if not improved and time.time() < deadline - 30:
            for s1 in range(len(U_cols)):
                if time.time() > deadline - 25:
                    break
                for s2 in range(s1+1, len(U_cols)):
                    if time.time() > deadline - 25:
                        break
                    
                    # Check if U[:,s1] and U[:,s2] are proportional
                    U1 = np.array(U_cols[s1])
                    U2 = np.array(U_cols[s2])
                    V1 = np.array(V_cols[s1])
                    V2 = np.array(V_cols[s2])
                    W1 = np.array(W_cols[s1])
                    W2 = np.array(W_cols[s2])
                    
                    # If U1 = a*U2 and V1 = b*V2 and W1 = c*W2 with a*b*c = 1,
                    # then we can merge
                    # Check proportionality
                    if np.allclose(U1, 0) or np.allclose(U2, 0):
                        continue
                    if np.allclose(V1, 0) or np.allclose(V2, 0):
                        continue
                    if np.allclose(W1, 0) or np.allclose(W2, 0):
                        continue
                    
                    # Find a such that U1 ≈ a*U2
                    nonzero = np.abs(U2) > 1e-10
                    if not np.any(nonzero):
                        continue
                    a_vals = U1[nonzero] / U2[nonzero]
                    if not np.allclose(a_vals, a_vals[0]):
                        continue
                    a = a_vals[0]
                    
                    if not np.allclose(U1, a*U2):
                        continue
                    
                    nonzero = np.abs(V2) > 1e-10
                    if not np.any(nonzero):
                        continue
                    b_vals = V1[nonzero] / V2[nonzero]
                    if not np.allclose(b_vals, b_vals[0]):
                        continue
                    b = b_vals[0]
                    
                    if not np.allclose(V1, b*V2):
                        continue
                    
                    nonzero = np.abs(W2) > 1e-10
                    if not np.any(nonzero):
                        continue
                    c_vals = W1[nonzero] / W2[nonzero]
                    if not np.allclose(c_vals, c_vals[0]):
                        continue
                    c = c_vals[0]
                    
                    if not np.allclose(W1, c*W2):
                        continue
                    
                    # Check if a*b*c is close to 1 (so we can absorb into one term)
                    # Actually, we need the product to be preserved.
                    # Term s1: U1*V1*W1 = a*U2 * b*V2 * c*W2 = a*b*c * (U2*V2*W2)
                    # Term s2: U2*V2*W2
                    # Sum = (a*b*c + 1) * U2*V2*W2
                    # So we can replace both with one term: (a*b*c+1)*U2*V2*W2
                    factor = a*b*c + 1
                    if abs(factor) < 1e-10:
                        # Both terms cancel! Remove both.
                        U_cols.pop(s2)
                        V_cols.pop(s2)
                        W_cols.pop(s2)
                        U_cols.pop(s1)
                        V_cols.pop(s1)
                        W_cols.pop(s1)
                        improved = True
                        break
                    else:
                        # Replace s2 with factor*U2, s1 is removed
                        U_cols[s2] = [factor*x for x in U_cols[s2]]
                        V_cols.pop(s1)
                        W_cols.pop(s1)
                        U_cols.pop(s1)
                        improved = True
                        break
                if improved:
                    break
        
        # Try linear combination reduction
        if not improved and time.time() < deadline - 30:
            rank = len(U_cols)
            for s in range(rank):
                if time.time() > deadline - 25:
                    break
                # Try to express term s as linear combination of others
                # This requires solving a system, which is complex for the product form.
                # Skip for now.
                pass
    
    rank = len(U_cols)
    
    # Convert to output format
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    # Round to half-integers
    for i in range(n*m):
        for r in range(rank):
            U[i][r] = round(U[i][r]*2)/2
    for i in range(m*p):
        for r in range(rank):
            V[i][r] = round(V[i][r]*2)/2
    for i in range(p*n):
        for r in range(rank):
            W[i][r] = round(W[i][r]*2)/2
    
    return {"U": U, "V": V, "W": W}
