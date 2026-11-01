import numpy as np
import time
import random

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy: Find rank-R decomposition for 3x3x3 core, then extend to 3x3x6.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    n, m, p = 3, 3, 6
    
    def generate_schoolbook():
        """Generate the standard rank-54 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = generate_schoolbook()
    best_rank = n * m * p
    
    # Search for 3x3x3 core decomposition
    # For 3x3x3: U(9, R), V(9, R), W(9, R)
    # The tensor equation: for each (i,k), sum_j U[i,j,r] * V[j,k,r] * W[i,k,r] must reconstruct
    # Specifically, we need: for each output (i,k), the equation sum_r (U[i*3+j, r] * V[j*3+k, r]) * W[i*3+k, r] = delta_j... 
    # Actually the correct formulation: The multiplication tensor T has T[(i,j),(j',k),(i',k')] = 1 iff j=j', i=i', k=k'
    # For a rank-R decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # Where u_r[i*m+j] = U[i*m+j, r], v_r[j'*p+k] = V[j'*p+k, r], w_r[i'*n+k'] = W[i'*n+k', r]
    # The condition is: for all (i,j,j',k,k'), sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = T[(i,j),(j',k),(i',k')]
    # T[(i,j),(j',k),(i',k')] = 1 iff j==j' and i==i' and k==k', else 0
    
    # So for fixed (i,k), we need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] = 1 for all j where j is the "middle" index
    # Wait, let me re-derive. For 3x3x3:
    # T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # So sum_r U[i*3+j, r] * V[j'*3+k, r] * W[i'*3+k', r] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # This means: for each (i,k), the vector [sum_r U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] for j in 0,1,2]
    # should equal [delta_{j,0} * W[i*3+k, r]... no wait.
    
    # Let me think differently. Fix i and k. Then for each j, j', k':
    # sum_r U[i*3+j, r] * V[j'*3+k, r] * W[i'*3+k', r] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # If we fix i, k, and consider this as an equation in the "free" indices, it's complex.
    # Better approach: For a given U, V, solve for W.
    # For fixed (i,k), define M_{j, r} = U[i*3+j, r] * V[j*3+k, r] for j=0,1,2
    # Then we need: M @ W[i*3+k, :] = [1, 1, 1]? No.
    
    # Actually: sum_r U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] should equal 1 for the "diagonal" case
    # But the full tensor condition is more complex. Let me use the standard approach:
    # The rank-R decomposition satisfies: for all (a,b,c) where a=(i,j), b=(j',k), c=(i',k'),
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # So: sum_r U[i*3+j, r] * V[j'*3+k, r] * W[i'*3+k', r] = 1 if (j==j' and i==i' and k==k') else 0
    
    # This is a system of 27*27*27 = 19683 equations with 9*R unknowns in W (given U,V).
    # But many are redundant. Let's solve for W given U,V by checking consistency.
    
    # For each (i,k), the equations involving W[i*3+k, r] are:
    # For all j, j', k': if i==i' and k==k', then sum_r U[i*3+j, r] * V[j'*3+k, r] * W[i*3+k, r] = delta_{j,j'}
    # If i!=i' or k!=k', the LHS involves W[i'*3+k', r] not W[i*3+k, r], so no constraint on W[i*3+k, r] from those.
    
    # So for fixed (i,k), we have 9 equations (j, j' in 0,1,2):
    # sum_r U[i*3+j, r] * V[j'*3+k, r] * W[i*3+k, r] = delta_{j,j'}
    
    # Let M_{j,j',r} = U[i*3+j, r] * V[j'*3+k, r]. Then sum_r M_{j,j',r} * W[i*3+k, r] = delta_{j,j'}
    # This is 9 equations, R unknowns. We can write this as a 9xR system.
    
    def find_core_decomposition(R, max_attempts=5000):
        """Search for rank-R decomposition of 3x3x3."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U, V with entries from {-1, 0, 1}
            U_r = np.random.randint(-1, 2, size=(9, R))
            V_r = np.random.randint(-1, 2, size=(9, R))
            
            W_c = np.zeros((9, R))
            valid = True
            
            for i in range(3):
                for k in range(3):
                    # Build 9 x R matrix: M[j*3+j', r] = U[i*3+j, r] * V[j'*3+k, r]
                    M = np.zeros((9, R))
                    for j in range(3):
                        for jp in range(3):
                            M[j * 3 + jp, :] = U_r[i * 3 + j, :] * V_r[jp * 3 + k, :]
                    
                    # RHS: delta_{j,j'} for j,j' in 0,1,2 -> [1,0,0, 0,1,0, 0,0,1]
                    rhs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
                    
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integers or half-integers
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-9:
                        W_c[i * 3 + k, :] = w_vec
                    else:
                        valid = False
                        break
            
            if valid:
                # Verify the full tensor equation
                U_np = np.array(U_r)
                V_np = np.array(V_r)
                W_np = np.array(W_c)
                
                # Check: sum_r U[i*3+j, r] * V[jp*3+k, r] * W[ip*3+kp, r] == delta_{j,jp} * delta_{i,ip} * delta_{k,kp}
                correct = True
                for i in range(3):
                    for ip in range(3):
                        for k in range(3):
                            for kp in range(3):
                                for j in range(3):
                                    for jp in range(3):
                                        expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                        actual = 0.0
                                        for r in range(R):
                                            actual += U_np[i * 3 + j, r] * V_np[jp * 3 + k, r] * W_np[ip * 3 + kp, r]
                                        if abs(actual - expected) > 1e-9:
                                            correct = False
                                            break
                                    if not correct:
                                        break
                                if not correct:
                                    break
                            if not correct:
                                break
                        if not correct:
                            break
                    if not correct:
                        break
                
                if correct:
                    return (U_r.tolist(), V_r.tolist(), W_c.tolist())
        
        return None
    
    # Try ranks from low to high
    for R in range(23, 28):
        if time.time() > deadline:
            break
        
        core = find_core_decomposition(R)
        if core is not None:
            U_core, V_core, W_core = core
            total_rank = 2 * R
            
            if total_rank < best_rank:
                # Extend to 3x3x6
                U_tot = [[0] * total_rank for _ in range(n * m)]
                V_tot = [[0] * total_rank for _ in range(m * p)]
                W_tot = [[0] * total_rank for _ in range(p * n)]
                
                # Fill U: same for both blocks
                for i in range(3):
                    for j in range(3):
                        for r in range(R):
                            U_tot[i * 3 + j][r] = U_core[i * 3 + j][r]
                            U_tot[i * 3 + j][R + r] = U_core[i * 3 + j][r]
                
                # Fill V, W for both blocks
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for r in range(R):
                                # Block 1 (k=0,1,2)
                                V_tot[j * 6 + k][r] = V_core[j * 3 + k][r]
                                W_tot[k * 3 + i][r] = W_core[k * 3 + i][r]
                                
                                # Block 2 (k=3,4,5)
                                V_tot[j * 6 + k + 3][R + r] = V_core[j * 3 + k][r]
                                W_tot[(k + 3) * 3 + i][R + r] = W_core[k * 3 + i][r]
                
                best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                best_rank = total_rank
                break
    
    return best_result
