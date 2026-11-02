import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the multiplication tensor T[i,j,k] = delta[i,j] * delta[j,k]
    # A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i, j, k] = 1 always (the tensor is all ones for this simple case)
    # No wait: T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k
    
    # We need to find rank-1 tensors u*v*w such that sum of u_t*v_t*w_t = T
    # A rank-1 term has support: for each (i,j) in u, (j',k) in v, (k',i') in w
    # with j=j' and k=k' and i'=i
    
    # Let's define the tensor as a 3D array
    T = np.ones((n, m, p))
    
    # Pool of candidate rank-1 terms
    # Each term is (u, v, w) where:
    # u is a vector of length n*m (flattened A indices)
    # v is a vector of length m*p (flattened B indices)  
    # w is a vector of length p*n (flattened C indices)
    
    # For a rank-1 term to be valid, it must be of the form:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] contributes to T[i,j,k]
    
    # Schoolbook terms: single (i,j,k) triples
    # Strassen-like terms: linear combinations
    
    # Let's generate a pool of terms
    # Each term is represented by (u_indices, v_indices, w_indices)
    # where u_indices is a dict {(i,j): coeff}, etc.
    
    pool = []
    
    # Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                pool.append((u, v, w))
    
    # Strassen-like terms for 2x2 blocks
    # For A block [[a,b],[c,d]] and B block [[x,y],[u,v]]:
    # m1 = (a+d)(x+u) -> covers C[0,0]+C[1,1]
    # m2 = (c+d)x -> covers C[1,0]+C[1,1]
    # m3 = a(y-u) -> covers C[0,1]-C[1,1]
    # m4 = d(v-w) -> but B is 2x2, so this is d*v -> covers C[1,1]
    # Actually let's use the standard Strassen:
    # m1 = (a+d)(x+u)
    # m2 = (c+d)x
    # m3 = a(y-u)
    # m4 = d(v-u)  -- wait, B[1,0]=u, B[1,1]=v
    # m5 = (a+b)v
    # m6 = (c-a)(x+u)
    # m7 = (b-d)(y+v)
    
    # For 2x2x2 Strassen:
    # A = [[a,b],[c,d]], B = [[x,y],[u,v]]
    # m1 = (a+d)(x+u)
    # m2 = (c+d)x
    # m3 = a(y-u)
    # m4 = d(v-u)  -- this doesn't look right
    # Let me use the correct Strassen:
    # m1 = (a+d)(x+u)
    # m2 = (c+d)x
    # m3 = a(y-u)
    # m4 = d(v-u)
    # m5 = (a+b)v
    # m6 = (c-a)(x+u)
    # m7 = (b-d)(y+v)
    # C[0,0] = m1 - m4 - m6 + m7
    # C[0,1] = m3 + m4 + m5 - m7
    # C[1,0] = m2 + m6
    # C[1,1] = m1 - m3 + m5
    
    # Let's add Strassen terms for all 2x2 sub-blocks
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                # A block: rows i0, i0+1; cols j0, j0+1
                # B block: rows j0, j0+1; cols k0, k0+1
                # C block: rows i0, i0+1; cols k0, k0+1
                
                # m1 = (a+d)(x+u)
                u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
                v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
                w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
                pool.append((u1, v1, w1))
                
                # m2 = (c+d)x
                u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
                v2 = {(j0, k0): 1}
                w2 = {(k0, i0+1): 1, (k0+1, i0+1): 1}
                pool.append((u2, v2, w2))
                
                # m3 = a(y-u)
                u3 = {(i0, j0): 1}
                v3 = {(j0, k0+1): 1, (j0+1, k0): -1}
                w3 = {(k0+1, i0): 1, (k0+1, i0+1): -1}
                pool.append((u3, v3, w3))
                
                # m4 = d(v-u)
                u4 = {(i0+1, j0+1): 1}
                v4 = {(j0+1, k0+1): 1, (j0+1, k0): -1}
                w4 = {(k0+1, i0+1): 1, (k0, i0+1): -1}
                pool.append((u4, v4, w4))
                
                # m5 = (a+b)v
                u5 = {(i0, j0): 1, (i0, j0+1): 1}
                v5 = {(j0+1, k0+1): 1}
                w5 = {(k0+1, i0): 1, (k0+1, i0+1): 1}
                pool.append((u5, v5, w5))
                
                # m6 = (c-a)(x+u)
                u6 = {(i0+1, j0): 1, (i0, j0): -1}
                v6 = {(j0, k0): 1, (j0+1, k0+1): 1}
                w6 = {(k0, i0+1): 1, (k0+1, i0+1): 1}
                pool.append((u6, v6, w6))
                
                # m7 = (b-d)(y+v)
                u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
                v7 = {(j0, k0+1): 1, (j0+1, k0+1): 1}
                w7 = {(k0+1, i0): 1, (k0+1, i0+1): 1}
                pool.append((u7, v7, w7))
    
    # Now we need to find a subset of pool terms whose sum equals T
    # This is a linear system: sum_t alpha_t * (u_t outer v_t outer w_t) = T
    # where alpha_t are scalars
    
    # Let's set up the linear system
    # Variables: alpha_t for each term t
    # Equations: for each (i,j,k), sum_t alpha_t * u_t[i,j] * v_t[j,k] * w_t[k,i] = T[i,j,k] = 1
    
    n_terms = len(pool)
    n_equations = n * m * p
    
    # Build coefficient matrix
    A_mat = np.zeros((n_equations, n_terms))
    for t in range(n_terms):
        u, v, w = pool[t]
        for i in range(n):
            for j in range(m):
                if (i, j) in u:
                    for k in range(p):
                        if (j, k) in v and (k, i) in w:
                            eq_idx = i * m * p + j * p + k
                            A_mat[eq_idx, t] = u[(i, j)] * v[(j, k)] * w[(k, i)]
    
    # We want to find a sparse solution to A_mat @ alpha = ones(n_equations)
    # Use greedy set cover: at each step, pick the term that covers the most uncovered equations
    
    target = np.ones(n_equations)
    covered = np.zeros(n_equations)
    selected = []
    
    while time.time() < deadline:
        # Find which equations are not yet covered
        uncovered = np.where(np.abs(covered - target) > 1e-10)[0]
        if len(uncovered) == 0:
            break
        
        # For each term, compute how much it helps cover uncovered equations
        best_term = -1
        best_score = -1
        for t in range(n_terms):
            if t in selected:
                continue
            # Compute contribution to uncovered equations
            contrib = A_mat[uncovered, t]
            # We want terms that have non-zero contribution to uncovered equations
            # Score: number of uncovered equations this term can help with
            score = np.sum(np.abs(contrib) > 1e-10)
            if score > best_score:
                best_score = score
                best_term = t
        
        if best_term == -1 or best_score == 0:
            # No term can help, break
            break
        
        selected.append(best_term)
        
        # Update covered: we need to solve for alpha values
        # For now, let's just track which terms are selected
        # and solve the linear system at the end
        
        # Check if we've selected too many terms
        if len(selected) >= n_equations:
            break
    
    # Solve for alpha values
    if len(selected) > 0:
        A_sel = A_mat[:, selected]
        # Solve A_sel @ alpha = target
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A_sel, target, rcond=None)
        except:
            alpha = np.zeros(len(selected))
        
        # Check if the solution is valid
        reconstruction = A_sel @ alpha
        if np.allclose(reconstruction, target, atol=1e-8):
            # Build U, V, W
            rank = len(selected)
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for t_idx, t in enumerate(selected):
                u, v, w = pool[t]
                for (i, j), coeff in u.items():
                    U[i*m+j, t_idx] = coeff * alpha[t_idx]
                for (j, k), coeff in v.items():
                    V[j*p+k, t_idx] = coeff * alpha[t_idx]
                for (k, i), coeff in w.items():
                    W[k*n+i, t_idx] = coeff * alpha[t_idx]
            
            # Verify
            C_recon = np.zeros((n, p))
            for t_idx in range(rank):
                u_vec = U[:, t_idx].reshape(n, m)
                v_vec = V[:, t_idx].reshape(m, p)
                w_vec = W[:, t_idx].reshape(p, n)
                C_recon += np.sum(u_vec * v_vec) * 0  # This isn't quite right
                # Actually, the reconstruction is:
                # C[i,k] = sum_j sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
                # Wait, that's not standard. Let me re-derive.
                # The tensor decomposition is: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
                # But we need C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
                # So the reconstruction should be:
                # C[i,k] = sum_j sum_t (U[i,j,t] * A[i,j]) * (V[j,k,t] * B[j,k]) * W[k,i,t]
                # This doesn't simplify nicely unless W[k,i,t] = 1 for the relevant terms.
            
            # Let me just return the schoolbook to be safe
            pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
