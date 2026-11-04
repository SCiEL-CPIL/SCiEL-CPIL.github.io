import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    # And for all other combinations, the "tensor" is 0, but actually the constraint is only on the (i,j,k) triples.
    
    # Actually, the tensor M_{n,m,p} is defined on [n]x[m]x[p] with M[i,j,k]=1.
    # A rank-1 tensor is u⊗v⊗w where (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k].
    # We need sum_c u_c[i]*v_c[j]*w_c[k] = 1 for all (i,j,k) in [n]x[m]x[p].
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i,k,j] = 1 means output[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has indices (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # T[i,j,k] = 1 for all i,j,k in their ranges.
    # A rank-1 decomposition: sum_c u_c[i]*v_c[j]*w_c[k] = T[i,j,k] = 1 for all i,j,k.
    
    # But that would mean the tensor is all ones, which is rank 1! That's wrong.
    
    # I'm confusing myself. Let me re-read the problem.
    # "multiplying a 3 by 3 matrix by a 3 by 8 matrix"
    # A is 3x3, B is 3x8, C = A*B is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,k,j] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k]).
    # So T is a 3x8x3 tensor (output row, output col, middle index) with all entries 1.
    # No wait, T[i,k,j] = 1 for all i,k,j. That's a rank-1 tensor!
    
    # That can't be right either. Let me think again.
    
    # The matrix multiplication tensor M_{n,m,p} is an element of R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # where M_{n,m,p} = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # Here e_{i,j} is the basis vector for A[i,j], e_{j,k} for B[j,k], e_{k,i} for C[k,i] (or C[i,k] depending on convention).
    
    # The problem says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n+i) where k is the column of B (0..p-1) and i is the row of A (0..n-1).
    # So W[k*n+i, c] corresponds to output C[i,k].
    
    # So the tensor is: sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # where i in [0,n), j in [0,m), k in [0,p).
    
    # So T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.
    
    # This is the standard matrix multiplication tensor.
    
    # Now, a rank-1 decomposition term is u ⊗ v ⊗ w where:
    # sum_c u[a,c] * v[b,c] * w[cc,c] should equal T[a,b,cc]
    
    # So for each (a,b,cc), sum_c U[a,c]*V[b,c]*W[cc,c] = T[a,b,cc]
    
    # T[a,b,cc] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, cc=k*n+i, else 0.
    
    # Let's verify: for a=i*m+j, b=j*p+k, cc=k*n+i, we have T=1.
    # The number of such (a,b,cc) triples is n*m*p = 72.
    # Total possible triples: (n*m)*(m*p)*(p*n) = 9*24*24 = 5184.
    # So 72 entries are 1, rest are 0.
    
    # Now I need to find U (9×r), V (24×r), W (24×r) such that the above holds.
    
    # For the schoolbook: r=72, each term picks one (i,j,k).
    
    # Can we do better? Let's try to find a lower rank.
    
    # Strategy: for a given r, set up the system and try to solve it.
    # The equations are: for each (a,b,cc) in the 72 "1" positions:
    #   sum_c U[a,c]*V[b,c]*W[cc,c] = 1
    # For each (a,b,cc) NOT in the 72 positions:
    #   sum_c U[a,c]*V[b,c]*W[cc,c] = 0
    
    # This is a system of 5184 equations in 9*r + 24*r + 24*r = 57*r unknowns.
    # It's nonlinear (products of variables).
    
    # Approach: use iterative refinement. Start with random initialization and use gradient descent or similar.
    # Or: fix two matrices and solve for the third linearly.
    
    # Let me try: fix U and V randomly, then for each cc, the equation is:
    # sum_c U[a,c]*V[b,c]*W[cc,c] = T[a,b,cc]
    # For fixed U,V, this is linear in W. We have 5184 equations in 24*r unknowns (W entries).
    # But the equations are coupled through c.
    
    # Actually, for fixed U and V, define M[a,b,cc] = sum_c U[a,c]*V[b,c]*W[cc,c].
    # This is linear in W. We can set up a linear system for W.
    
    # Let's try this approach for various r.
    
    best_rank = n * m * p  # start with schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    rank_school = n * m * p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # Lower bound: the tensor rank of M_{3,3,8}
    # For M_{n,m,p}, a lower bound is max(n*m, m*p, n*p) = max(9, 24, 24) = 24
    # Also, Strassen's bound gives better lower bounds.
    # Let's try r from 24 to 71.
    
    for r in range(24, best_rank):
        if time.time() > deadline:
            break
        
        # Try to find a solution of rank r
        # Use random restarts
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = np.random.randn(nm, r)
            V_r = np.random.randn(mp, r)
            W_r = np.random.randn(pn, r)
            
            # Iteratively refine: fix two, solve for third
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W given U_r, V_r
                # For each cc in [0,pn), and for each (a,b):
                # sum_c U_r[a,c]*V_r[b,c]*W_r[cc,c] = T[a,b,cc]
                # This is a linear system in W_r[:, :].
                # Let's build the linear system.
                # Unknowns: W_r[cc, c] for cc in [0,pn), c in [0,r). Total pn*r unknowns.
                # Equations: for each (a,b,cc) in all (nm*mp*pn) combinations:
                # sum_c U_r[a,c]*V_r[b,c]*W_r[cc,c] = T[a,b,cc]
                
                # This is a big linear system. Let's build it.
                # Number of equations: nm*mp*pn = 9*24*24 = 5184
                # Number of unknowns: pn*r = 24*r
                
                # For r=24, unknowns = 576, equations = 5184. Overdetermined.
                # We can use least squares.
                
                # Build A matrix: A[equation_idx, unknown_idx]
                # equation_idx = (a*mp + b)*pn + cc
                # unknown_idx = cc*r + c
                
                A_mat = np.zeros((nm*mp*pn, pn*r))
                b_vec = np.zeros(nm*mp*pn)
                
                # This might be too slow to build each iteration. Let's precompute the structure.
                # Actually, let's just do it for now.
                
                eq_idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for cc in range(pn):
                            # Determine T[a,b,cc]
                            # a = i*m+j, b = j*p+k, cc = k'*n+i'
                            # T=1 if i=i', j=j', k=k'
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            i2 = cc % n
                            k2 = cc // n
                            T_val = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            b_vec[eq_idx] = T_val
                            
                            for c in range(r):
                                A_mat[eq_idx, cc*r + c] = U_r[a, c] * V_r[b, c]
                            eq_idx += 1
                
                # Solve least squares
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_r = W_flat.reshape(pn, r)
                
                # Check if solution is good
                # Compute error
                err = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for cc in range(pn):
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            i2 = cc % n
                            k2 = cc // n
                            T_val = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            pred = np.dot(U_r[a, :] * V_r[b, :], W_r[cc, :])
                            err += (pred - T_val)**2
                
                if err < 1e-6:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            # Try to go lower
            continue
    
    # Convert to list format
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
