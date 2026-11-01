import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2 x 2
    # B is m x p = 2 x 4
    # C is n x p = 2 x 4, but indexed as W[k*n+i, r] per the trace convention
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) = (8, rank), indexed by k*n+i where k is column of C, i is row of C.
    
    # The tensor: T_{(i,j),(j',k),(k',i')} = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # where (i,j) is A index (i=row of A, j=col of A), (j',k) is B index, (k',i') is C index
    # C index: k' is column of C, i' is row of C. W index = k'*n + i'
    
    # Constraint: sum_r u_r[i*m+j] * v_r[j'*p+k] * w_r[k'*n+i'] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    
    deadline = time.time() + 140  # safety margin
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Generate the constraint equations
    # Non-trivial: i=i', j=j', k=k' -> sum_r u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # Trivial: otherwise -> sum_r u[i*m+j] * v[j'*p+k] * w[k'*n+i'] = 0
    
    # Let's set up for a given rank r
    def solve_for_rank(r, max_iter=200, tol=1e-8):
        if time.time() > deadline - 10:
            return None
        
        # Variables: for each rank term, u (length n*m), v (length m*p), w (length p*n)
        # Total variables: r * (n*m + m*p + p*n)
        total_vars = r * (n * m + m * p + p * n)
        
        # Initialize with a structured guess
        x0 = np.zeros(total_vars)
        # Try to initialize with schoolbook-like structure
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < total_vars:
                        # u[i*m+j]
                        x0[col] = 1.0
                        col += 1
                    if col < total_vars:
                        # v[j*p+k]
                        x0[col] = 1.0
                        col += 1
                    if col < total_vars:
                        # w[k*n+i]
                        x0[col] = 1.0
                        col += 1
                    col += 1
        
        # Actually, let me use a better initialization
        # For rank r, try to use a pattern based on Strassen
        x0 = np.random.randn(total_vars) * 0.1
        
        def residuals(x):
            res = []
            idx = 0
            for r_idx in range(r):
                u = x[idx:idx+n*m].reshape(n, m)
                idx += n*m
                v = x[idx:idx+m*p].reshape(m, p)
                idx += m*p
                w = x[idx:idx+p*n].reshape(p, n)
                idx += p*n
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Non-trivial: sum_r u[i,j] * v[j,k] * w[k,i] = 1
                            val = 0.0
                            for r2 in range(r):
                                u2 = x[r2*(n*m+m*p+p*n):r2*(n*m+m*p+p*n)+n*m].reshape(n, m)
                                v2 = x[r2*(n*m+m*p+p*n)+n*m:(r2+1)*(n*m+m*p+p*n)-p*n].reshape(m, p)
                                w2 = x[(r2+1)*(n*m+m*p+p*n)-p*n:(r2+1)*(n*m+m*p+p*n)].reshape(p, n)
                                val += u2[i,j] * v2[j,k] * w2[k,i]
                            res.append(val - 1.0)
                        
                        # Trivial equations
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if i2 == i and j2 == j and k2 == k:
                                        continue
                                    val = 0.0
                                    for r2 in range(r):
                                        u2 = x[r2*(n*m+m*p+p*n):r2*(n*m+m*p+p*n)+n*m].reshape(n, m)
                                        v2 = x[r2*(n*m+m*p+p*n)+n*m:(r2+1)*(n*m+m*p+p*n)-p*n].reshape(m, p)
                                        w2 = x[(r2+1)*(n*m+m*p+p*n)-p*n:(r2+1)*(n*m+m*p+p*n)].reshape(p, n)
                                        val += u2[i2,j2] * v2[j2,k2] * w2[k2,i2]
                                    res.append(val)
            
            return np.array(res)
        
        # This is too slow. Let me vectorize.
        return None
    
    # Let me use a vectorized approach
    def solve_for_rank_v2(r):
        if time.time() > deadline - 10:
            return None
        
        total_vars = r * (n * m + m * p + p * n)
        
        # Precompute the constraint structure
        # For each equation (i,j,k,i2,j2,k2), the coefficient of u_r2[i2,j2] * v_r2[j2,k2] * w_r2[k2,i2] is 1
        # This is a cubic system. Let me use a different approach.
        
        # Fix one set of variables and solve linearly for others.
        # For example, fix all u and v, solve for w.
        # The equation: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
        # For fixed u,v: sum_r (u_r[i,j] * v_r[j,k]) * w_r[k,i] = target
        # This is linear in w!
        
        # Let's use this: iterate: fix u,v, solve for w; fix u,w, solve for v; fix v,w, solve for u.
        
        # Initialize
        u = np.random.randn(r, n, m) * 0.5
        v = np.random.randn(r, m, p) * 0.5
        w = np.random.randn(r, p, n) * 0.5
        
        for iteration in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Solve for w: for each (i,k,i'), sum_r u_r[i,j] v_r[j,k] w_r[k,i'] = δ_{i,i'} (for j=j', k=k')
            # Actually the full constraint: sum_r u_r[i2,j2] v_r[j2,k2] w_r[k2,i2'] = δ_{i2,i2'}δ_{j2,j2'}δ_{k2,k2'}
            # For fixed u,v: sum_r (u_r[i2,j2] v_r[j2,k2]) w_r[k2,i2'] = target[i2,j2,k2,i2']
            # This is a linear system in w.
            
            # Build the linear system for w
            # Unknowns: w_r[k2, i2'] for r=0..r-1, k2=0..p-1, i2'=0..n-1 -> r*p*n unknowns
            # Equations: for each (i2,j2,k2,i2'): sum_r u_r[i2,j2] v_r[j2,k2] w_r[k2,i2'] = δ_{i2,i2'}δ_{j2,j2'}δ_{k2,k2'}
            
            num_w_unknowns = r * p * n
            num_eqs = n * m * p * n  # (i2, j2, k2, i2')
            
            A_mat = np.zeros((num_eqs, num_w_unknowns))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        for i2p in range(n):
                            target = 1.0 if (i2 == i2p and j2 == j2 and k2 == k2) else 0.0
                            # Wait, the target is δ_{i2,i2'}δ_{j2,j2'}δ_{k2,k2'}
                            # But j2 and k2 are fixed in the equation index. The equation is for specific (i2,j2,k2,i2').
                            # The target: δ_{i2,i2'} * δ_{j2,j2'} * δ_{k2,k2'}
                            # But j2 and k2 are the same in both, so δ_{j2,j2'}=1 and δ_{k2,k2'}=1 always.
                            # So target = δ_{i2,i2'}
                            target = 1.0 if i2 == i2p else 0.0
                            b_vec[eq_idx] = target
                            for r_idx in range(r):
                                coeff = u[r_idx, i2, j2] * v[r_idx, j2, k2]
                                w_idx = r_idx * p * n + k2 * n + i2p
                                A_mat[eq_idx, w_idx] = coeff
                            eq_idx += 1
            
            # Solve least squares
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            w = w_flat.reshape(r, p, n)
            
            # Solve for v: fix u, w
            num_v_unknowns = r * m * p
            A_mat = np.zeros((num_eqs, num_v_unknowns))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        for i2p in range(n):
                            target = 1.0 if i2 == i2p else 0.0
                            b_vec[eq_idx] = target
                            for r_idx in range(r):
                                coeff = u[r_idx, i2, j2] * w[r_idx, k2, i2p]
                                v_idx = r_idx * m * p + j2 * p + k2
                                A_mat[eq_idx, v_idx] = coeff
                            eq_idx += 1
            
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            v = v_flat.reshape(r, m, p)
            
            # Solve for u: fix v, w
            num_u_unknowns = r * n * m
            A_mat = np.zeros((num_eqs, num_u_unknowns))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        for i2p in range(n):
                            target = 1.0 if i2 == i2p else 0.0
                            b_vec[eq_idx] = target
                            for r_idx in range(r):
                                coeff = v[r_idx, j2, k2] * w[r_idx, k2, i2p]
                                u_idx = r_idx * n * m + i2 * m + j2
                                A_mat[eq_idx, u_idx] = coeff
                            eq_idx += 1
            
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            u = u_flat.reshape(r, n, m)
        
        # Check the solution
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += u[r_idx, i, j] * v[r_idx, j, k] * w[r_idx, k, i]
                    max_err = max(max_err, abs(val - 1.0))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                val = 0.0
                                for r_idx in range(r):
                                    val += u[r_idx, i2, j2] * v[r_idx, j2, k2] * w[r_idx, k2, i2]
                                max_err = max(max_err, abs(val))
        
        if max_err < 1e-6:
            return u, v, w, max_err
        return None
    
    # Try ranks from 12 to 16
    for r in range(12, 17):
        if time.time() > deadline - 15:
            break
        result = solve_for_rank_v2(r)
        if result is not None:
            u, v, w, err = result
            best_rank = r
            best_solution = (u, v, w)
            break
    
    if best_solution is None:
        # Fall back to schoolbook
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
    
    u, v, w = best_solution
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r_idx in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r_idx] = u[r_idx, i, j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r_idx] = v[r_idx, j, k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r_idx] = w[r_idx, k, i]
    
    return {"U": U, "V": V, "W": W}
