import scipy.optimize as so

def _solve_with_scipy(U, V, n, m, p, deadline):
    R = len(U[0])
    nm, mp = n * m, m * p
    
    # Flatten U and V into a single vector
    x0 = np.concatenate([np.array(U).flatten(), np.array(V).flatten()])
    
    def residual(x):
        U_flat = x[:nm*R]
        V_flat = x[nm*R:]
        U_mat = U_flat.reshape(nm, R)
        V_mat = V_flat.reshape(mp, R)
        
        # Compute residual for all (i,j,k)
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        # Residual[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] - delta_{j,k}
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i*m + j]
                    v_row = V_mat[j*p + k]
                    val = np.dot(u_row, v_row)
                    target = 1.0 if j == k else 0.0
                    res.append(val - target)
        return np.array(res)
    
    # Use least_squares
    try:
        sol = so.least_squares(residual, x0, method='lm', max_nfev=2000)
        x_sol = sol.x
        U_sol = x_sol[:nm*R].reshape(nm, R).tolist()
        V_sol = x_sol[nm*R:].reshape(mp, R).tolist()
        sse = np.sum(sol.fun**2)
        return U_sol, V_sol, float(sse)
    except Exception:
        return U, V, float('inf')
