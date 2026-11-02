import numpy as np
from scipy.optimize import fsolve

def construct():
    n, m, p = 2, 2, 5
    # Support points
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    # Try rank 16
    rank = 16
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    def equations(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        eqs = []
        # Support constraints: sum should be 1
        for (a, b, c) in support:
            val = np.sum(U[a] * V[b] * W[c])
            eqs.append(val - 1)
        
        # Non-support constraints: sum should be 0
        all_points = set()
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    all_points.add((a, b, c))
        
        support_set = set(support)
        non_support = [pt for pt in all_points if pt not in support_set]
        
        for (a, b, c) in non_support:
            val = np.sum(U[a] * V[b] * W[c])
            eqs.append(val)
        
        return eqs
    
    # This is a large system, let's try to solve it
    # Number of variables: 24*16 = 384
    # Number of equations: 400
    # Overdetermined, so we need least squares
    
    from scipy.optimize import least_squares
    
    def residual(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        res = []
        for (a, b, c) in support:
            val = np.sum(U[a] * V[b] * W[c])
            res.append(val - 1)
        
        all_points = set()
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    all_points.add((a, b, c))
        
        support_set = set(support)
        non_support = [pt for pt in all_points if pt not in support_set]
        
        for (a, b, c) in non_support:
            val = np.sum(U[a] * V[b] * W[c])
            res.append(val)
        
        return res
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    result = least_squares(residual, x0, method='trf', max_nfev=10000)
    
    if result.success or np.linalg.norm(result.fun) < 1e-6:
        U_sol = result.x[:n*m*rank].reshape(n*m, rank)
        V_sol = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W_sol = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        # Convert to lists
        U_list = U_sol.tolist()
        V_list = V_sol.tolist()
        W_list = W_sol.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
