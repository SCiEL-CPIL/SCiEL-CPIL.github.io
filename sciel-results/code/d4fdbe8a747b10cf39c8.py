"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
import sys
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook: rank 120
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    # Try to find lower-rank decompositions
    def try_rank(R, time_budget):
        """Try to find exact rank-R decomposition via optimization."""
        deadline = time.time() + time_budget
        
        # Use random initialization with symmetry breaking
        best_sol = None
        best_obj = float('inf')
        
        n_restarts = 0
        while time.time() < deadline - 5:
            n_restarts += 1
            if n_restarts > 30:
                break
            
            # Initialize with structured random values
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Add some structure: make certain entries nonzero
            for c in range(R):
                # Pick random i,j,k patterns
                for _ in range(3):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    U[i * m + j, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                    V[j * p + k, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                    W[k * n + i, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
            
            # Objective: minimize max over (i,j,k) of |sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1|
            # We minimize the sum of squares of violations for smoothness
            def obj(x):
                Ux = x[:nm*R].reshape(nm, R)
                Vx = x[nm*R:nm*R+mp*R].reshape(mp, R)
                Wx = x[nm*R+mp*R:][:, :R].reshape(pn, R)
                
                # Compute all tensor entries
                # T[i,j,k] = sum_r Ux[i*m+j, r] * Vx[j*p+k, r] * Wx[k*n+i, r]
                # For efficiency, compute via broadcasting
                viol = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += Ux[i*m+j, r] * Vx[j*p+k, r] * Wx[k*n+i, r]
                            viol[i,j,k] = s - 1.0
                
                return float(np.sum(viol * viol))
            
            # Compute gradient (analytical)
            def grad(x):
                Ux = x[:nm*R].reshape(nm, R).copy()
                Vx = x[nm*R:nm*R+mp*R].reshape(mp, R).copy()
                Wx = x[nm*R+mp*R:][:, :R].reshape(pn, R).copy()
                
                dU = np.zeros_like(Ux)
                dV = np.zeros_like(Vx)
                dW = np.zeros_like(Wx)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r Ux[i*m+j, r] * Vx[j*p+k, r] * Wx[k*n+i, r]
                            # dT/dU[i*m+j, r] = Vx[j*p+k, r] * Wx[k*n+i, r]
                            # dT/dV[j*p+k, r] = Ux[i*m+j, r] * Wx[k*n+i, r]
                            # dT/dW[k*n+i, r] = Ux[i*m+j, r] * Vx[j*p+k, r]
                            for r in range(R):
                                t_val = 0.0
                                for rr in range(R):
                                    t_val += Ux[i*m+j, rr] * Vx[j*p+k, rr] * Wx[k*n+i, rr]
                                diff = t_val - 1.0  # 2 * (T-1) * ...
                                dU[i*m+j, r] += 2.0 * diff * Vx[j*p+k, r] * Wx[k*n+i, r]
                                dV[j*p+k, r] += 2.0 * diff * Ux[i*m+j, r] * Wx[k*n+i, r]
                                dW[k*n+i, r] += 2.0 * diff * Ux[i*m+j, r] * Vx[j*p+k, r]
                
                gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return gx
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                              options={'maxiter': 200, 'maxfun': 5000,
                                       'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                fval = res.fun
                if fval < best_obj:
                    best_obj = fval
                    x_opt = res.x
                    U_opt = x_opt[:nm*R].reshape(nm, R)
                    V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = x_opt[nm*R+mp*R:][:, :R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)
            except Exception:
                pass
            
            # Check if we're close to exact
            if best_obj < 1e-14:
                break
        
        # Check if solution is exact
        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            # Verify exactness with higher precision
            max_viol = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U_opt[i*m+j, r] * V_opt[j*p+k, r] * W_opt[k*n+i, r]
                        max_viol = max(max_viol, abs(s - 1.0))
            
            if max_viol < 1e-12:
                # Convert to list of lists
                U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
                V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
                W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # Known: schoolbook gives 120. Can we do better?
    # For 4x5x6, the rank is at least ceil(4*5*6/2) by some bounds, but let's try practical values.
    # Actually, let's try a range of ranks.
    
    ranks_to_try = [48, 54, 60, 64, 72, 80, 90, 100, 110, 119]
    
    for R in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        time_left = DEADLINE - time.time() - 5
        if time_left < 5:
            break
        
        result = try_rank(R, min(time_left, 40))
        if result is not None:
            best = result
            break
    
    return best
