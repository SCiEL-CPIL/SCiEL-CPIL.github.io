import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 4, 5]

def construct():
        n, m, p = DIMENSIONS
        # Target tensor Y
        # Shape: (n*m, m*p, p*n)
        Y = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    Y[a, b, c] = 1.0

        # Try ranks from 45 upwards
        # Start with 45 to save time
        for rank in [45, 50, 55]:
            if time.time() > 100: break
            
            # Random init
            np.random.seed(42)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            def objective(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                # Compute tensor product sum
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # This is slow if done naively.
                # Use einsum
                T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                diff = T - Y
                return np.sum(diff * diff)
            
            def gradient(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                diff = T - Y
                
                # Gradients
                # dT/da[r] = V[b,r] W[c,r]
                # grad_U[a,r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
                # grad_U = einsum('abc,br,cr->ar', diff, V_flat, W_flat)
                gU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
                gV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
                gW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
                
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                               options={'maxiter': 500, 'maxfun': 500, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    # Extract and round to half-integers
                    U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                    V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                    W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                    
                    # Round to nearest half-integer
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    
                    # Verify
                    T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    if np.max(np.abs(T_check - Y)) < 1e-9:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception as e:
                pass

        # Fallback to schoolbook
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
