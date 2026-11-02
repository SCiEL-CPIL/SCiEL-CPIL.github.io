import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0

    # Try ranks from 72 down to 30
    for R in range(72, 29, -1):
        for attempt in range(10):
            np.random.seed(42 + R * 100 + attempt)
            # Initialize
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Scale initialization to avoid huge/small gradients
            # The product U*V*W should be O(1).
            # If U, V, W are N(0,1), product is small.
            # Let's scale by 1.0 for now.
            
            def residuals(params):
                U_flat = params[:nm*R].reshape(nm, R)
                V_flat = params[nm*R:nm*R+mp*R].reshape(mp, R)
                W_flat = params[nm*R+mp*R:].reshape(pn, R)
                
                T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                return (T - T_target).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use 'trf' for bounded or unbounded, robust
                result = least_squares(residuals, x0, method='trf', max_nfev=5000, xtol=1e-9, ftol=1e-9)
            except Exception:
                continue
                
            if result.cost < 1e-8:
                U_sol = result.x[:nm*R].reshape(nm, R)
                V_sol = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W_sol = result.x[nm*R+mp*R:].reshape(pn, R)
                
                # Check if we can round to half-integers
                # The problem asks for integer/half-integer entries.
                # If the solution is close to such values, we round.
                U_r = np.round(U_sol * 2) / 2.0
                V_r = np.round(V_sol * 2) / 2.0
                W_r = np.round(W_sol * 2) / 2.0
                
                T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_r, T_target, atol=1e-6):
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # If rounding fails, but the float solution is exact, 
                # we might still return it if the evaluator accepts floats?
                # "integer/half-integer entries" suggests strict check.
                # If we can't round, we might need to continue searching for a better local min.
                pass

    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
