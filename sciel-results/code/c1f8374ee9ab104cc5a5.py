import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j (A index), b = j*p + k (B index), c = k*n + i (C index)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Fallback: Rank 28 Strassen solution
    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen_products = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            k2 = 2 * q + 1
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        T_calc = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_calc - T))
        return err < tol, err
    
    # Start with rank 28 as baseline
    best_rank = 28
    best_U, best_V, best_W = get_strassen_28()
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: rank >= 24 (from various lower bound arguments)
    # Try ranks from 24 up to 27
    
    for target_rank in [24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        
        # Use multiple random restarts with gradient-based optimization
        np.random.seed(42)
        found = False
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            sizes = [n*m*target_rank, m*p*target_rank, p*n*target_rank]
            
            def loss(x):
                U = x[:sizes[0]].reshape(n*m, target_rank)
                V = x[sizes[0]:sizes[0]+sizes[1]].reshape(m*p, target_rank)
                W = x[sizes[0]+sizes[1]:].reshape(p*n, target_rank)
                T_calc = np.einsum('at,bt,ct->abc', U, V, W)
                err = T_calc - T
                return np.sum(err**2)
            
            def grad(x):
                U = x[:sizes[0]].reshape(n*m, target_rank)
                V = x[sizes[0]:sizes[0]+sizes[1]].reshape(m*p, target_rank)
                W = x[sizes[0]+sizes[1]:].reshape(p*n, target_rank)
                T_calc = np.einsum('at,bt,ct->abc', U, V, W)
                err = T_calc - T
                # Gradient: d/dU = err contracted with V,W
                gU = np.einsum('abc,bt,ct->at', err, V, W)
                gV = np.einsum('abc,at,ct->bt', err, U, W)
                gW = np.einsum('abc,at,bt->ct', err, U, V)
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            # Use L-BFGS-B with gradient
            res = minimize(loss, x0, jac=grad, method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15})
            
            if res.success or res.fun < 1e-10:
                U_opt = res.x[:sizes[0]].reshape(n*m, target_rank)
                V_opt = res.x[sizes[0]:sizes[0]+sizes[1]].reshape(m*p, target_rank)
                W_opt = res.x[sizes[0]+sizes[1]:].reshape(p*n, target_rank)
                valid, err = verify(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    break
        
        if found:
            break
    
    # Final verification
    valid, err = verify(best_U, best_V, best_W)
    if not valid:
        # Fallback to rank 28
        best_U, best_V, best_W = get_strassen_28()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
