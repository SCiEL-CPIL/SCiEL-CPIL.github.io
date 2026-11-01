import time
import numpy as np
import sympy as sp
from scipy.optimize import root

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _algebraic_search(n, m, p, deadline):
    """
    Strategy:
    1. Assume a rank r decomposition exists.
    2. The tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    3. We want T[i,j,k] = 1 for all i,j,k.
    
    For small n, m, p, we can treat U and V as variables to be eliminated.
    However, symbolic elimination of 60 variables is too slow.
    
    Instead, we use the structure:
    For fixed r, let u_{ij,r}, v_{jk,r}, w_{ki,r}.
    
    We can fix W to specific values (e.g., random or structured) and solve for U, V?
    No, U and V are also unknowns.
    
    Let's try a different algebraic approach:
    The equations are linear in U, V, W separately.
    
    We can use the "border rank" or "tensor rank" properties.
    But let's stick to a computational algebra approach using sympy for small r.
    
    For 2x3x10, rank 60 is trivial. Can we do better?
    Known lower bounds:
    rank(2x3x1) >= 5.
    rank(2x3x10) >= ?
    
    Let's try to solve for rank r = 30, 35, ... using scipy root on the flattened system, 
    but with a smarter initialization derived from the algebraic structure.
    
    Actually, the previous gradient descent was slow. 
    Let's try using `scipy.optimize.root` on the full system of equations F(U, V, W) = 0.
    The system is:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1 = 0 for all i,j,k.
    
    Number of equations: n*m*p = 60.
    Number of variables: (n*m + m*p + p*n) * r = (6 + 30 + 20) * r = 56 * r.
    
    For r < 60, the system is underdetermined.
    We can set some variables to 0 or 1 to reduce the dimension, or just let scipy find a solution.
    However, `root` requires N equations and N variables (or handles underdetermined with 'lm' method? No, lm is for least squares).
    
    Since it's underdetermined, we can fix a subset of variables to break the scaling symmetry (U, V, W are invariant under scaling).
    Specifically, U * alpha, V * alpha^-1, W * 1 is a symmetry. And permutations of r.
    
    To use `root`, we need a square system. We can add constraints:
    1. The 60 tensor equations.
    2. 56*r - 60 additional constraints to fix the gauge.
       E.g., fix the first column of U to 1, 0, ..., or fix norms.
       
    This is complex.
    
    Alternative: Use the fact that for matrix multiplication, the "standard" decomposition is rank nmp.
    There are known algorithms (Strassen-like) that reduce rank.
    
    Let's try a specific known lower bound search.
    For 2x3x1, rank is 5.
    For 2x3x10, we can view it as 10 instances of 2x3x1 coupled? No, the middle dimension is shared.
    
    Let's try a randomized algebraic approach:
    1. Pick r.
    2. Initialize U, V, W randomly.
    3. Use `scipy.optimize.least_squares` to minimize the residual of the tensor equations.
    4. If residual is 0, we have a solution.
    
    This is similar to the previous gradient descent but `least_squares` is more robust and uses Jacobians (or approximations).
    """
    num_U = n * m
    num_V = m * p
    num_W = p * n
    num_eq = n * m * p
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
                
    def residual(x):
        # x is flattened [U, V, W]
        U = x[:num_U*r].reshape(num_U, r)
        V = x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
        W = x[num_U*r+num_V*r:].reshape(num_W, r)
        
        res = np.zeros(num_eq)
        for idx, (iu, iv, iw) in enumerate(indices):
            res[idx] = np.sum(U[iu] * V[iv] * W[iw]) - 1.0
        return res

    def jac(x):
        # Jacobian of residual w.r.t x
        # res_e = sum_r U[iu_e, r] V[iv_e, r] W[iw_e, r] - 1
        # d res_e / d U[iu, r] = V[iv, r] * W[iw, r]
        # etc.
        U = x[:num_U*r].reshape(num_U, r)
        V = x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
        W = x[num_U*r+num_V*r:].reshape(num_W, r)
        
        J = np.zeros((num_eq, (num_U+num_V+num_W)*r))
        
        for e, (iu, iv, iw) in enumerate(indices):
            # Derivatives w.r.t U
            for rr in range(r):
                J[e, iu*r + rr] = V[iv, rr] * W[iw, rr]
            # Derivatives w.r.t V
            for rr in range(r):
                J[e, num_U*r + iv*r + rr] = U[iu, rr] * W[iw, rr]
            # Derivatives w.r.t W
            for rr in range(r):
                J[e, (num_U+num_V)*r + iw*r + rr] = U[iu, rr] * V[iv, rr]
        return J

    best_result = None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for 2x3x10 is not trivially known, but likely < 60.
    # Let's start from 30 (half of schoolbook) and go up.
    # To save time, we'll try a few ranks.
    
    for r in range(30, 61):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this rank
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + r * 100 + attempt)
            # Initialize with small random values
            x0 = np.random.randn((num_U + num_V + num_W) * r) * 0.1
            
            try:
                sol = root(residual, x0, jac=jac, method='hybr', tol=1e-12)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-9:
                    # Extract solution
                    U_sol = sol.x[:num_U*r].reshape(num_U, r)
                    V_sol = sol.x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
                    W_sol = sol.x[num_U*r+num_V*r:].reshape(num_W, r)
                    
                    # Convert to lists for verification and return
                    U_list = U_sol.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_sol.tolist()
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best_result = (U_list, V_list, W_list)
                        found = True
                        break
            except Exception:
                pass
                
        if found:
            break
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook solution as fallback
    U, V, W = _schoolbook(n, m, p)
    
    # Try algebraic search
    result = _algebraic_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        
    return {"U": U, "V": V, "W": W}
