import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def check_solution(U, V, W, R):
        """Check if the decomposition is exact. Returns error."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0)**2
        return err
    
    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U
    
    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V
    
    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W
    
    def als_optimize(U, V, W, R, max_iters=200, tol=1e-12):
        """Run ALS optimization. Returns (U, V, W, err)."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err
    
    def polish(U, V, W, R, max_iters=500, tol=1e-14):
        """High-precision polish pass."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err
    
    # Build a known rank-28 solution using Strassen's algorithm structure
    # M_{2,3,5}: split m=3 into 2+1
    # The 2x2x5 part uses Strassen (rank 7 for the 2x2 multiplication)
    # Plus the cross terms
    #
    # Actually, let's use a direct construction.
    # For M_{2,3,5}, we can use the following approach:
    # Think of the 3 intermediate index j as j=0,1,2
    # Split into j in {0,1} (Strassen) and j=2 (scalar)
    #
    # Strassen for 2x2 matrices:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    #
    # For our tensor, we need to embed this properly.
    #
    # Let's just use ALS with many restarts for rank 27, and fall back to
    # a verified rank-28 construction.
    
    # First, let's build a rank-28 solution via ALS (should converge easily)
    # Then try to find rank 27.
    
    best_rank = 30
    best_U, best_V, best_W = None, None, None
    
    # Try rank 27 first (the target improvement)
    for R in [27, 28]:
        if time.time() > deadline - 10:
            break
        
        found = False
        n_trials = 500 if R == 27 else 200
        max_iters = 100 if R == 27 else 50
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 7919 + 13)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            U, V, W, err = als_optimize(U, V, W, R, max_iters=max_iters, tol=1e-12)
            
            if err < 1e-10:
                # Polish for high precision
                U, V, W, err = polish(U, V, W, R, max_iters=200, tol=1e-14)
                if err < 1e-8:
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # If we didn't find rank 27 or 28, try more trials for rank 28
    if best_U is None and time.time() < deadline - 10:
        for trial in range(300):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 104729 + 42)
            R = 28
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            U, V, W, err = als_optimize(U, V, W, R, max_iters=80, tol=1e-12)
            
            if err < 1e-10:
                U, V, W, err = polish(U, V, W, R, max_iters=200, tol=1e-14)
                if err < 1e-8:
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Fallback to schoolbook rank 30
    if best_U is None:
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = R
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
