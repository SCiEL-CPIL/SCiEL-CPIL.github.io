import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R, V: 12 x R, W: 12 x R
    
    # Build target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def rationalize_matrix(M, tol=1e-5):
        rows, cols = M.shape
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                found = False
                for d in [1, 2]:
                    num = round(val * d)
                    frac = Fraction(num, d)
                    if abs(float(frac) - val) < tol:
                        result[i][j] = frac
                        found = True
                        break
                if not found:
                    for d in range(3, 11):
                        num = round(val * d)
                        frac = Fraction(num, d)
                        if abs(float(frac) - val) < tol:
                            result[i][j] = frac
                            found = True
                            break
                if not found:
                    return None
        return result

    def verify(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if (val * 2).denominator != 1:
                        return False
        
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0, 1)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
                        
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                if j1 != j2:
                    for c in range(p*n):
                        s = Fraction(0, 1)
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
                else:
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        if i1 == i2 and k1 == k2:
                            continue
                        else:
                            s = Fraction(0, 1)
                            for r in range(rank):
                                s += U[a][r] * V[b][r] * W[c][r]
                            if s != 0:
                                return False
        return True

    def fast_als_decompose(T, R, max_iters=80, tol=1e-8, seed=42):
        """ALS with precomputed Kronecker structure for speed."""
        np.random.seed(seed)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Precompute index mappings for efficient residual computation
        # For each a, the non-zero (b,c) pairs
        a_to_bc = {}
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a_to_bc[a] = [(j * p + k, k * n + i) for k in range(p)]
        
        # For each b, the non-zero (a,c) pairs
        b_to_ac = {}
        for j in range(m):
            for k in range(p):
                b = j * p + k
                b_to_ac[b] = [(i * m + j, k * n + i) for i in range(n)]
        
        # For each c, the non-zero (a,b) pairs
        c_to_ab = {}
        for k in range(p):
            for i in range(n):
                c = k * n + i
                c_to_ab[c] = [(i * m + j, j * p + k) for j in range(m)]
        
        err = float('inf')
        for it in range(max_iters):
            # Update U: for each a, solve least squares
            for a in range(n*m):
                bc_list = a_to_bc[a]
                A = np.zeros((len(bc_list), R))
                y = np.zeros(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A[idx, :] = V[b, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
            
            # Update V: for each b, solve least squares
            for b in range(m*p):
                ac_list = b_to_ac[b]
                A = np.zeros((len(ac_list), R))
                y = np.zeros(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A[idx, :] = U[a, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
            
            # Update W: for each c, solve least squares
            for c in range(p*n):
                ab_list = c_to_ab[c]
                A = np.zeros((len(ab_list), R))
                y = np.zeros(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A[idx, :] = U[a, :] * V[b, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence using sparse structure
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err)
            
            # Also check zero entries
            for a in range(n*m):
                i1 = a // m
                j1 = a % m
                for b in range(m*p):
                    j2 = b // p
                    k2 = b % p
                    if j1 != j2:
                        for c in range(p*n):
                            s = 0.0
                            for r in range(R):
                                s += U[a, r] * V[b, r] * W[c, r]
                            err += s * s
                    else:
                        for c in range(p*n):
                            k1 = c // n
                            i2 = c % n
                            if i1 != i2 or k1 != k2:
                                s = 0.0
                                for r in range(R):
                                    s += U[a, r] * V[b, r] * W[c, r]
                                err += s * s
            err = np.sqrt(err)
            
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    # Phase 1: Structured pebbling-inspired search
    # Key insight: T factors through j. For fixed (i,j), the k-dependence is separable.
    # Try patterns where U and W share structure across k for same (i,j).
    
    def try_structured_patterns(R, deadline):
        """Try structured patterns based on pebbling/tensor network slicing."""
        if time.time() > deadline:
            return None
        
        # Pattern 1: Group by (i,j) - use same U entry for all k, 
        # and let V,W handle the k-dependence
        # U[i*m+j, r] is constant in k for fixed r
        # This is naturally handled by the tensor structure
        
        # Pattern 2: Try binary patterns with specific sharing
        # For 3x3x4, try to find patterns where some terms are shared
        
        # Generate candidate patterns using the structure
        # T[a,b,c] = 1 iff i(a)=i(c) and j(a)=j(b) and k(b)=k(c)
        
        # Try random structured initializations
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(100 + trial)
            
            # Initialize with structure: U depends only on (i,j), 
            # V depends on (j,k), W depends on (k,i)
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Random half-integer initialization with structure
            for r in range(R):
                # Each term r should "cover" some subset of (i,j,k) triples
                # Assign random weights
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = np.random.choice([-1, 0, 0, 1])
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = np.random.choice([-1, 0, 0, 1])
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, r] = np.random.choice([-1, 0, 0, 1])
            
            # Refine with ALS
            U, V, W, err = fast_als_decompose(T, R, max_iters=40, tol=1e-6, seed=100+trial)
            
            if err < 1e-4:
                U_frac = rationalize_matrix(U, tol=1e-5)
                V_frac = rationalize_matrix(V, tol=1e-5)
                W_frac = rationalize_matrix(W, tol=1e-5)
                
                if U_frac and V_frac and W_frac:
                    if verify(U_frac, V_frac, W_frac):
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    # Phase 2: Fast ALS search
    def als_search(R, deadline, num_restarts=15):
        """Search for rank-R decomposition using fast ALS."""
        if time.time() > deadline:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            seed = 42 + restart * 7
            U, V, W, err = fast_als_decompose(T, R, max_iters=60, tol=1e-7, seed=seed)
            
            if err < 1e-4:
                U_frac = rationalize_matrix(U, tol=1e-5)
                V_frac = rationalize_matrix(V, tol=1e-5)
                W_frac = rationalize_matrix(W, tol=1e-5)
                
                if U_frac and V_frac and W_frac:
                    if verify(U_frac, V_frac, W_frac):
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    # Search for minimal R
    # Lower bound: max(n*m, m*p, p*n) = 12
    # But realistically, for 3x3x4, known bounds suggest rank around 24-36
    
    # Try from lower ranks up, but prioritize higher ranks first for time efficiency
    # Actually, we want minimal, so try from low to high
    
    # Time budget allocation:
    # - Spend ~60s on ranks 20-30 (most likely to have solutions)
    # - Spend ~40s on ranks 31-35
    
    best_result = None
    
    # First, try structured patterns for a few ranks
    for R in [24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 40:
            break
        
        result = try_structured_patterns(R, deadline - 40)
        if result is not None:
            return result
    
    # Then, try ALS for remaining ranks
    for R in range(20, 36):
        if time.time() > deadline - 10:
            break
        
        # Skip ranks already tried with structured patterns
        if R in [24, 25, 26, 27, 28, 29, 30] and time.time() > deadline - 30:
            continue
            
        result = als_search(R, deadline - 5, num_restarts=10)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
