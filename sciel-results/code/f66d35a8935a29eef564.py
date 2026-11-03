import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
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

    def verify_exact(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        U_int = [[int(round(2*val)) for val in row] for row in U]
        V_int = [[int(round(2*val)) for val in row] for row in V]
        W_int = [[int(round(2*val)) for val in row] for row in W]
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                    if s != 8:
                        return False
        
        # Check zero entries
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                if j1 != j2:
                    for c in range(p*n):
                        s = 0
                        for r in range(rank):
                            s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                        if s != 0:
                            return False
                else:
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        if i1 == i2 and k1 == k2:
                            continue
                        else:
                            s = 0
                            for r in range(rank):
                                s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                            if s != 0:
                                return False
        return True

    def fast_als_decompose(T, R, max_iters=50, tol=1e-9, seed=42, use_snap=True, init_type='random'):
        np.random.seed(seed)
        
        if init_type == 'tt':
            # Tensor Train inspired initialization
            # Initialize with structured patterns based on tensor train core tensors
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Create small bond dimensions for TT-like structure
            # This is a heuristic initialization that exploits the structure
            # of matrix multiplication as a sequence of operations
            
            # Initialize with a mix of identity-like and random patterns
            for r in range(R):
                # Pattern 1: Identity-like
                if r < n*m:
                    U[r, r] = 1.0
                    if r < m*p:
                        V[r, r] = 1.0
                    if r < p*n:
                        W[r, r] = 1.0
                # Pattern 2: Cyclic
                elif r < 2*n*m:
                    idx = r - n*m
                    U[idx % (n*m), r] = 1.0
                    if idx < m*p:
                        V[idx % (m*p), r] = 1.0
                    if idx < p*n:
                        W[idx % (p*n), r] = 1.0
                # Pattern 3: Random with structure
                else:
                    U[r % (n*m), r] = np.random.choice([-1.0, 1.0])
                    if r < m*p:
                        V[r % (m*p), r] = np.random.choice([-1.0, 1.0])
                    if r < p*n:
                        W[r % (p*n), r] = np.random.choice([-1.0, 1.0])
            
            # Add some random noise to break symmetry
            U += np.random.randn(n*m, R) * 0.1
            V += np.random.randn(m*p, R) * 0.1
            W += np.random.randn(p*n, R) * 0.1
            
            if use_snap:
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0
        else:
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            if use_snap:
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0

        # Precompute index mappings
        a_to_bc = {}
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a_to_bc[a] = [(j * p + k, k * n + i) for k in range(p)]
        
        b_to_ac = {}
        for j in range(m):
            for k in range(p):
                b = j * p + k
                b_to_ac[b] = [(i * m + j, k * n + i) for i in range(n)]
        
        c_to_ab = {}
        for k in range(p):
            for i in range(n):
                c = k * n + i
                c_to_ab[c] = [(i * m + j, j * p + k) for j in range(m)]
        
        err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            for a in range(n*m):
                bc_list = a_to_bc[a]
                A = np.zeros((len(bc_list), R))
                y = np.zeros(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A[idx, :] = V[b, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                if use_snap:
                    U[a, :] = np.round(U[a, :] * 2) / 2.0

            # Update V
            for b in range(m*p):
                ac_list = b_to_ac[b]
                A = np.zeros((len(ac_list), R))
                y = np.zeros(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A[idx, :] = U[a, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                if use_snap:
                    V[b, :] = np.round(V[b, :] * 2) / 2.0

            # Update W
            for c in range(p*n):
                ab_list = c_to_ab[c]
                A = np.zeros((len(ab_list), R))
                y = np.zeros(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A[idx, :] = U[a, :] * V[b, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
                if use_snap:
                    W[c, :] = np.round(W[c, :] * 2) / 2.0

            # Calculate error
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
            
            if err < tol:
                # Check zero entries
                zero_err = 0.0
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
                                zero_err += s * s
                        else:
                            for c in range(p*n):
                                k1 = c // n
                                i2 = c % n
                                if i1 != i2 or k1 != k2:
                                    s = 0.0
                                    for r in range(R):
                                        s += U[a, r] * V[b, r] * W[c, r]
                                    zero_err += s * s
                if zero_err < tol:
                    return U, V, W, err
                else:
                    err += np.sqrt(zero_err)
                    
        return U, V, W, err

    def local_search_polish(U, V, W, R, max_steps=100, deadline=deadline):
        """
        Local search on the half-integer lattice to polish the solution.
        Tries small perturbations and keeps improvements.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        def compute_error(U, V, W):
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
            # Zero entries
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
            return np.sqrt(err)
        
        current_err = compute_error(U, V, W)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Try perturbing one entry at a time
            improved = False
            np.random.shuffle(range(n*m))
            np.random.shuffle(range(m*p))
            np.random.shuffle(range(p*n))
            
            for a in np.random.choice(n*m, size=min(3, n*m), replace=False):
                for r in range(R):
                    old_val = U[a, r]
                    for delta in [-1.0, 1.0]:
                        U[a, r] = old_val + delta
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-9:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[a, r] = old_val
                if improved:
                    break
            
            for b in np.random.choice(m*p, size=min(3, m*p), replace=False):
                for r in range(R):
                    old_val = V[b, r]
                    for delta in [-1.0, 1.0]:
                        V[b, r] = old_val + delta
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-9:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[b, r] = old_val
                if improved:
                    break
            
            for c in np.random.choice(p*n, size=min(3, p*n), replace=False):
                for r in range(R):
                    old_val = W[c, r]
                    for delta in [-1.0, 1.0]:
                        W[c, r] = old_val + delta
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-9:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[c, r] = old_val
                if improved:
                    break
            
            if not improved:
                break
                
        return U, V, W, current_err

    def try_rank(R, deadline, num_restarts=20):
        if time.time() > deadline:
            return None
            
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = restart * 13 + 7
            init_type = 'tt' if restart % 2 == 0 else 'random'
            U, V, W, err = fast_als_decompose(T, R, max_iters=30, tol=1e-6, seed=seed, use_snap=True, init_type=init_type)
            
            # Local search polish
            U, V, W, err = local_search_polish(U, V, W, R, max_steps=50, deadline=deadline - 5)
            
            if err < best_err:
                best_err = err
                best_UVW = (U, V, W)
            
            # Convert to lists for exact verification
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Search for minimal R
    best_result = schoolbook()
    
    # Try ranks from 27 to 35, allocating more time to lower ranks
    rank_time_budgets = {
        27: 15, 28: 14, 29: 13, 30: 12, 31: 11, 32: 10, 33: 9, 34: 8, 35: 7
    }
    
    for R in range(27, 36):
        if time.time() > deadline - 5:
            break
        time_budget = rank_time_budgets.get(R, 10)
        num_restarts = max(5, min(20, int(time_budget * 2)))
        res = try_rank(R, deadline - 5, num_restarts=num_restarts)
        if res is not None:
            return res

    return best_result
