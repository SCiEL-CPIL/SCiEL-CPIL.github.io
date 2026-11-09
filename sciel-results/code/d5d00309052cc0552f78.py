import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    dim_U = n * m  # 6
    dim_V = m * p  # 15
    dim_W = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the tensor: T[a,b,c] = 1 for valid (i,j,k), 0 otherwise
    # Valid entries: a=i*m+j, b=j*p+k, c=k*n+i
    # We need sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c] for ALL a,b,c
    # This gives dim_U * dim_V * dim_W = 6*15*10 = 900 equations
    # But most are 0, only 30 are 1
    
    # For efficiency, we only need to satisfy:
    # 1. The 30 "1" equations
    # 2. The 870 "0" equations (or a subset that's sufficient)
    
    # Actually, for exact decomposition, we need ALL entries to match.
    # Let's build the full constraint set but use sparsity.
    
    # The key insight: for the multiplication tensor, the "1" positions
    # determine the structure. The "0" positions are automatically satisfied
    # if we use a decomposition that respects the tensor structure.
    
    # Let's try a structured approach:
    # For rank r, we need U (6×r), V (15×r), W (10×r) such that
    # for all a,b,c: sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    
    # Strategy: Try ranks from 18 upward (known lower bound for some cases)
    # Use ALS with random restarts
    
    def check_solution(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][s] * V[b][s] * W[c][s] for s in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check some zero entries
        for a in range(dim_U):
            for b in range(dim_V):
                for c in range(dim_W):
                    # Check if this is a valid entry
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue  # This is a "1" entry, already checked
                    val = sum(U[a][s] * V[b][s] * W[c][s] for s in range(r))
                    if abs(val) > 1e-9:
                        return False
        return True
    
    def try_rank_als(r, num_restarts=50, max_iter=200):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(dim_U, r) * 0.3
            V = np.random.randn(dim_V, r) * 0.3
            W = np.random.randn(dim_W, r) * 0.3
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (a,b,c), fix V and W
                # sum_s U[a,s] * (V[b,s] * W[c,s]) = T[a,b,c]
                # Build linear system
                A = np.zeros((dim_U * dim_V * dim_W, dim_U * r))
                rhs = np.zeros(dim_U * dim_V * dim_W)
                
                idx = 0
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            # Determine T[a,b,c]
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            
                            for s in range(r):
                                A[idx, a * r + s] = V[b, s] * W[c, s]
                            idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    U = sol.reshape(dim_U, r)
                except:
                    continue
                
                # Update V
                A = np.zeros((dim_U * dim_V * dim_W, dim_V * r))
                idx = 0
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            
                            for s in range(r):
                                A[idx, b * r + s] = U[a, s] * W[c, s]
                            idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    V = sol.reshape(dim_V, r)
                except:
                    continue
                
                # Update W
                A = np.zeros((dim_U * dim_V * dim_W, dim_W * r))
                idx = 0
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs[idx] = 1.0
                            else:
                                rhs[idx] = 0.0
                            
                            for s in range(r):
                                A[idx, c * r + s] = U[a, s] * V[b, s]
                            idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    W = sol.reshape(dim_W, r)
                except:
                    continue
                
                # Check convergence
                err = 0.0
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            target = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                            val = sum(U[a, s] * V[b, s] * W[c, s] for s in range(r))
                            err += (val - target) ** 2
                
                if err < 1e-12:
                    # Found a solution! Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    # Verify with exact arithmetic
                    if check_solution(U_list, V_list, W_list, r):
                        return (U_list, V_list, W_list)
                    else:
                        # Try rounding to half-integers
                        U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                        V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                        W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                        if check_solution(U_rounded, V_rounded, W_rounded, r):
                            return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try ranks from low to high
    # Known: schoolbook is 30. Try from 18 upward.
    for r in range(18, 31):
        if time.time() > deadline - 15:
            break
        
        result = try_rank_als(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p  # 30
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
