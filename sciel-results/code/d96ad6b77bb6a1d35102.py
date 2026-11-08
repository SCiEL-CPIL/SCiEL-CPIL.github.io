"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = 4, 4, 5
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Build the matrix multiplication tensor indices
    # M[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
    # We only need to store the non-zero indices
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    # Schoolbook decomposition (Rank 80)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = 80

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        rank = U.shape[1]
        # Check all nonzero entries
        for a, b, c in nonzero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries (sample)
        # To be rigorous, we should check all, but 16*20*20 = 6400 is small enough
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Check if it's a nonzero triple
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue # Skip nonzero
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_int(x):
        return np.round(2 * x) / 2.0

    # Try to find rank 79
    # Strategy: Use the schoolbook solution as a starting point, 
    # set one column to zero (effectively reducing rank if it becomes linearly dependent or we just use 79 columns),
    # and optimize the remaining entries to fit the tensor.
    
    # A more robust approach: Random restarts with ALS optimized for speed.
    
    def fast_als(U, V, W, n_iters=200, tol=1e-10):
        """Fast ALS using sparsity."""
        rank = U.shape[1]
        
        for iteration in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for a in range(nm):
                # Find all (b, c) such that M[a, b, c] != 0
                # And we also need to satisfy M[a, b, c] = 0 for others.
                # This is a least squares problem: minimize || A U_a - y ||^2
                # where A rows correspond to all (b, c) pairs.
                # Since the tensor is dense in the "zero" sense (most are 0), 
                # we must include them in the least squares to ensure they stay 0.
                
                # However, for speed, we can use the fact that we only care about 
                # the 80 non-zeros being 1 and the rest being 0.
                # Let's build the system for all 20*20 = 400 pairs for a fixed a.
                
                A_mat = np.zeros((mp * pn, rank))
                rhs = np.zeros(mp * pn)
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        A_mat[idx, :] = V[b, :] * W[c, :]
                        # Determine if M[a, b, c] is 1 or 0
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
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    U[a, :] = sol
                except:
                    pass
            
            # Update V
            for b in range(mp):
                A_mat = np.zeros((nm * pn, rank))
                rhs = np.zeros(nm * pn)
                idx = 0
                for a in range(nm):
                    for c in range(pn):
                        A_mat[idx, :] = U[a, :] * W[c, :]
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
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    V[b, :] = sol
                except:
                    pass
            
            # Update W
            for c in range(pn):
                A_mat = np.zeros((nm * mp, rank))
                rhs = np.zeros(nm * mp)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        A_mat[idx, :] = U[a, :] * V[b, :]
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
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    W[c, :] = sol
                except:
                    pass
            
            # Check error
            err = 0.0
            for a, b, c in nonzero_indices:
                pred = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (pred - 1.0) ** 2
            # Sample zero errors
            np.random.seed(42 + iteration)
            for _ in range(50):
                a = np.random.randint(nm)
                b = np.random.randint(mp)
                c = np.random.randint(pn)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (i_a == i_c and j_a == j_b and k_b == k_c):
                    pred = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += pred ** 2
            
            if err < tol:
                break
        
        return U, V, W, err

    # Search for rank 79
    # We try to initialize with schoolbook and drop one term, then refine.
    # We also try random half-integer initializations.
    
    for target_rank in [79, 78, 77]:
        if time.time() > deadline - 20:
            break
        
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                # Initialize with schoolbook, remove last column
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                # Perturb slightly
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            elif restart % 2 == 0:
                # Random half-integer initialization
                U = np.round(np.random.randn(nm, target_rank) * 2) / 2
                V = np.round(np.random.randn(mp, target_rank) * 2) / 2
                W = np.round(np.random.randn(pn, target_rank) * 2) / 2
            else:
                # Sparse random initialization
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    u_idx = np.random.choice(nm, 3, replace=False)
                    v_idx = np.random.choice(mp, 3, replace=False)
                    w_idx = np.random.choice(pn, 3, replace=False)
                    U[u_idx, r] = np.random.choice([-1, 0, 1], 3)
                    V[v_idx, r] = np.random.choice([-1, 0, 1], 3)
                    W[w_idx, r] = np.random.choice([-1, 0, 1], 3)
            
            U, V, W, err = fast_als(U, V, W, n_iters=100)
            
            # Round to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                    return best_result
            
            # Also try without rounding if error was very small
            if err < 1e-12:
                 if target_rank < best_rank:
                     best_rank = target_rank
                     best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                     return best_result

    return best_result
