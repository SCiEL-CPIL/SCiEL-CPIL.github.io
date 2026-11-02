import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s limit
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition gives T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # and U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c equals 1 for all (i,j,k)
    
    # Try ranks from low to high
    for target_rank in range(12, 25):
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline)
        if result is not None:
            return result
    
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


def _try_rank(target_rank, n, m, p, deadline):
    """
    Try to find a rank-target_rank decomposition using structured search with
    a set-cover-inspired greedy approach and local refinement.
    """
    rows_U = n * m  # 6
    rows_V = m * p  # 12
    rows_W = p * n  # 8
    
    # Grid of allowed values (integers and half-integers)
    grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # All (i,j,k) indices
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)  # 24
    
    def compute_error(U, V, W):
        """Compute sum of squared errors over all (i,j,k) triples."""
        err = 0.0
        for i, j, k in triples:
            s = 0.0
            for c in range(target_rank):
                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            diff = s - 1.0
            err += diff * diff
        return err
    
    def compute_error_vector(U, V, W):
        """Compute error for each triple individually."""
        errs = np.zeros(num_triples)
        for idx, (i, j, k) in enumerate(triples):
            s = 0.0
            for c in range(target_rank):
                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            errs[idx] = s - 1.0
        return errs
    
    def try_local_search(U, V, W, max_iters=3000):
        """Local search to improve the decomposition with coordinate descent."""
        current_err = compute_error(U, V, W)
        
        # Precompute row indices for each matrix
        U_rows = list(range(rows_U))
        V_rows = list(range(rows_V))
        W_rows = list(range(rows_W))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random variable
            total_vars = rows_U * target_rank + rows_V * target_rank + rows_W * target_rank
            var_idx = np.random.randint(0, total_vars)
            
            if var_idx < rows_U * target_rank:
                idx = var_idx
                mat = U
                r, c = divmod(idx, target_rank)
            elif var_idx < rows_U * target_rank + rows_V * target_rank:
                idx = var_idx - rows_U * target_rank
                mat = V
                r, c = divmod(idx, target_rank)
            else:
                idx = var_idx - rows_U * target_rank - rows_V * target_rank
                mat = W
                r, c = divmod(idx, target_rank)
                
            old_val = mat[r, c]
            
            # Try changing to a random other grid value
            new_val = np.random.choice(grid)
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-10:
                    return U, V, W, current_err
            else:
                mat[r, c] = old_val
                
        return U, V, W, current_err
    
    def try_greedy_set_cover(n_restarts=20):
        """
        Greedy set-cover approach: precompute candidate rank-1 terms,
        then iteratively select the term that best covers unsatisfied coefficients.
        """
        best_result = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with zeros
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            # Track which triples are "covered" (close to 1)
            covered = np.zeros(num_triples, dtype=bool)
            
            for c in range(target_rank):
                if time.time() > deadline:
                    break
                
                # Find the best single rank-1 term to add at position c
                best_score = -1
                best_u, best_v, best_w = None, None, None
                
                # Sample candidate values from grid
                # For efficiency, try a subset of combinations
                # Focus on values in {-1, 0, 1} for speed
                simple_grid = np.array([-1.0, 0.0, 1.0])
                
                # Try random candidates
                n_candidates = 500
                for _ in range(n_candidates):
                    if time.time() > deadline:
                        break
                    
                    u_cand = np.random.choice(simple_grid, size=rows_U)
                    v_cand = np.random.choice(simple_grid, size=rows_V)
                    w_cand = np.random.choice(simple_grid, size=rows_W)
                    
                    # Skip if all zeros
                    if np.all(u_cand == 0) or np.all(v_cand == 0) or np.all(w_cand == 0):
                        continue
                    
                    # Compute contribution of this term to each triple
                    # and score based on how well it helps uncovered triples
                    score = 0.0
                    for idx, (i, j, k) in enumerate(triples):
                        contrib = u_cand[i*m+j] * v_cand[j*p+k] * w_cand[k*n+i]
                        if not covered[idx]:
                            # Reward terms that move uncovered triples toward 1
                            # Current value at this triple is 0 (since we're building up)
                            # We want contrib to be positive and close to 1
                            if contrib > 0:
                                score += contrib
                            else:
                                score -= abs(contrib)
                        else:
                            # For covered triples, we don't want to disturb them
                            # but since we're adding a new term, it will add to existing
                            # This is a simplification
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u_cand.copy(), v_cand.copy(), w_cand.copy()
                
                if best_u is not None:
                    U[:, c] = best_u
                    V[:, c] = best_v
                    W[:, c] = best_w
                    
                    # Update coverage
                    for idx, (i, j, k) in enumerate(triples):
                        # Compute current sum at this triple
                        s = 0.0
                        for cc in range(c + 1):
                            s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                        if abs(s - 1.0) < 0.1:
                            covered[idx] = True
                
                # Check if all covered
                if np.all(covered):
                    break
            
            # Refine with local search
            U, V, W, err = try_local_search(U, V, W, max_iters=2000)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-10:
                break
        
        return best_result, best_err
    
    best_result = None
    best_err = float('inf')
    
    # Strategy 1: Random restarts with local search
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Structured initialization: identity-like pattern
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            for c in range(target_rank):
                ci = c % n
                cj = c % m
                ck = c % p
                
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = 1.0 if (i == ci and j == cj) else 0.0
                        
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, c] = 1.0 if (j == cj and k == ck) else 0.0
                        
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, c] = 1.0 if (k == ck and i == ci) else 0.0
                        
        elif restart == 1:
            # Binary pattern
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            for c in range(target_rank):
                for r in range(rows_U):
                    U[r, c] = 1.0 if (r + c) % 2 == 0 else 0.0
                for r in range(rows_V):
                    V[r, c] = 1.0 if (r + c) % 3 == 0 else 0.0
                for r in range(rows_W):
                    W[r, c] = 1.0 if (r + c) % 4 == 0 else 0.0
                    
        elif restart == 2:
            # Sparse random
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            for c in range(target_rank):
                # Randomly set a few entries
                for _ in range(2):
                    r = np.random.randint(0, rows_U)
                    U[r, c] = np.random.choice([-1.0, 1.0])
                for _ in range(3):
                    r = np.random.randint(0, rows_V)
                    V[r, c] = np.random.choice([-1.0, 1.0])
                for _ in range(2):
                    r = np.random.randint(0, rows_W)
                    W[r, c] = np.random.choice([-1.0, 1.0])
                    
        else:
            # Random initialization from grid
            U = np.random.choice(grid, size=(rows_U, target_rank))
            V = np.random.choice(grid, size=(rows_V, target_rank))
            W = np.random.choice(grid, size=(rows_W, target_rank))
        
        U, V, W, err = try_local_search(U, V, W, max_iters=1500)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
            
        if best_err < 1e-10:
            break
    
    # Strategy 2: Greedy set cover approach
    if best_err > 1e-10 and time.time() < deadline:
        greedy_result, greedy_err = try_greedy_set_cover(n_restarts=10)
        if greedy_result is not None and greedy_err < best_err:
            best_err = greedy_err
            best_result = greedy_result
    
    if best_result is not None and best_err < 1e-10:
        U, V, W = best_result
        # Round to nearest grid value
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification
        if compute_error(U, V, W) < 1e-10:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None
