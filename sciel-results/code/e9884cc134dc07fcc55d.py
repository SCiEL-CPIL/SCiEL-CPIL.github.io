"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Precompute the target tensor support
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k, else 0
    # We can represent this as a set of (u,v,w) tuples where the value is 1
    target_support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_support.add((u, v, w))
                
    # Also, we need to ensure that for any (u,v,w) NOT in support, the sum is 0.
    # The full tensor has dimensions nm x mp x np_.
    # Total elements: 6 * 21 * 14 = 1764.
    # Support size: 42.
    
    def check_decomposition(U, V, W, rank):
        """
        Check if U, V, W of shape (nm, rank), (mp, rank), (np_, rank)
        exactly decompose the matrix multiplication tensor.
        U, V, W are numpy arrays.
        """
        # Vectorized check might be faster but let's try a simple loop first
        # Or better, construct the tensor and check.
        # T = sum_c U[:, c] * V[:, c] * W[:, c] (outer product)
        # We can compute the full tensor and compare.
        # T_flat = np.zeros((nm, mp, np_))
        # for c in range(rank):
        #     T_flat += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Since rank is small (<=42), and dimensions are small, this is fast.
        T = np.zeros((nm, mp, np_), dtype=float)
        for c in range(rank):
            T += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
            
        # Check support
        for u, v, w in target_support:
            if abs(T[u, v, w] - 1.0) > 1e-9:
                return False
                
        # Check zeros
        # Create a mask of the support
        mask = np.zeros((nm, mp, np_), dtype=bool)
        for u, v, w in target_support:
            mask[u, v, w] = True
            
        # Check all non-support elements are 0
        if np.any(np.abs(T[~mask]) > 1e-9):
            return False
            
        return True

    def error_decomposition(U, V, W, rank):
        """
        Compute the squared error of the decomposition.
        """
        T = np.zeros((nm, mp, np_), dtype=float)
        for c in range(rank):
            T += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
            
        # Target tensor
        Target = np.zeros((nm, mp, np_), dtype=float)
        for u, v, w in target_support:
            Target[u, v, w] = 1.0
            
        return np.sum((T - Target) ** 2)

    def solve_rank(rank, values, max_restarts=100):
        """
        Try to find a decomposition of given rank using local search.
        values: list of allowed values for entries.
        """
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None, float('inf')
                
            # Initialize randomly
            U = np.random.choice(values, size=(nm, rank))
            V = np.random.choice(values, size=(mp, rank))
            W = np.random.choice(values, size=(np_, rank))
            
            current_err = error_decomposition(U, V, W, rank)
            if current_err < 1e-9:
                return (U, V, W), 0.0
                
            stuck_counter = 0
            max_stuck = 50
            
            while time.time() < deadline:
                if stuck_counter > max_stuck:
                    break
                    
                # Pick a random matrix and entry to modify
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    rows = nm
                elif mat_idx == 1:
                    mat = V
                    rows = mp
                else:
                    mat = W
                    rows = np_
                    
                row = np.random.choice(rows)
                col = np.random.choice(rank)
                
                old_val = mat[row, col]
                new_val = np.random.choice(values)
                
                if new_val == old_val:
                    continue
                    
                mat[row, col] = new_val
                new_err = error_decomposition(U, V, W, rank)
                
                if new_err < current_err:
                    current_err = new_err
                    stuck_counter = 0
                    if current_err < 1e-9:
                        return (U, V, W), 0.0
                    if current_err < best_err:
                        best_err = current_err
                else:
                    mat[row, col] = old_val
                    stuck_counter += 1
                    
        return None, best_err

    # Try ranks from 21 to 41
    # We know rank(2,3,7) <= 41.
    # Let's try to find a valid decomposition for rank < 42.
    
    # Values to try: integers and half-integers
    values_int = [0, 1, -1]
    values_half = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    best_solution = None
    best_rank = n * m * p
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((np_, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Search for lower rank
    # Start from 21 (lower bound for 2x3x7 is actually 21? No, lower bound is 2*3*7 / max(2,3,7) ? No.
    # Lower bound for rank is nm+mp+np - (n+m+p-2) ? No.
    # For 2x3x7, rank is at least 2*3 + 3*7 + 7*2 - (2+3+7-2) = 6+21+14-10 = 31?
    # Actually, the lower bound is max(nm, mp, np) = 21.
    # Known rank for 2x3x7 is 41? Or is it lower?
    # Let's just try from 21 up to 41.
    
    for r in range(21, 42):
        if time.time() > deadline:
            break
            
        # Try with integers first
        sol, err = solve_rank(r, values_int, max_restarts=50)
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = r
            break # Found a valid one, can we go lower? No, we are iterating up.
            # Wait, if we find a valid one at rank r, we should check if we can find one at r-1?
            # But we are iterating from 21 up. So if we find one at 35, we don't know if 34 works.
            # However, finding ANY valid decomposition with rank < 42 is an improvement.
            # To be safe and fast, if we find one, we can try to see if we can go lower?
            # Or just return the first one found?
            # The goal is rank-minimal. So we should keep searching for lower ranks?
            # But we are iterating 21, 22, ...
            # So if we find one at 21, that's optimal (likely).
            # If we find one at 30, that's good.
            # So we should break only if we are confident or if time is running out.
            # Actually, if we find a solution at rank r, we should still check r-1?
            # No, the loop goes up. So if we find one at 30, we have a candidate.
            # But maybe 29 works? We didn't find it in the previous iterations.
            # So we should continue? No, that would waste time.
            # Let's just return the first valid one found in the increasing order.
            # This ensures minimal rank among those found.
            pass 
            
        # If not found with integers, try half-integers
        if time.time() < deadline - 5:
            sol, err = solve_rank(r, values_half, max_restarts=50)
            if sol is not None:
                best_U, best_V, best_W = sol
                best_rank = r
                break

    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
