"""Tabu Search approach for QAP tai256c with fast swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a_sym, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    We want to swap p[i] and p[j].
    
    Let x = p[i], y = p[j].
    The total cost is sum_{m,n} a[m,n] * b[p[m], p[n]].
    Since a and b are symmetric (or we use a_sym = a + a.T to handle general case efficiently in the sum derivation),
    the delta for swapping positions i and j (facilities x and y) is:
    delta = sum_{k != i, j} (a_sym[i, k] - a_sym[j, k]) * (b[y, p[k]] - b[x, p[k]])
    
    We precompute a_sym = a + a.T.
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    delta = sum_{k} diff_a[k] * diff_b[p[k]] - diff_a[i]*diff_b[x] - diff_a[j]*diff_b[y]
    """
    x = p[i]
    y = p[j]
    
    # Vectorized computation
    # a_sym[i, :] is the row i of a_sym
    # a_sym[j, :] is the row j of a_sym
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    # diff_b[p] gives the vector where element k is diff_b[p[k]]
    temp = diff_a * diff_b[p]
    
    # Subtract the terms for k=i and k=j
    # When k=i, p[k]=x, so term is diff_a[i] * diff_b[x]
    # When k=j, p[k]=y, so term is diff_a[j] * diff_b[y]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta


def _solve_tai256c(a, b, time_limit=145.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute symmetric flow matrix
    a_sym = a + a.T
    
    # Initial solution: Rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Tabu Search Parameters
    tabu_tenure = 20  # Length of tabu list
    tabu_list = {}    # Dictionary mapping (i, j) to time when it will expire
    current_cost = best_cost
    
    # Main Tabu Search Loop
    iter_count = 0
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Check time more frequently
        if time.time() - start_time > time_limit:
            break
            
        # Select best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        is_tabu = True
        
        # To speed up, we can sample a subset of pairs if n is large, but n=256 is small enough
        # to check all pairs if optimized, but O(n^2) * O(n) = O(n^3) per iteration might be slow.
        # n=256, n^2/2 ~ 32k pairs. O(n) per pair ~ 256 ops. 32k * 256 ~ 8M ops per iteration.
        # 145s / 8M ops ~ 18k iterations. This is feasible in Python/NumPy if vectorized well.
        # However, pure Python loop over 32k pairs is slow.
        # Let's try to vectorize the move selection partially or just use a random subset if too slow.
        # Actually, for n=256, a full scan might take ~0.05-0.1s per iteration in Python if not careful.
        # 145s / 0.1s = 1450 iterations.
        
        # We will do a full scan but with early exit if we find a very good move? No, we need the best.
        # Let's just do the full scan.
        
        current_time = time.time()
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                
                # Check tabu status
                key = (i, j) if i < j else (j, i)
                if key in tabu_list and tabu_list[key] > iter_count:
                    # It's tabu, check aspiration
                    # Aspiration: if the move leads to a solution better than the best found so far
                    # We don't know the new cost without computing delta, so we compute delta first?
                    # Or we can just skip tabu moves unless they are very promising.
                    # Standard Tabu: skip unless aspiration holds.
                    # Since we don't know new cost, we can't check aspiration exactly.
                    # Approximation: skip tabu moves.
                    continue
                
                delta = _swap_delta(a_sym, b, p, i, j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    is_tabu = False
        
        if best_i == -1:
            # No non-tabu move found? Or all deltas positive?
            # If all deltas are positive, we are at a local minimum in the non-tabu neighborhood.
            # We should still make a move to escape, typically the best available (even if tabu or positive).
            # Let's re-scan allowing tabu moves but only if no non-tabu move was found?
            # Or just pick the best delta overall, regardless of tabu, if no non-tabu found.
            best_delta = np.inf
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    delta = _swap_delta(a_sym, b, p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i == -1:
                break # No moves possible? Should not happen for n>1
        
        # Execute the move
        i, j = best_i, best_j
        x = p[i]
        y = p[j]
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update tabu list
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iter_count + tabu_tenure
        # Also mark the reverse move (j, i) which is the same key
        # And potentially other related moves? Standard is just the swap (i,j).
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Clean up old tabu entries to save memory (optional for n=256)
        if iter_count % 1000 == 0:
            for k in list(tabu_list.keys()):
                if tabu_list[k] <= iter_count:
                    del tabu_list[k]
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
