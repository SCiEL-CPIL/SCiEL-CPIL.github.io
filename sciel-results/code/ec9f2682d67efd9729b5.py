"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for set A."""
    A_list = list(A)
    n = len(A_list)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_list[i] + A_list[j])
            prods.add(A_list[i] * A_list[j])
    return len(sums), len(prods)

def _score(A):
    s, p = _compute_sizes(A)
    return max(s, p)

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Candidate pool: Small integers are generally better for minimizing sum/product sets.
    # We use a range that allows for some structure but keeps computation fast.
    max_candidate = 300
    candidates = list(range(1, max_candidate + 1))
    
    # 1. Greedy Construction (Graph Clustering Approximation)
    # We build a set A by iteratively adding the element that causes the fewest new sums/products.
    # This is equivalent to finding a subgraph with high internal collision density.
    
    A = {1}
    sums = {2}
    prods = {1}
    
    # Pre-compute potential additions? No, just iterate.
    # To speed up, we can sort candidates by value initially, but greedy needs dynamic eval.
    # 300 candidates * 20 steps * (20 sums + 20 prods checks) = 300 * 20 * 40 = 240,000 ops. Very fast.
    
    for _ in range(19):
        best_c = None
        best_cost = float('inf')
        
        for c in candidates:
            if c in A:
                continue
            
            new_sums = 0
            new_prods = 0
            
            # Check collisions with current set
            # Optimization: if c is large, many sums/prods will be new.
            # We can break early if cost exceeds current best.
            
            for s in A:
                val_s = c + s
                if val_s not in sums:
                    new_sums += 1
                    if new_sums + new_prods >= best_cost:
                        break
            if new_sums + new_prods >= best_cost:
                continue
                
            for p in A:
                val_p = c * p
                if val_p not in prods:
                    new_prods += 1
                    if new_sums + new_prods >= best_cost:
                        break
            
            cost = new_sums + new_prods
            if cost < best_cost:
                best_cost = cost
                best_c = c
                
        if best_c is not None:
            A.add(best_c)
            for s in A:
                sums.add(best_c + s)
                prods.add(best_c * s)
        else:
            # Fallback if no candidate is better (shouldn't happen with this pool)
            break

    current_score = _score(A)
    best_A = set(A)
    best_score = current_score

    # 2. Local Search Refinement
    # Try to swap elements to further reduce the score.
    # We restrict swaps to the candidate pool to keep it bounded.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_list = list(A)
        random.shuffle(A_list)
        
        for old_val in A_list:
            if (time.time() - start_time) > time_limit:
                break
                
            # Try swapping old_val with a candidate
            # Sort candidates by likelihood of improvement (smaller is usually better)
            # But we should try a mix.
            for new_val in candidates:
                if new_val in A or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A.discard(old_val)
                new_A.add(new_val)
                
                new_score = _score(new_A)
                if new_score < current_score:
                    A = new_A
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = set(A)
                    improved = True
                    break  # Restart inner loop with new A
                elif new_score == current_score:
                    # Accept equal score to explore plateau
                    if random.random() < 0.1:
                        A = new_A
                        improved = True
                        break
                        
    return sorted(best_A)
# EVOLVE-BLOCK-END
