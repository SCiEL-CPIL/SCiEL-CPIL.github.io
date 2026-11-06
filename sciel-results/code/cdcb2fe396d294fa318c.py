import time
import random
import math
from itertools import combinations

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def generate_smooth_candidates(max_val=10000, primes=(2, 3, 5, 7, 11)):
    """Generate a list of smooth numbers up to max_val."""
    smooths = set()
    def gen(idx, current_val):
        if idx == len(primes):
            if current_val <= max_val and current_val > 0:
                smooths.add(current_val)
            return
        p = primes[idx]
        val = current_val
        while val <= max_val:
            gen(idx + 1, val)
            val *= p
    gen(0, 1)
    return sorted(smooths)

def incremental_score(A, new_elem):
    """
    Compute the change in max(|A+A|, |A*A|) when adding new_elem to A.
    Returns (new_sumset_size, new_prodset_size, new_max).
    """
    n = len(A)
    # Count new sums: new_elem + a for a in A, and 2*new_elem
    new_sums = set()
    for a in A:
        new_sums.add(new_elem + a)
    new_sums.add(2 * new_elem)
    
    # Count new products: new_elem * a for a in A, and new_elem^2
    new_prods = set()
    for a in A:
        new_prods.add(new_elem * a)
    new_prods.add(new_elem * new_elem)
    
    # Current sizes
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    
    cur_sum_size = len(sum_set)
    cur_prod_size = len(prod_set)
    
    # New sizes
    new_sum_size = cur_sum_size + len(new_sums - sum_set)
    new_prod_size = cur_prod_size + len(new_prods - prod_set)
    
    return new_sum_size, new_prod_size, max(new_sum_size, new_prod_size)

def greedy_construct(pool, target_size=20, max_val=10000):
    """
    Greedy construction: build set one element at a time,
    choosing the element that minimizes the resulting max(|A+A|, |A*A|).
    """
    A = []
    A_set = set()
    
    for _ in range(target_size):
        best_elem = None
        best_score = float('inf')
        best_sum_size = 0
        best_prod_size = 0
        
        for candidate in pool:
            if candidate in A_set or candidate > max_val:
                continue
            
            # Estimate the score if we add this candidate
            n = len(A)
            if n == 0:
                new_sum_size = 1  # {2*candidate}
                new_prod_size = 1  # {candidate^2}
                new_max = 1
            else:
                new_sum_size, new_prod_size, new_max = incremental_score(A, candidate)
            
            if new_max < best_score or (new_max == best_score and new_sum_size + new_prod_size < best_sum_size + best_prod_size):
                best_score = new_max
                best_sum_size = new_sum_size
                best_prod_size = new_prod_size
                best_elem = candidate
        
        if best_elem is None:
            break
        
        A.append(best_elem)
        A_set.add(best_elem)
    
    return sorted(A)

def local_search_swap(A, pool, max_val=10000, time_limit=5.0):
    """
    Local search: try swapping elements to improve the score.
    """
    start_time = time.time()
    A = A[:]
    A_set = set(A)
    current_score = compute_metrics(A)
    best_A = A[:]
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        improved = False
        n = len(A)
        
        # Try each position
        for i in range(n):
            old_val = A[i]
            
            # Try replacing with elements from pool
            for candidate in pool:
                if candidate == old_val or candidate in A_set:
                    continue
                
                # Temporarily swap
                A[i] = candidate
                A_set.discard(old_val)
                A_set.add(candidate)
                
                new_score = compute_metrics(A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A[:]
                else:
                    # Revert
                    A[i] = old_val
                    A_set.discard(candidate)
                    A_set.add(old_val)
        
        if not improved:
            break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 40.0  # Safety margin
    
    # Generate candidate pools
    smooth_pool = generate_smooth_candidates(10000, [2, 3, 5, 7])
    
    # Also include some arithmetic progression elements and mixed
    ap_pool = list(range(1, 51))
    
    # Combined pool with emphasis on smooth numbers
    combined_pool = sorted(set(smooth_pool + ap_pool))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple greedy constructions with different starting strategies
    strategies = [
        ("smooth_first", smooth_pool),
        ("ap_first", ap_pool),
        ("combined", combined_pool),
    ]
    
    for name, pool in strategies:
        if time.time() > deadline:
            break
        
        # Run greedy construction
        A = greedy_construct(pool, 20, 10000)
        
        if len(A) < 20:
            continue
        
        score = compute_metrics(A)
        
        if score < best_score:
            best_score = score
            best_A = A[:]
        
        # Local search refinement
        A_refined, score_refined = local_search_swap(A, pool, 10000, time_limit=3.0)
        
        if score_refined < best_score:
            best_score = score_refined
            best_A = A_refined[:]
    
    # Final local search on the best result
    if best_A is not None and time.time() < deadline - 2.0:
        remaining_time = deadline - time.time() - 1.0
        A_final, score_final = local_search_swap(best_A, combined_pool, 10000, time_limit=min(remaining_time, 10.0))
        
        if score_final < best_score:
            best_score = score_final
            best_A = A_final[:]
    
    # Fallback if nothing worked
    if best_A is None or len(best_A) != 20:
        best_A = list(range(1, 21))
    
    return best_A
