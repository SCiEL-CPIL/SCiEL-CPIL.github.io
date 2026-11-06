# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Get a greedy solution to establish an upper bound
    def greedy_solution(r):
        denoms = []
        while r > 0:
            k = (r.denominator + r.numerator - 1) // r.numerator
            if k <= 0:
                break
            denoms.append(k)
            r -= Fraction(1, k)
            if len(denoms) > 10:
                break
        return denoms

    greedy_denoms = greedy_solution(target)
    best_max = max(greedy_denoms) if greedy_denoms else float('inf')
    best_solution = greedy_denoms
    
    # Dijkstra search
    # State: (max_denom, remainder, last_k, path)
    heap = []
    heapq.heappush(heap, (0, target, 0, ()))
    
    visited = set()
    max_terms = 8
    
    while heap:
        if time.time() > deadline:
            break
            
        max_d, r, last_k, path = heapq.heappop(heap)
        
        if r == 0:
            if max_d < best_max:
                best_max = max_d
                best_solution = list(path)
            break
            
        if max_d >= best_max:
            continue
            
        if len(path) >= max_terms:
            continue
            
        state_key = (r, last_k)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # Calculate k_min
        # 1/k < r => k > 1/r
        # r = n/d. 1/r = d/n.
        # k > d/n => k >= floor(d/n) + 1
        d_val = r.denominator
        n_val = r.numerator
        k_min = max(last_k + 1, d_val // n_val + 1)
        
        k_max = best_max - 1
        if k_max < k_min:
            continue
            
        # Iterate k
        # To avoid too many iterations, we can break if we've spent too much time
        # or if k is very large.
        # But we must check all k < best_max to be sure.
        # However, if k is large, the new remainder is small change.
        # Let's just iterate.
        
        for k in range(k_min, k_max + 1):
            if time.time() > deadline:
                break
                
            r_new = r - Fraction(1, k)
            if r_new <= 0:
                if r_new == 0:
                    heapq.heappush(heap, (k, Fraction(0), k, path + (k,)))
                continue
            
            new_key = (r_new, k)
            if new_key not in visited:
                heapq.heappush(heap, (k, r_new, k, path + (k,)))
                
    if best_solution:
        return best_solution
    return greedy_denoms
# EVOLVE-BLOCK-END
