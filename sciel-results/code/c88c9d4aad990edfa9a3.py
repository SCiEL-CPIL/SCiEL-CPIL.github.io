"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 57  # Safety margin for 90s limit
    
    N = 256
    all_values = list(range(1, N + 1))
    
    # 1. Precompute conflict information using prime factorization for speed
    # For each value, store its prime factors
    # Two values conflict if they share any prime factor
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        prime_factors[v] = get_prime_factors(v)
    
    # conflicts[v] = set of u in 1..256, u != v, such that gcd(u, v) > 1
    conflicts = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        for u in range(1, N + 1):
            if u != v and prime_factors[v] & prime_factors[u]:
                conflicts[v].add(u)
    
    # 2. Greedy Initialization
    def greedy_init(order):
        selected = set()
        for v in order:
            if not (selected & conflicts[v]):
                selected.add(v)
        return selected
    
    best_set = set()
    best_sum = 0
    
    # Try multiple initializations
    # 1. Descending order
    init_order = list(range(N, 0, -1))
    s = greedy_init(init_order)
    current_sum = sum(s)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(s)
    
    # 2. A few random orders
    for _ in range(5):
        rand_order = all_values[:]
        random.shuffle(rand_order)
        s = greedy_init(rand_order)
        cs = sum(s)
        if cs > best_sum:
            best_sum = cs
            best_set = set(s)
    
    current_set = set(best_set)
    current_sum = best_sum
    
    # 3. Local Search with Best-Improvement for adds, systematic swaps
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Best-improvement ADD: find the highest-value unselected v that can be added
        unselected = [v for v in all_values if v not in current_set]
        best_add = None
        for v in sorted(unselected, reverse=True):
            if not (current_set & conflicts[v]):
                best_add = v
                break
        
        if best_add is not None:
            current_set.add(best_add)
            current_sum += best_add
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            improved = True
            stuck_count = 0
        
        # 2. If no add, try systematic SWAPS with best-improvement
        if not improved:
            selected_list = list(current_set)
            unselected_list = [v for v in all_values if v not in current_set]
            
            if selected_list and unselected_list:
                # Sort unselected descending to find high-value swaps first
                unsorted_unselected = sorted(unselected_list, reverse=True)
                found_swap = False
                
                # For each selected u (sorted ascending to try removing small ones first),
                # check if we can swap with a high-value unselected v
                for u in sorted(selected_list):
                    if found_swap:
                        break
                    for v in unsorted_unselected:
                        if v <= u:
                            break  # No gain possible
                        # Check if swapping u with v is valid
                        common = current_set & conflicts[v]
                        if not common or common == {u}:
                            # Valid swap
                            current_set.remove(u)
                            current_set.add(v)
                            current_sum += (v - u)
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = set(current_set)
                            found_swap = True
                            break
                
                if found_swap:
                    improved = True
                    stuck_count = 0
        
        # 3. If still stuck, try 2-1 moves: Remove 1, Add 2
        if not improved:
            selected_list = list(current_set)
            if selected_list:
                # Try each selected u (sorted by value ascending - remove small ones first)
                found_21 = False
                for u in sorted(selected_list):
                    if found_21:
                        break
                    S_minus_u = current_set - {u}
                    
                    # Find all addable v's
                    addable_vs = []
                    for v in all_values:
                        if v in current_set:
                            continue
                        if not (S_minus_u & conflicts[v]):
                            addable_vs.append(v)
                    
                    if len(addable_vs) >= 2:
                        # Sort by value descending
                        addable_vs.sort(reverse=True)
                        # Find best coprime pair
                        best_pair = None
                        best_gain = 0
                        for i in range(len(addable_vs)):
                            v1 = addable_vs[i]
                            if v1 + addable_vs[0] - u <= best_gain:
                                break  # Can't beat best_gain with any v2
                            for j in range(i + 1, len(addable_vs)):
                                v2 = addable_vs[j]
                                if v1 + v2 - u <= best_gain:
                                    break
                                if gcd(v1, v2) == 1:
                                    gain = v1 + v2 - u
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_pair = (v1, v2)
                        
                        if best_pair and best_gain > 0:
                            v1, v2 = best_pair
                            current_set.remove(u)
                            current_set.add(v1)
                            current_set.add(v2)
                            current_sum += best_gain
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = set(current_set)
                            found_21 = True
                            improved = True
                            stuck_count = 0
        
        # 4. If truly stuck, do a kick
        if not improved:
            stuck_count += 1
            
            if stuck_count % 50 == 0:
                # Random restart with perturbation
                # Remove 1-3 random elements from current_set, then greedy fill
                kick_set = set(current_set)
                kick_size = min(len(kick_set), random.randint(1, 3))
                for _ in range(kick_size):
                    if kick_set:
                        kick_set.remove(random.choice(list(kick_set)))
                
                # Greedy fill on kick_set
                for v in sorted(all_values, reverse=True):
                    if v not in kick_set and not (kick_set & conflicts[v]):
                        kick_set.add(v)
                
                ks = sum(kick_set)
                if ks > best_sum:
                    best_sum = ks
                    best_set = set(kick_set)
                    current_set = set(kick_set)
                    current_sum = ks
                    stuck_count = 0
                else:
                    # Keep current_set, just reset stuck count to avoid too many restarts
                    stuck_count = 0

    # Final result
    result = list(best_set)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
