import random
import time
from itertools import combinations

def can_add(x, S):
    """
    Check if x can be added to set S such that the new set is valid.
    Validity: No element in new set divides the sum of any 3 distinct other elements.
    """
    S_list = list(S)
    n = len(S_list)
    
    # 1. Check if x divides sum of any 3 distinct elements in S
    # If x is very large, it might not divide any sum.
    # Max sum is 3 * 192 = 576.
    # If x > 576, this check is always False (no violation).
    # But x <= 192, so we always check.
    
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            for k in range(j + 1, n):
                d = S_list[k]
                if (b + c + d) % x == 0:
                    return False
                    
    # 2. For each a in S, check if a divides sum of any 3 distinct elements in (S \ {a}) U {x}
    # The triples must include x.
    # Sum is x + b + c, where b, c in S \ {a} and b != c.
    for a in S_list:
        for i in range(n):
            b = S_list[i]
            if b == a:
                continue
            for j in range(i + 1, n):
                c = S_list[j]
                if c == a:
                    continue
                # b and c are distinct and not a
                if (x + b + c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    best_set = set()
    best_size = 0
    
    # Initial heuristics
    orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        list(range(192, 0, -1, 2)),
        list(range(191, 0, -2)),
    ]
    
    # Add a few random orders
    for _ in range(5):
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() - start_time > time_limit:
            break
            
        S = set()
        # Greedy Phase
        for x in order:
            if len(S) >= 48:
                break
            if can_add(x, S):
                S.add(x)
                
        # Local Search Phase
        improved = True
        local_iterations = 0
        max_local_iterations = 500
        
        while improved and time.time() - start_time < time_limit and local_iterations < max_local_iterations:
            improved = False
            local_iterations += 1
            
            # 1. Try to add any remaining element
            # Iterate in a fixed order or random? Fixed is deterministic, random might find different adds.
            # Let's use a shuffled list of candidates for each iteration
            candidates = [x for x in range(1, 193) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if len(S) >= 48:
                    break
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                        
            # 2. If no add was possible, try swaps
            if not improved:
                swap_found = False
                # Try a limited number of random swaps
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    if len(S) == 0:
                        break
                    y = random.choice(list(S))
                    x_candidates = [x for x in range(1, 193) if x not in S]
                    if not x_candidates:
                        break
                    x = random.choice(x_candidates)
                    
                    # Check if swapping y for x is valid
                    # This is equivalent to can_add(x, S - {y})
                    if can_add(x, S - {y}):
                        S.remove(y)
                        S.add(x)
                        swap_found = True
                        improved = True # Restart the loop to try adds again
                        break # Break swap loop, go back to adds
                
                if not swap_found:
                    # No swap found, we are stuck in a local optimum
                    # We can break or try more sophisticated moves
                    # For now, break local search for this restart
                    pass

        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        if best_size == 48:
            break

    return list(best_set)
