import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    
    # Precompute divisors for products
    # divs[p] is a set of a in [2, 256] such that a divides p
    max_prod = 256 * 255
    divs = [set() for _ in range(max_prod + 1)]
    for a in range(2, N + 1):
        for multiple in range(a, max_prod + 1, a):
            divs[multiple].add(a)
            
    def is_valid_set(S):
        """
        Check if set S is valid.
        Valid means: for all distinct a, b, c in S, a does not divide b*c.
        """
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j: continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = S_list[k]
                    if a in divs[b * c]:
                        return False
        return True

    def can_add(S, x):
        """
        Check if x can be added to S without creating a conflict.
        Assumes x not in S.
        """
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(n):
                if i == j: continue
                c = S_list[j]
                if x in divs[b * c]:
                    return False
        return True

    # 1. Greedy Initialization
    # Try multiple greedy orders to find a good starting point
    best_set = set()
    best_size = 0
    
    orders = [
        list(range(N, 1, -1)), # Descending
        list(range(2, N + 1)), # Ascending
        list(range(2, N + 1, 2)), # Evens first? No, mixed.
        [x for x in range(N, 1, -1) if x % 2 != 0] + [x for x in range(N, 1, -1) if x % 2 == 0]
    ]
    
    for order in orders:
        S = set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # 2. Local Search
    # We will use a hill-climbing approach with 1-swap moves.
    # If stuck, we might try 2-swap, but 1-swap is faster.
    # We also try to add elements greedily after each successful swap.
    
    current_set = best_set.copy()
    current_size = len(current_set)
    all_numbers = set(numbers)
    
    # Precompute list of numbers for faster iteration
    nums_list = list(all_numbers)
    random.shuffle(nums_list)
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < DEADLINE:
        if no_improve_count > max_no_improve:
            # Try a random restart or kick
            # Remove 2 random elements, try to add 2 random elements
            if len(current_set) >= 2:
                items_to_remove = random.sample(list(current_set), 2)
                temp_set = current_set - set(items_to_remove)
                
                # Try to add 2 random elements not in temp_set
                candidates = [x for x in nums_list if x not in temp_set]
                random.shuffle(candidates)
                
                added = 0
                for y in candidates:
                    if added == 2: break
                    if can_add(temp_set, y):
                        temp_set.add(y)
                        added += 1
                        
                if len(temp_set) > len(current_set):
                    current_set = temp_set
                    current_size = len(current_set)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            continue
            
        # 1-Swap Move
        # Pick random x in current_set, random y not in current_set
        x = random.choice(list(current_set))
        y = random.choice([z for z in nums_list if z not in current_set])
        
        new_set = current_set - {x} | {y}
        
        # Check validity
        if is_valid_set(new_set):
            # Try to add more elements greedily
            improved = False
            for z in nums_list:
                if z not in new_set and can_add(new_set, z):
                    new_set.add(z)
                    improved = True
                    
            if len(new_set) > current_size:
                current_set = new_set
                current_size = len(current_set)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                no_improve_count = 0
            elif improved:
                # Same size but different composition, might unlock future additions
                current_set = new_set
                no_improve_count = 0
            else:
                # No improvement, but valid swap
                # Accept with some probability or just accept to explore
                current_set = new_set
                no_improve_count += 1
        else:
            no_improve_count += 1
            
    # Final validation
    if not is_valid_set(best_set):
        # Fallback to greedy result if somehow invalid (should not happen)
        S = set()
        for x in range(N, 1, -1):
            if can_add(S, x):
                S.add(x)
        best_set = S

    return sorted(list(best_set))
