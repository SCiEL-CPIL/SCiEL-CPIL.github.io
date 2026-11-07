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

    def can_add(S_set, S_list, x):
        """
        Check if x can be added to S without creating a conflict.
        Assumes x not in S.
        S_set is a set for O(1) lookup if needed, S_list for iteration.
        """
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j: continue
                b = S_list[j]
                # Check a | x*b
                if a in divs[x * b]:
                    return False
                # Check x | a*b
                if x in divs[a * b]:
                    return False
        return True

    # 1. Greedy Initialization
    best_set = set()
    best_size = 0
    
    # Try multiple greedy orders
    orders = [
        list(range(N, 1, -1)), # Descending
        list(range(2, N + 1)), # Ascending
        list(range(N, 1, -1, 2)), # Odd descending
        list(range(N, 1, -1, 2)), # Even descending
        [x for x in range(N, 1, -1) if x % 2 != 0] + [x for x in range(N, 1, -1) if x % 2 == 0]
    ]
    
    for order in orders:
        S = set()
        S_list = []
        for x in order:
            if x not in S and can_add(S, S_list, x):
                S.add(x)
                S_list.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # 2. Local Search
    current_set = best_set.copy()
    current_size = len(current_set)
    all_numbers = set(numbers)
    
    nums_list = list(all_numbers)
    random.shuffle(nums_list)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < DEADLINE:
        if no_improve_count > max_no_improve:
            # Kick: Remove 2, Add 2
            if len(current_set) >= 2:
                items_to_remove = random.sample(list(current_set), 2)
                temp_set = current_set - set(items_to_remove)
                temp_list = list(temp_set)
                
                candidates = [x for x in nums_list if x not in temp_set]
                random.shuffle(candidates)
                
                added = 0
                for y in candidates:
                    if added == 2: break
                    if can_add(temp_set, temp_list, y):
                        temp_set.add(y)
                        temp_list.append(y)
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
        x = random.choice(list(current_set))
        y_candidates = [z for z in nums_list if z not in current_set]
        if not y_candidates:
            break
        y = random.choice(y_candidates)
        
        # Check if swap is valid
        temp_set = current_set - {x}
        temp_list = list(temp_set)
        
        if can_add(temp_set, temp_list, y):
            new_set = temp_set | {y}
            new_list = temp_list + [y]
            
            if len(new_set) > current_size:
                current_set = new_set
                current_size = len(current_set)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                no_improve_count = 0
            else:
                # Same size, accept to explore
                current_set = new_set
                no_improve_count += 1
        else:
            no_improve_count += 1

    # Final validation
    if not is_valid_set(best_set):
        S = set()
        S_list = []
        for x in range(N, 1, -1):
            if can_add(S, S_list, x):
                S.add(x)
                S_list.append(x)
        best_set = S

    return sorted(list(best_set))
