import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    def can_add(S, x):
        """Check if S U {x} is valid."""
        # Case 1: x is a member of the triple.
        # Need b, c in S, b != c, such that x+b in S, x+c in S, and (b+c in S or b+c == x).
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            b = S_list[i]
            if (x + b) not in S:
                continue
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                if (x + c) not in S:
                    continue
                if (b + c) in S or (b + c) == x:
                    return False
        
        # Case 2: x is a sum of two members.
        # Need a, b, c in S, distinct, such that a+b=x, a+c in S, b+c in S.
        for a in S:
            b = x - a
            if b == a or b not in S:
                continue
            # Check if there exists c in S, c!=a, c!=b, such that a+c in S and b+c in S
            for c in S:
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
        
        return True

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    def greedy_construct(order):
        """Greedy construction given a fixed order of candidates."""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_passes=20):
        """Improve S via additions and systematic swaps."""
        S = set(S)
        not_in_S = [x for x in range(1, 129) if x not in S]
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S[:]:
                if time.time() > deadline:
                    break
                
                # Try to add x directly
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    improved = True
                    update_best(S)
                    continue
                
                # Try swaps: remove y from S, add x
                S_list = list(S)
                random.shuffle(S_list)
                
                found_swap = False
                for y in S_list[:15]:
                    if time.time() > deadline:
                        break
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        S = S_temp
                        S.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    # Re-check additions after swap
                    break  # Restart the pass
            
            if not improved:
                break
        
        return S

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)
    
    # Ascending
    S = greedy_construct(list(range(1, 129)))
    update_best(S)
    
    # Descending
    S = greedy_construct(list(range(128, 0, -1)))
    update_best(S)
    
    # Even/odd interleaved
    evens = list(range(2, 129, 2))
    odds = list(range(1, 129, 2))
    interleaved = []
    for i in range(64):
        if i < len(odds):
            interleaved.append(odds[i])
        if i < len(evens):
            interleaved.append(evens[i])
    S = greedy_construct(interleaved)
    update_best(S)
    
    # Several random orders
    for i in range(30):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        update_best(S)
    
    # Phase 2: Local search from best set
    if time.time() < deadline:
        S = local_search(best_set, max_passes=30)
        update_best(S)
    
    # Phase 3: More random restarts with local search
    for i in range(40):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        S = local_search(S, max_passes=15)
        update_best(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
    
    # One more local search pass on the final best set
    if time.time() < deadline and best_size < 128:
        S = local_search(best_set, max_passes=10)
        update_best(S)
    
    result = [int(x) for x in sorted(best_set)]
    return result
