import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute divisibility: div[s][a] = True if a divides s (for a < s)
    # We need: for a < b <= c, a does NOT divide b+c
    # So for each possible sum s = b+c, and each a < min(b,c), we need a !| s
    
    # Precompute: for each sum s in [2, 512], which a values (1..255) divide s?
    # divisors[s] = set of a in [1, 255] such that a | s
    divisors = [set() for _ in range(513)]
    for a in range(1, 256):
        for m in range(2, 513 // a + 1):
            s = a * m
            if s <= 512:
                divisors[s].add(a)
    
    # Base: upper tail [171, 256] is valid
    s = set(range(171, 257))
    s_list = sorted(s)
    s_set = set(s_list)
    
    def can_add_fast(x, elems, elem_set):
        """Check if x can be added to current set without violating the property.
        elems: sorted list of current elements
        elem_set: set of current elements for O(1) lookup
        """
        # Case 1: x is 'a' (smallest): need x !| (b+c) for all b,c in elems with x < b <= c
        for i, b in enumerate(elems):
            if b <= x:
                continue
            for j in range(i, len(elems)):
                c = elems[j]
                if c < b:
                    continue
                if x in divisors[b + c]:
                    return False
        
        # Case 2: x is 'b' (middle): need a !| (x+c) for all a in elems with a < x, c in elems with c >= x
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if a in divisors[x + c]:
                    return False
        
        # Case 3: x is 'c' (largest): need a !| (b+x) for all a,b in elems with a < b <= x
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if a in divisors[b + x]:
                    return False
        
        return True
    
    def greedy_add(order, initial_set):
        """Greedy: try to add elements in given order to initial_set. Returns final set."""
        current = set(initial_set)
        current_list = sorted(current)
        current_set = set(current_list)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if can_add_fast(x, current_list, current_set):
                current.add(x)
                current_set.add(x)
                # Insert into sorted list
                import bisect
                bisect.insort(current_list, x)
        
        return current
    
    # Try multiple strategies and keep the best
    best_result = set(range(171, 257))
    best_size = len(best_result)
    
    # Strategy 1: Descending order (original)
    order1 = list(range(170, 0, -1))
    result1 = greedy_add(order1, range(171, 257))
    if len(result1) > best_size:
        best_result = result1
        best_size = len(result1)
    
    # Strategy 2: Ascending order
    order2 = list(range(1, 171))
    result2 = greedy_add(order2, range(171, 257))
    if len(result2) > best_size:
        best_result = result2
        best_size = len(result2)
    
    # Strategy 3: Random orders (multiple tries)
    random.seed(42)
    candidates = list(range(1, 171))
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        random.shuffle(candidates)
        result_r = greedy_add(candidates, range(171, 257))
        if len(result_r) > best_size:
            best_result = result_r
            best_size = len(result_r)
    
    # Try to add remaining elements with a second pass using different orderings
    # For each element not in best_result, try adding it
    remaining = [x for x in range(1, 171) if x not in best_result]
    if remaining and time.time() < deadline - 5:
        best_list = sorted(best_result)
        best_set = set(best_list)
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add_fast(x, best_list, best_set):
                best_result.add(x)
                best_set.add(x)
                import bisect
                bisect.insort(best_list, x)
    
    return sorted(best_result)
