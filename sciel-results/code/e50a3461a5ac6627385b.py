import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    N = 128
    
    # Start with the safe base: {65, ..., 128}
    base = set(range(65, N + 1))
    
    def is_valid(S, candidate):
        """Check if adding candidate to S maintains validity."""
        S_new = S | {candidate}
        elems = list(S)
        n = len(elems)
        
        # 1. Check triples where candidate is a member: (candidate, x, y)
        for i in range(n):
            x = elems[i]
            s1 = candidate + x
            if s1 > N:
                continue
            if s1 not in S_new:
                continue
            for j in range(i + 1, n):
                y = elems[j]
                s2 = candidate + y
                s3 = x + y
                if s2 in S_new and s3 in S_new:
                    return False
        
        # 2. Check triples where candidate is a sum: (a, b, c) with a+b == candidate
        for i in range(n):
            a = elems[i]
            if a >= candidate:
                break
            b = candidate - a
            if b <= a or b not in S:
                continue
            # Found pair (a, b) with a+b == candidate
            # Check if there exists c in S, c != a, c != b, such that a+c in S and b+c in S
            for c in elems:
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
                            
        return True
    
    def is_valid_after_swap(S, remove_elem, add_elem):
        """Check if swapping remove_elem out and add_elem in maintains validity."""
        if add_elem in S or remove_elem not in S:
            return False
        S_new = (S - {remove_elem}) | {add_elem}
        elems = list(S_new)
        n = len(elems)
        
        # Check all triples in S_new for violations
        # Only need to check triples involving add_elem or pairs whose sum was remove_elem
        # But for safety, check triples involving add_elem as member or sum
        # Triples not involving add_elem and not having remove_elem as a sum are unchanged
        
        # Triples where add_elem is a member: (add_elem, x, y)
        for i in range(n):
            x = elems[i]
            if x == add_elem:
                continue
            s1 = add_elem + x
            if s1 > N or s1 not in S_new:
                continue
            for j in range(i + 1, n):
                y = elems[j]
                if y == add_elem:
                    continue
                s2 = add_elem + y
                s3 = x + y
                if s2 in S_new and s3 in S_new:
                    return False
        
        # Triples where add_elem is a sum: (a, b, c) with a+b == add_elem
        for i in range(n):
            a = elems[i]
            if a >= add_elem:
                break
            b = add_elem - a
            if b <= a or b not in S_new:
                continue
            for c in elems:
                if c == a or c == b:
                    continue
                if (a + c) in S_new and (b + c) in S_new:
                    return False
        
        # Also need to check: triples that previously were valid but now have remove_elem gone
        # Removing an element can't create new violations, so we only need to check
        # triples that involve add_elem. Triples not involving add_elem are a subset of
        # the old S minus remove_elem, which was valid.
        
        return True
    
    # Greedy: try to add elements from 1 to 64
    S = set(base)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid(S, x):
            S.add(x)
    
    best_S = set(S)
    rng = random.Random(42)
    
    # Phase 1: Multiple random greedy constructions
    for trial in range(100):
        if time.time() > deadline:
            break
        
        S_try = set(base)
        small_elements = list(range(1, 65))
        rng.shuffle(small_elements)
        
        for x in small_elements:
            if time.time() > deadline:
                break
            if is_valid(S_try, x):
                S_try.add(x)
        
        if len(S_try) > len(best_S):
            best_S = set(S_try)
    
    # Phase 2: Local search with swaps on best_S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_try = set(best_S)
        
        # Try all possible swaps: remove one element, add another
        in_S = list(S_try)
        not_in_S = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(not_in_S)
        rng.shuffle(in_S)
        
        for remove_elem in in_S:
            if time.time() > deadline:
                break
            for add_elem in not_in_S:
                if time.time() > deadline:
                    break
                if is_valid_after_swap(S_try, remove_elem, add_elem):
                    S_try.discard(remove_elem)
                    S_try.add(add_elem)
                    improved = True
                    # After a swap, restart checking
                    break
            if improved:
                break
        
        if improved and len(S_try) >= len(best_S):
            best_S = set(S_try)
    
    # Phase 3: Try to add more elements to best_S
    S_try = set(best_S)
    missing = [x for x in range(1, N + 1) if x not in S_try]
    rng.shuffle(missing)
    for x in missing:
        if time.time() > deadline:
            break
        if is_valid(S_try, x):
            S_try.add(x)
    if len(S_try) > len(best_S):
        best_S = set(S_try)
    
    # Phase 4: More random restarts with swap-based local search
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Start from a random greedy construction
        S_try = set(base)
        small_elements = list(range(1, 65))
        rng.shuffle(small_elements)
        
        for x in small_elements:
            if time.time() > deadline:
                break
            if is_valid(S_try, x):
                S_try.add(x)
        
        # Apply swap local search
        local_improved = True
        while local_improved and time.time() < deadline:
            local_improved = False
            in_S = list(S_try)
            not_in_S = [x for x in range(1, N + 1) if x not in S_try]
            rng.shuffle(not_in_S)
            rng.shuffle(in_S)
            
            for remove_elem in in_S:
                if time.time() > deadline:
                    break
                for add_elem in not_in_S:
                    if time.time() > deadline:
                        break
                    if is_valid_after_swap(S_try, remove_elem, add_elem):
                        S_try.discard(remove_elem)
                        S_try.add(add_elem)
                        local_improved = True
                        break
                if local_improved:
                    break
        
        if len(S_try) > len(best_S):
            best_S = set(S_try)
    
    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]
