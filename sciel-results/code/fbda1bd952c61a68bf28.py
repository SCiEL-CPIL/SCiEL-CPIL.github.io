import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 180
    # Precompute all forbidden triples (a, b, c) with a < b, c = ab/(a+b), c < a, all distinct
    # For each pair (a, b) with a < b, check if c = ab/(a+b) is integer and c < a and c not in {a,b}
    forbidden = []  # list of (a, b, c) with c < a < b
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c < a and c != b and c != a:
                    # c < a < b, all distinct
                    forbidden.append((a, b, c))
    
    # We want to find a large subset S of [1,180] with no forbidden triple fully contained in S.
    # This is the maximum independent set in a 3-uniform hypergraph.
    # Use greedy: start with all elements, iteratively remove the element in the most remaining forbidden triples.
    
    # Build incidence: for each element, list of triples containing it
    from collections import defaultdict
    elem_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(forbidden):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Greedy approach: repeatedly remove the element that appears in the most currently-active triples
    # A triple is "active" if all three of its elements are still in the set.
    
    def greedy_solve():
        in_set = [True] * (n + 1)  # 1-indexed
        # Count active triples per element
        # For each element, count how many triples it's in where all 3 elements are in_set
        triple_active = [True] * len(forbidden)
        
        # Precompute for each triple the 3 elements
        triple_elems = [(a, b, c) for (a, b, c) in forbidden]
        
        # Initial count: all triples are active
        elem_active_count = [0] * (n + 1)
        for idx, (a, b, c) in enumerate(forbidden):
            elem_active_count[a] += 1
            elem_active_count[b] += 1
            elem_active_count[c] += 1
        
        removed = set()
        while True:
            if time.time() > deadline:
                break
            # Find element with max active count that is still in set
            best_elem = -1
            best_count = 0
            for x in range(1, n + 1):
                if in_set[x] and elem_active_count[x] > best_count:
                    best_count = elem_active_count[x]
                    best_elem = x
            
            if best_elem == -1 or best_count == 0:
                break
            
            # Remove best_elem
            in_set[best_elem] = False
            removed.add(best_elem)
            
            # Update: for each triple containing best_elem that was active, mark it inactive
            for idx in elem_to_triples[best_elem]:
                if triple_active[idx]:
                    a, b, c = triple_elems[idx]
                    # Check if all three are still in set (before removal, they were; now best_elem is out)
                    # So this triple becomes inactive
                    triple_active[idx] = False
                    for x in (a, b, c):
                        if in_set[x]:
                            elem_active_count[x] -= 1
                    # best_elem is no longer in set, so don't decrement its count further
                    # Actually we should decrement for all three, but best_elem's count doesn't matter since it's removed
                    # Let's just decrement for the other two
                    # Wait, we already checked in_set[x] which is False for best_elem now
                    # So the loop above handles it correctly
        
        # Return the set
        result = [x for x in range(1, n + 1) if in_set[x]]
        return result
    
    # Run greedy multiple times with different random orderings to find a better solution
    best_result = None
    best_size = 0
    
    # First, try the basic greedy
    result = greedy_solve()
    best_result = result
    best_size = len(result)
    
    # Try randomized greedy: break ties randomly
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        in_set = [True] * (n + 1)
        triple_active = [True] * len(forbidden)
        triple_elems = [(a, b, c) for (a, b, c) in forbidden]
        
        elem_active_count = [0] * (n + 1)
        for idx, (a, b, c) in enumerate(forbidden):
            elem_active_count[a] += 1
            elem_active_count[b] += 1
            elem_active_count[c] += 1
        
        while True:
            if time.time() > deadline - 5:
                break
            # Find all elements with max active count
            max_count = 0
            for x in range(1, n + 1):
                if in_set[x] and elem_active_count[x] > max_count:
                    max_count = elem_active_count[x]
            
            if max_count == 0:
                break
            
            candidates = [x for x in range(1, n + 1) if in_set[x] and elem_active_count[x] == max_count]
            best_elem = random.choice(candidates)
            
            in_set[best_elem] = False
            
            for idx in elem_to_triples[best_elem]:
                if triple_active[idx]:
                    a, b, c = triple_elems[idx]
                    triple_active[idx] = False
                    for x in (a, b, c):
                        if in_set[x]:
                            elem_active_count[x] -= 1
        
        result = [x for x in range(1, n + 1) if in_set[x]]
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    if not best_result:
        best_result = list(range(91, 181))
    
    return best_result
