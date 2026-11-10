import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # 1. Generate triples in {51, ..., 100} summing to 200
    large_range = list(range(51, 101))
    triples = []
    n = len(large_range)
    # Map value to index for quick lookup if needed, but we iterate
    # To find hitting set, we need to know which elements are in which triples
    element_triples = {x: [] for x in large_range}
    
    for i in range(n):
        a = large_range[i]
        for j in range(i+1, n):
            b = large_range[j]
            if a + b >= 200: continue
            c = 200 - a - b
            if c > b and c <= 100:
                # Found a triple (a, b, c)
                # c must be in large_range
                if c in element_triples:
                    triples.append((a, b, c))
                    element_triples[a].append((a, b, c))
                    element_triples[b].append((a, b, c))
                    element_triples[c].append((a, b, c))
                    
    # 2. Greedy Hitting Set
    remaining_triples = set(triples)
    removed = set()
    
    # Sort elements by number of triples they are in (descending)
    # Recalculate counts as we remove
    while remaining_triples:
        if time.time() > deadline - 5: break
        # Find element with max coverage
        best_elem = None
        max_count = 0
        # Only consider elements not yet removed
        for elem in large_range:
            if elem in removed: continue
            count = 0
            for t in remaining_triples:
                if elem in t:
                    count += 1
            if count > max_count:
                max_count = count
                best_elem = elem
        
        if best_elem is None or max_count == 0:
            break
            
        removed.add(best_elem)
        # Remove all triples containing best_elem
        remaining_triples = {t for t in remaining_triples if best_elem not in t}

    S_large = set(large_range) - removed
    
    # 3. Precompute pair and triple sums of S_large
    S_list = list(S_large)
    valid_sums = set()
    n_large = len(S_list)
    
    # Pairs
    for i in range(n_large):
        a = S_list[i]
        for j in range(i+1, n_large):
            b = S_list[j]
            valid_sums.add(a + b)
            
    # Triples
    for i in range(n_large):
        a = S_list[i]
        for j in range(i+1, n_large):
            b = S_list[j]
            s_ab = a + b
            for k in range(j+1, n_large):
                c = S_list[k]
                valid_sums.add(s_ab + c)
                
    # 4. Add small elements {1, ..., 50}
    S = set(S_large)
    for x in range(1, 51):
        if time.time() > deadline - 2: break
        target = 200 - x
        # Check if target is in valid_sums
        # Note: valid_sums contains sums of 2 and 3 distinct elements from S_large.
        # We need to ensure x itself is not in S_large (it isn't, ranges are disjoint).
        # We also need to ensure we don't form a sum with x + 1 element?
        # Min element in S_large is 51. x <= 50. x + 51 <= 101 < 200.
        # So x + 1 element is never 200.
        # So we only need to check if target is a sum of 2 or 3 elements from S_large.
        if target not in valid_sums:
            S.add(x)
            
    return list(S)
